"""
Driver for the examples: train along the windows, detect, refine, certify, save.

    from examples.configs import EXAMPLES
    from examples.core import run_example
    result = run_example(EXAMPLES["disk"].quick(), seed=0, out_dir="examples_out")

Everything numerical is done by `pinn_eigen` (train_one, evaluate_residual, local_minima,
a_posteriori, kato_temple, temple_bracket); this module only organizes the runs.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

import pinn_eigen as pe
from examples.configs import Example


# ── training along one chain of nodes ───────────────────────────────────────────────────────────
def make_model(ex: Example) -> torch.nn.Module:
    return pe.EigenFunctionNN(dim=ex.domain.dim, boundary_fn=ex.domain.boundary,
                              hidden_size=ex.hidden, n_hidden=ex.n_hidden)


def _fresh(ex: Example, i: int) -> bool:
    if ex.start == "warm":
        return i == 0
    if ex.start == "cold":
        return True
    return i % int(ex.start) == 0


def normalized_loss(ex: Example, ev: dict) -> float:
    """r^2 = mean(res^2)/mean(u^2) for p = 2; Lambda_p / (int|u|^p)^{2(p-1)/p} for p != 2."""
    if ex.p == 2.0:
        return ev["r"] ** 2
    return ev["loss_res"] / ev["mass"] ** (2.0 * (ex.p - 1.0) / ex.p)


def train_chain(ex: Example, Es: List[float], init: Optional[torch.nn.Module] = None, verbose: bool = True):
    """Train on the nodes Es (in order). Returns (records, state_dicts).
    `init`: network to start from (warm start from another run); otherwise a fresh one."""
    records, states = [], []
    model = None
    for i, E in enumerate(Es):
        if i == 0 and init is not None:
            model = make_model(ex)
            model.load_state_dict(init.state_dict())
        elif model is None or _fresh(ex, i):
            model = make_model(ex)
        sched = pe.cosine_lr(ex.n_epochs, eta_min=1e-5) if ex.lr_decay else None
        res, norm, info = pe.train_one(model, E, ex.domain.sampler, ex.V, n_samples=ex.n_samples,
                                       n_epochs=ex.n_epochs, lr=ex.lr, mu=ex.mu, p=ex.p,
                                       patience=ex.patience, tol=ex.tol, lr_scheduler=sched,
                                       vol=ex.vol, return_info=True)
        ev = pe.evaluate_residual(model, ex.domain.sampler, E, ex.V, p=ex.p,
                                  n_samples=ex.eval_samples, vol=ex.vol)
        rec = dict(E=float(E), loss_res=res, norm_err=norm, epochs_run=info["epochs_run"],
                   wall_time=info["wall_time"], eval_loss_res=ev["loss_res"], mass=ev["mass"],
                   r=ev["r"], lam_n=normalized_loss(ex, ev))
        records.append(rec)
        states.append({k: v.clone() for k, v in model.state_dict().items()})
        if verbose:
            print(f"  E={E:9.4f}  recorded residual={res:.4e}  normalized loss={rec['lam_n']:.4e}  "
                  f"mass={rec['mass']:.3f}  epochs={info['epochs_run']:5d}  t={info['wall_time']:.1f}s",
                  flush=True)
    return records, states


def _load(ex: Example, state: dict) -> torch.nn.Module:
    m = make_model(ex)
    m.load_state_dict(state)
    return m


# ── refinement and certificate ─────────────────────────────────────────────────────────────────
def nearest_reference(ex: Example, E: float):
    if not ex.reference:
        return None
    return min(ex.reference, key=lambda t: abs(t[0] - E))


def refine_and_certify(ex: Example, model: torch.nn.Module, E_node: float) -> dict:
    """Closed-form refinement of the frozen network + a posteriori quantities (Section 3.1)."""
    ev = pe.a_posteriori(model, ex.domain.sampler, ex.V, E_node, p=ex.p, vol=ex.vol,
                         n_chunks=ex.cert_chunks, chunk=ex.cert_chunk)
    out = dict(E_node=float(E_node), **{k: (list(v) if isinstance(v, tuple) else v) for k, v in ev.items()})
    E_hat = ev["rho_res"] if ex.p == 2.0 else ev["E_pinn"]
    out["E_hat"] = E_hat
    ref = nearest_reference(ex, E_hat)
    if ref is not None:
        out.update(E_ref=ref[0], ref_mult=ref[1], ref_label=ref[2],
                   rel_diff=(E_hat - ref[0]) / ref[0])
        if ex.p == 2.0:
            lo, hi = ev["weinstein"]
            out["weinstein_contains_ref"] = bool(lo <= ref[0] <= hi)
    if ex.p == 2.0 and ex.kato_temple and ex.reference:
        # gap = distance from E_hat to the rest of the (reference) spectrum; NOT certified by the scan
        others = [e for e, _, _ in ex.reference if abs(e - ref[0]) > 1e-9]
        if others:
            gamma = min(abs(E_hat - e) for e in others)
            out["kato_temple_gap"] = gamma
            out["kato_temple_bound"] = pe.kato_temple(ev["rho_res"], ev["eta"], gamma)
        if ref is not None and abs(ref[0] - ex.reference[0][0]) < 1e-9 and len(ex.reference) > 1:
            out["temple_bracket_E1"] = list(pe.temple_bracket(ev["rho_res"], ev["eta"], ex.reference[1][0]))
    return out


def parabola_vertex(Es, L, i):
    """Vertex of the parabola through nodes i-1, i, i+1 (E_curve of Section 5)."""
    c = np.polyfit(Es[i - 1:i + 2], L[i - 1:i + 2], 2)
    return float(-c[1] / (2 * c[0])) if c[0] > 0 else float("nan")


# ── one example ────────────────────────────────────────────────────────────────────────────────
def run_example(ex: Example, seed: int = 0, out_dir="examples_out", verbose: bool = True,
                plots: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()
    windows, detections = [], []
    for w, Es in enumerate(ex.windows):
        if verbose:
            print(f"[{ex.name}] window {w + 1}/{len(ex.windows)}: E = {Es[0]} ... {Es[-1]} ({len(Es)} nodes)",
                  flush=True)
        recs, states = train_chain(ex, Es, verbose=verbose)
        windows.append(dict(E=list(Es), records=recs))
        curve = [r[ex.detect_on] for r in recs]
        if ex.mode == "windows":
            picks = [int(np.argmin([r["lam_n"] for r in recs]))]
        else:
            picks = pe.local_minima(curve, ex.threshold)
        for i in picks:
            det = dict(window=w, node=i)
            if ex.local_h:      # p-Laplacian: local three-node scan around the coarse minimum
                det["E_curve"] = parabola_vertex(np.array(Es), np.array(curve), i) if 0 < i < len(Es) - 1 else float("nan")
                h = ex.local_h
                locE = [round(Es[i] - h, 6), round(Es[i], 6), round(Es[i] + h, 6)]
                if verbose:
                    print(f"  local scan around E={Es[i]} : {locE}", flush=True)
                lrecs, lstates = train_chain(ex, locE, init=_load(ex, states[i]), verbose=verbose)
                j = int(np.argmin([r["lam_n"] for r in lrecs]))
                det.update(local=dict(E=locE, records=lrecs), local_node=j)
                det.update(refine_and_certify(ex, _load(ex, lstates[j]), locE[j]))
                det["_state"] = lstates[j]
            else:
                det.update(refine_and_certify(ex, _load(ex, states[i]), Es[i]))
                det["_state"] = states[i]
            detections.append(det)
    wall = time.perf_counter() - t0

    result = dict(example=ex.name, paper=ex.paper, section=ex.section, seed=seed,
                  settings=_settings(ex), windows=windows,
                  detections=[{k: v for k, v in d.items() if k != "_state"} for d in detections],
                  reference=[list(t) for t in ex.reference], reference_kind=ex.reference_kind,
                  paper_values=ex.paper_values, notes=ex.notes, wall_time_s=wall,
                  threads=torch.get_num_threads(), torch=torch.__version__)
    path = out_dir / f"{ex.name}_s{seed}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=1, default=float)
    result["json"] = str(path)
    if plots:
        from examples import plots as P
        result["figures"] = P.save_all(ex, result, [d["_state"] for d in detections], out_dir, seed)
    if verbose:
        print_summary(ex, result)
    return result


def _settings(ex: Example) -> dict:
    d = asdict(ex)
    for k in ("domain", "V", "reference", "windows", "paper_values", "notes"):
        d.pop(k, None)
    d.update(domain=ex.domain.name, dim=ex.domain.dim, vol_in_loss=ex.vol,
             n_nodes=sum(len(w) for w in ex.windows))
    return d


# ── printing ───────────────────────────────────────────────────────────────────────────────────
def print_summary(ex: Example, result: dict) -> None:
    print(f"\n== {ex.name}: {ex.paper} (Section {ex.section}); {ex.domain.name}, {ex.potential}, p={ex.p}")
    print(f"   wall time {result['wall_time_s']:.1f} s; results in {result['json']}")
    if not result["detections"]:
        print("   no minimum detected")
    elif ex.p == 2.0:
        print(f"   {'node E':>9} {'E_hat':>11} {'+-se':>8} {'r(node)':>8} {'eta':>8}  "
              f"{'Weinstein [rho-eta, rho+eta]':>28} {'reference':>10} {'rel.diff':>10}  extra")
        for d in result["detections"]:
            lo, hi = d["weinstein"]
            ref = f"{d['E_ref']:.4f}" if "E_ref" in d else "-"
            rd = f"{d['rel_diff']:+.2e}" if "rel_diff" in d else "-"
            extra = []
            if "weinstein_contains_ref" in d:
                extra.append("ref in enclosure" if d["weinstein_contains_ref"] else "ref NOT in enclosure")
            if "kato_temple_bound" in d:
                extra.append(f"KT |E-E_k|<={d['kato_temple_bound']:.1e} (gap {d['kato_temple_gap']:.2f}, from reference)")
            if "temple_bracket_E1" in d:
                a, b = d["temple_bracket_E1"]
                extra.append(f"Temple {a:.4f}<=E1<={b:.4f}")
            print(f"   {d['E_node']:9.4f} {d['E_hat']:11.5f} {d['rho_res_se']:8.1e} {d['r']:8.3f} {d['eta']:8.3f}  "
                  f"[{lo:12.4f},{hi:12.4f}] {ref:>10} {rd:>10}  {'; '.join(extra)}")
    else:
        print(f"   {'node E':>9} {'E_PINN':>10} {'E_curve':>9} {'Lam_n^1/2':>9} {'reference':>10} {'rel.diff':>10}  label")
        for d in result["detections"]:
            ref = f"{d['E_ref']:.4f}" if "E_ref" in d else "-"
            rd = f"{d['rel_diff']:+.2e}" if "rel_diff" in d else "-"
            print(f"   {d['E_node']:9.4f} {d['E_hat']:10.4f} {d.get('E_curve', float('nan')):9.3f} "
                  f"{d['Lam_n'] ** 0.5:9.3f} {ref:>10} {rd:>10}  {d.get('ref_label', '')}")
    print(f"   paper: {ex.paper_values}")
    if ex.notes:
        print(f"   notes: {ex.notes}")
