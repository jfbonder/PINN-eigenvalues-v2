# Theory notes for the coordinator (round-2 revision, Reviewer #1 points 1, 3, 5, 6)

Files in this folder
- `theory_fragments.tex` — paste-ready amsart fragments (Fragments 0–7 + new bibitems).
  Not compiled (no LaTeX in the container); brace/environment balance checked by script.
  Preamble additions needed: `proposition`, `lemma`, `corollary` theorem environments, `\dist`.
- `radial_plap_shoot.py` — shooting code for the radial p-Laplacian ODE (reference values below).
- Numerical sanity checks done: 20 000 random instances of Prop. `prop.stability` (c),(d) and of
  the parabolic-fit bound (0 violations); closed formula of Prop. `prop.penalized`(a) vs brute force.

Notation: `d(E)=dist(E,σ)`, `Λ(E)=d(E)²`; grid `E^i=E_*+(i−1)h`, `i=1..J` (paper calls the count `j`);
`ϵ` = detection threshold, `ε` = early-stopping tolerance (paper's clash is kept apart).

---------------------------------------------------------------------------------------------------
## What each fragment answers

| Fragment | Content | Reviewer point |
|---|---|---|
| 0 | Standing assumptions/notation (L self-adjoint, compact resolvent) | — |
| 1 | Prop. `prop.penalized`: exact loss curve of the *penalized* loss, minimizers, all critical points; Cor. `cor.mu`; Remark `rem.mu` (μ rule) | R1.3, R1.6 |
| 2 | Prop. `prop.stability` (a)–(e) + Remarks (interpretation, multiplicity, clusters, what is not proved, consistency of network class) | R1.6 |
| 3 | Prop. `prop.rayleigh`: Λ(u,E)=(E−ρ(u))²+η(u)², Weinstein, Kato–Temple; Remark `rem.certificate` (certificate, MC variance/bias) | R1.6 (+ old R1.7 "large residual vs small error") |
| 4 | Prop. `prop.parabola`: 3-point vertex formula with error bound | R1.1, R1.6 |
| 5 | Cost model, scan vs Rayleigh+deflation | R1.1 |
| 6 | p-Laplacian: Prop. `prop.plap` (zeros of the loss curve = spectrum, proved), Lemma `lem.critical`, error sources, radial benchmark | R1.5 |
| 7 | Correction to the appendix proof | — |

---------------------------------------------------------------------------------------------------
## Main mathematical findings (all proved in the fragments)

1. **Penalized loss, exact formula.** `inf_u [‖(L−E)u‖² + μ(‖u‖²−1)²] = d² − d⁴/(4μ)` if `d² ≤ 2μ`, `= μ`
   otherwise. Minimizer has `‖u‖² = 1 − d²/(2μ)`; it is `u=0` iff `d² ≥ 2μ`.
   The *residual term alone* at the minimizer is `d²(1 − d²/(2μ))₊`: increasing in d only while
   `d² ≤ μ`, and **equal to 0 when the network has collapsed** (`d² ≥ 2μ`).
   Since `sweep()` records `loss_res` (the residual term, not the total loss and not the quotient),
   a too small μ creates *spurious zeros/minima of the recorded curve*. Sharp thresholds, with
   `D = max_{window} d(E)` (half the largest gap, or distance of a window endpoint to the spectrum):
   - `μ ≥ D²`  ⇔ recorded curve is a monotone function of d (faithful);
   - `D²/2 ≤ μ < D²` : spurious local minima at mid-gap;
   - `μ < D²/2` : collapse `u=0`, spurious zeros.
   (The task statement suggested "μ > (max gap/2)²" — that is exactly `μ > D²`, now proved; the
   collapse threshold itself is `D²/2`.)
   The normalized quotient `Λ(u,E)=loss_res/mean(u²)` equals `d²` for every μ ⇒ recommend recording it.
2. **No spurious local minima in function space** (Prop. `prop.penalized`(c)): every local minimizer of
   the penalized loss is global; other critical points are saddles. So spurious minima can only come
   from the NN parametrization / optimizer / sampling. Good direct answer to R1.6.
3. **Stability proposition.** Under `|Λ̃_i − d(E^i)²| ≤ δ`:
   (a) `Λ̃_i ≤ ϵ ⇒ d(E^i) ≤ √(ϵ+δ)`; (b) each eigenvalue in the window has a node with
   `Λ̃ ≤ h²/4+δ`; (c) *every* discrete local minimum (no threshold needed, no gap condition) has
   `d ≤ max{h, h/2+δ/h}`; (d) under the gap condition `γ_k > 2τ+h`, `τ=√(h²/4+2δ)`, each eigenvalue
   has a discrete local minimum within τ; (e) exact count by grouping detected nodes at distance
   ≤ 2ρ, when eigenvalues are > 4ρ apart, `ρ=√(ϵ+δ)`. One-sided version stated.
   Corollary of (c): refining h below `√(2δ)` is useless.
   Discrete local minimum is defined with `≤` (the code uses strict `<`; ties are non-generic. With
   strict inequalities (d) could fail only in case of exact ties).
4. **Rayleigh quotient / Kato–Temple.** For fixed u, `E ↦ Λ(u,E)` is exactly the parabola
   `(E−ρ(u))²+η(u)²`. Hence `refine()` (which minimizes this over E with frozen network) returns the
   Rayleigh quotient of the trained network, and `|ρ(u)−E_k| ≤ η²/γ` (γ = gap): eigenvalue error is
   *quadratic* in the residual. This is the rigorous explanation of "accurate eigenvalue, large
   residual" and should replace the v3 heuristic "a nearly constant floor η does not shift the
   minimizer" (which is not an argument: η(E) need not be constant).
5. **Parabolic 3-point formula** `Ê = E^m + (h/2)(y₋−y₊)/(y₋−2y₀+y₊)`: exact for `(E−E_k)²+c`, error
   `≤ (4|e|+h)ω/(2h²−4ω)` (≤ 3ω/h if central node is the nearest and ω ≤ h²/4), where ω is the
   *oscillation* of the error over the 3 nodes (a constant floor cancels).
6. **p-Laplacian: zeros of the loss curve are exactly the eigenvalues** (Prop. `prop.plap`(d)), for
   `p ≥ 2d/(d+2)` (all p>1 in 2D; p ≥ 6/5 in 3D). Proof by compactness + continuity of `(−Δ_p)^{-1}`.
   I believe this is new in this context and it is the right replacement for the missing lower
   bound; there is still no *quantitative* bound (no Weinstein analogue) — stated honestly.
   Also: for fixed u the refined value is the least-squares quotient
   `ρ*_p(u) = −∫Δ_p u |u|^{p−2}u / ∫|u|^{2(p−1)}`, which is NOT stationary at eigenfunctions for p≠2
   ⇒ first-order accuracy only (vs second order for p=2). Recommend also reporting the variational
   quotient `∫|∇u|^p/∫|u|^p` (second order; upper bound for E_1).
7. **Lemma `lem.critical`**: for p>2 any C² function has `Δ_p u = 0` at its critical points; hence
   eigenfunctions are not C² at nonzero critical points and a smooth network has an irreducible
   pointwise residual `E|u|^{p−2}u` at each of its critical points (e.g. at the maximum).

---------------------------------------------------------------------------------------------------
## Problems found in the current manuscript / code (please read)

**P1. p-Laplacian results are probably for a different problem (confirms coordinator issue 1, with
numbers).** `loss_fn` uses residual `Δ_p u + E·u` and normalization `mean(u²)=1`. That is the
non-homogeneous problem `−Δ_p u = E u`, whose "eigenvalue" scales as `E ↦ |t|^{p−2}E` under `u ↦ tu`.
Radial shooting (`radial_plap_shoot.py` + variants) gives for the first eigenvalue of the unit ball:

| (p,d) | true `E_1` (homogeneous problem) | code problem (`−Δ_p u=Eu`, `mean u²=1`) | paper v3 |
|---|---|---|---|
| (2.2, 2) | 6.5317 (Horák: 6.5320) | 6.9350 | 6.6793 (Table 5) |
| (2.5, 3) | 14.1112 | 17.5899 | 17.4513 (caption of Fig. `loss.plap.disk.R3.fig`) |
| (2.2, 3) | 11.4699 | 12.4389 | — |
| (2.5, 2) | 7.7102 | 9.0965 | — |

The 3D value printed in the paper (17.4513) is within 0.8 % of the *wrong-problem* value 17.59 and
24 % away from the true `E_1(B_1)=14.11`. So the 3D p-Laplacian numbers in v3 are almost certainly
not eigenvalues of the p-Laplacian. For 2D, 6.679 lies between 6.53 and 6.935, so I cannot tell which
code produced Table 5; the 2–6 % "errors" of Tables 5–6 may be partly modelling error, not numerical
error. Table 6, row 2 (15.144490 vs 15.144, rel. error 3e-5) looks too good relative to row 1 (6.7 %)
— worth re-running. Action: fix the code (`E|u|^{p−2}u`, `mean(|u|^p)`), rerun, and use the radial
values above as an independent benchmark in 2D and 3D (Remark `rem.radial`). Second radial
eigenvalue for (2.2,2) is 41.789; the unidentified third row of Table 5 (37.62) matches nothing I
can compute — treat as "candidate" (the text of Fragment 6 says so).
Note: Fragment 6 Remark `rem.plap.errors`(3) says "in our implementation a safeguard 1e-12 is added
to |∇u|" (true for `_p_laplacian`).

**P2. Table 1, row (1,3) violates the Weinstein bound.** Residual `1.40e-3` ⇒ `r=0.037`, but the
reported eigenvalue is 0.102 away from `10π²` (the nearest eigenvalue). Weinstein is a theorem for any
`u∈D(L)`, so the residual entry (or the eigenvalue) is wrong, or `mean(u²)` was far below 1 (the
"Residual" column is not divided by `mean u²`). rev-2 had 1.40e-4, which is worse. All other rows of
Tables 1–4 are consistent with Weinstein (r = sqrt of the column).

**P3. Reported eigenvalues are less accurate than Kato–Temple predicts.** If the refined value were
the exact Rayleigh quotient, `|Ê−E_k| ≤ Residual/γ`. Examples: Table 2 row (0,1): 0.086/10.3 = 0.008
but observed error 0.0154; Table 1 row (1,2): 0.116/29.6 = 0.004 vs observed 0.037. Likely cause:
`refine()` calls `sampler(n_samples)` inside the objective, so `minimize_scalar` minimizes a *noisy*
function; the parabola is flat within noise ν over a width √ν (≈ 0.03–0.05 here). Fix: compute the
closed form `Ê = Σ u·Lu / Σ u²` on one fixed large sample (or freeze the sample in the objective).
This should improve the linear tables by about an order of magnitude at zero training cost — worth
asking the experiments agent to check. (Fragment 3, Remark `rem.certificate`(4) mentions this.)

**P4. Detection curve.** `sweep()` records `loss_res` only. By finding 1 this is fine iff `μ ≥ D²`;
otherwise use `loss_res/mean(u²)` and flag nodes with `mean(u²) < 1/2`.

**P5. Appendix proof.** `(cota.Lambda2)` is wrong as written: with `ε=1/(4E)` one gets
`Λ ≥ ½∫(Δu)² − E²` (not `+E²−E`). Conclusion unaffected. The H² bound from `‖Δu‖₂` needs Ω convex or
C^{1,1}; `E ≥ 0` is unnecessary. See Fragment 7.

**P6. Sentence in Sec. 2**: "The minima of Λ(E) correspond exactly to the eigenvalues" – fine, but
"zeroes ... correspond with eigenvalues E^i" (grid nodes are never exactly eigenvalues) should be
rephrased using Prop. `prop.stability`.

**P7. "μ ∝ E²" claim vs values used.** The correct statement (Remark `rem.mu`): two lower bounds,
`μ ≥ D²` (spectral; all that is needed with warm start) and `μ > q_0(E)/2 ≈ E²/2` (only for cold
starts / the periodic re-initialization `reset_every`), plus an upper guideline `μ ≲ Nϵ` from the MC
floor `μ·Var(u²)/N` of the penalty. Rough check of the values used (windows guessed from the
commented parameter blocks, starting at E=0 — please correct with the actual windows):

| Example | μ used | D² (needed) | 2D² (recommended) | E*² (cold-start scale) |
|---|---|---|---|---|
| disk, [0,~32] | 100 | 34 | 68 | ~1000 |
| square, [44,55] | 3000 | 32 | 64 | 3025 |
| square, [0,105] (fig. file says μ=450) | 450 | 390 (d(0)=2π²) | 779 | 11000 |
| ball 3D, [0,~70] | 200 | 97 | 195 | 4900 |
| ball 4D, [0,55] | 200 | 215 (d(0)=14.68) | 431 | 3025 |
| triangle, [0,180] | 15000 | 2435 (d(0)=5π²) | 4870 | 32400 |

So all values satisfy (or nearly satisfy: 4D ball, only near E=0) `μ ≥ D²` and are below `E*²`:
consistent with the theory; the spread 100…15000 is explained by `D²` (which is dominated by the
distance from `E_*=0` to `E_1` in the triangle) rather than by `E²`. For μ=15000, N=5000 the MC floor
of the penalty term is `μ·Var(u²)/N ≈ 3·Var(u²) ≈ 4` in the *total* loss — comparable to the
residuals 1–2 reported for the triangle; a smaller μ (≈5000) should work as well or better. The
sensitivity table placeholder is `[MU-TABLE-TBD]`; predicted outcome: eigenvalue location flat in μ
over `[D², E*²]`, collapse (`mean u² → 0`, residual → 0) for `μ < D²/2`.

---------------------------------------------------------------------------------------------------
## Assumptions, caveats, doubts

- Everything in Fragments 1–4 is for L self-adjoint with compact resolvent and `u ∈ D(L)`. The network
  `u=B·NN` (tanh) is in `D(L)` when B is W^{2,∞}, vanishes on ∂Ω, and `D(L)=H²∩H¹₀` (convex or C^{1,1}
  domain, bounded V). For the potentials "in ℝⁿ" the paper actually solves the Dirichlet problem in
  the unit ball; the certificates refer to that problem.
- Integrals in the code are means (`dx/|Ω|`); all statements are invariant under this change (said in
  Fragment 0). The quotient Λ is the same in both normalizations.
- Prop. `prop.stability` is conditional on the size δ of the total error; **no bound on δ is proved**
  (approximation rate for these minimizers and, above all, optimizer convergence). This is said
  explicitly in Remark `rem.stability`(5). Remark `rem.approx` proves only consistency
  (inf over all widths = Λ(E)) under `u_k/B ∈ C²(Ω̄)`; I checked this for disk/ball, square and
  triangle with polynomial B (quotient of analytic functions vanishing on the faces); for the
  annulus it should hold for a smooth product-type B (not checked against the code). Not claimed for nonsmooth B.
- The "cluster" statement (Remark `rem.stability`(4)) is deliberately weak: I do NOT claim that two
  eigenvalues closer than 2√δ can never be resolved (in principle the wings of the parabolas could
  be extrapolated); I only claim that data compatible with the error model exist which do not
  determine the number of eigenvalues in the cluster — i.e. thresholding/local-minimum detection
  cannot resolve them. Threshold: `2√(2δ)` two-sided, `2√δ` one-sided.
- MC bias/variance formulas (Remark `rem.certificate`(4)) are delta-method asymptotics (need finite
  4th moments and `mean u²` bounded away from 0); eq. `mc.floor` is exact. The confidence statement is
  asymptotic (CLT), flagged as "statistical certificate".
- Cost model (Fragment 5): `c_ep = κ(d+2)NW` reflects `_laplacian` (one grad + d second-order grads);
  constants κ, κ′, c₀ are not measured. Statements "only ≈K_I nodes need a full training" and
  `J ≈ 2|I|/γ_min` are model predictions — to be confirmed/qualified by the runtime agent
  (`[RUNTIME-TBD]` ×5). I wrote plainly that the scan is NOT competitive for the lowest few
  eigenvalues; its advantage is for windows high in the spectrum, parallelism, a-posteriori bounds,
  and nonlinear problems. The description of Rayleigh+deflation is structural (generic penalty
  deflation); I did not verify implementation details of refs [Rowan et al.], [Holliday et al.], so
  the text cites them only "as described in the Introduction". The downward-bias statement is the
  exact Courant–Fischer inequality (for exactly enforced constraints).
- p-Laplacian: Prop. `prop.plap` uses the class `D_p` with distributional `Δ_p u ∈ L²`; for a smooth
  network and p ≥ 2 the pointwise formula coincides with it. Integrability thresholds in Remark
  `rem.plap.errors`(2),(3) are for nondegenerate critical points ("in general"); experiments only use
  p>2 where the residual is bounded. The heavy-tail claim for the parameter gradient (second moment
  finite iff p > 3 − d/2) is a local scaling computation, not a theorem — phrased accordingly.
  `[PLAP-TBD]` placeholders: convergence study (grid, width, N) and the variational-quotient column.
- Citations: all 15 new bibitems are standard and I am confident of the data given (journal, volume,
  first page, year). Least certain detail: page range of Anane–Tsouli (pp. 1–9) and the exact
  author-order/first page of Benedikt–Drábek–Girg (Nonlinear Anal. 75 (2012) 4422–4435) — please
  verify these two against MathSciNet if possible. I did not include Cianchi–Maz'ya (not needed).
- Labels used that must exist in the paper: `Lambda`, `cota.igual`, `eigen`, `lossmu`, `p-eigen`,
  `sec.nolin`, `table.5`, `table.6`. New labels: `prop.penalized`, `cor.mu`, `rem.mu`, `mu.rule`,
  `mc.floor`, `expansion`, `prop.stability`, `hyp.delta`, `error.split`, `gap.cond`, `rem.stability`,
  `rem.approx`, `prop.rayleigh`, `rem.certificate`, `prop.parabola`, `parabola.vertex`,
  `rem.parabola`, `cost.scan`, `cost.rq`, `prop.plap`, `rem.plap.lost`, `lem.critical`,
  `rem.plap.errors`, `rem.radial`, `radial.ode`, `rem.nospurious`.
