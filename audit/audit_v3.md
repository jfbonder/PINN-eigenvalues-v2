# Audit of v3 (`paper/v3/FBS-rev2-arxiv.tex`, arXiv:2507.03126v3)

Scope: LaTeX source v3 (line numbers `L…` refer to `paper/v3/FBS-rev2-arxiv.tex`), cross-checked against
`pinn_eigen.py`, `FBS_PINN_notebook_v2.ipynb`, the figure PNGs in `paper/v3/`, the rev-2 source/PDF, v1/v2 PDFs,
the round-1 response letters, and independent computations (scripts in `audit/scripts/`, scipy 1.x):

- `tables_check.py` – exact eigenvalues, relative errors, Weinstein check for Tables 1–6, annulus/disk spectra.
- `plap_radial_check.py` – radial p-Laplacian eigenvalues by ODE shooting (correct equation and the equation the code solves).
- `potentials_reference.py` – reference eigenvalues for the harmonic potential in the disk; first-order estimates for the well potentials.

Severity: **C** = critical (wrong/misleading result or statement, must be fixed), **M** = major, **m** = minor.
No experiment was re-run (no GPU); statements about "what the code produced" are inferences and are marked as such.

---

## 0. Ten findings that matter most

| # | Sev | Finding | Where |
|---|-----|---------|-------|
| 1 | C | The 3-D p-Laplacian value "E1 = 17.4513" (p = 2.5, unit ball) is **not** the first eigenvalue. Radial shooting gives λ1 = **14.1112**. The equation the repo code solves (−Δ_p u = E·u with mean(u²)=1) gives 17.590, i.e. the published number reproduces the *code bug* to 0.8 %. | L804, L722; code L166–171 |
| 2 | C | Table 1 row (1,3): residual 1.40e-3 (v3) / 1.40e-4 (rev-2, v2) **violates the Weinstein bound stated in the paper** (\|err\| = 0.102 > √Res = 0.037 / 0.012). The number is wrong in every version. | L361 |
| 3 | C | Fig. 9 (annulus): the "theoretical" curve has a single zero at ≈ 39–40, but the annulus 0.5<r<1 has eigenvalues 39.013, **40.873 (double), 46.429 (double)** in the plotted window [0,50]. The reference curve is wrong and the PINN curve missed two eigenvalues (four eigenfunctions). | L570–577 |
| 4 | C | Table 4 (triangle) lists 8 rows for 4 **simple** eigenvalues; Table 1 (square) dropped the legitimately double rows. Exactly backwards. | L535–542, L358–361 |
| 5 | C | The loss actually implemented/plotted is not the one in the paper: (i) MC `mean()` instead of integrals; (ii) p-Laplacian branch uses `E*u` and `mean(u²)`; (iii) "refinement" is a noisy 1-D minimisation over E with frozen network (= a Monte-Carlo Rayleigh quotient), not "a finer grid". | L229–251, L665; code L163–172, L334–343 |
| 6 | M | The penalised loss curve has a closed form the paper misses: inf_u Λ̄(u,E) = d − d²/(4μ) if d ≤ 2μ, = μ (at u≡0) if d ≥ 2μ, with d = min_k(E−E_k)². The plotted residual part equals d − d²/(2μ) ≤ μ/2. This (a) explains why all blue curves sit below the orange one at peaks, (b) gives the rigorous rule μ > d_max/2 (largest half-gap squared), **not** μ ∝ E², (c) lets one read μ off the figures: Fig. 6 (3-D) and Figs. 12–13 were run with μ = 100, not the μ = 200 / 400 stated in the text. | L312–320, L382, L617 |
| 7 | M | Proof of Theorem 1: the displayed estimate should read ½∫(Δu)² − E², not "+E²−E"; ε = 1/(4E) is undefined at E = 0; E ≥ 0 is unnecessary; "Ω bounded" is not enough for the H² bound (needs convex or C^{1,1}). | L869–897 |
| 8 | M | Harmonic ω = 10 (Fig. 12): the true spectrum has 45.089 (double) and 46.291 (simple); the scan (step ≈ 0.7) shows one minimum. Close eigenvalues/multiplicities are not resolved; nowhere stated. Also the first grid point E = 0 has loss 0: the network collapsed to u ≡ 0 (d = 204 > 2μ = 200), exactly as item 6 predicts. | L637, L602 |
| 9 | M | Overclaims: "independent of the space dimension", "effective regardless of the dimensionality", "robust … independent of V or the geometry", "no previous numerical results" for p-Laplacian in higher dimension (false: radial case is a 1-D ODE; inverse-power computations for balls exist), "does not rely on the Rayleigh quotient" (the refine step *is* one). | L38, L184–188, L722, L847–849 |
| 10 | M | Editor's format list largely unmet: amsart class, 3-sentence abstract, alphabetical (not citation-order) bibliography, no roadmap, no Declarations, caption without period, table style, broken reference [18]. | see §5 |

---

## 1. Tables

Exact values used (scipy): π²(m²+n²); 3-D ball j²_{m+1/2,n}; 4-D ball j²_{m+1,n}.

### 1.1 Recomputation of every row

`|err|` = |computed − exact|; `√Res` = Weinstein radius if Residual ≈ Λ = r²; `KT` = Kato–Temple radius Res/gap
(relevant because the code's refine step returns the Rayleigh quotient, see C-3).

**Table 1 (unit square), L355–362**

| row | exact | computed | rel.err (mine) | rel.err (paper) | Res | \|err\| | √Res | KT | verdict |
|---|---|---|---|---|---|---|---|---|---|
| (1,1) | 19.739209 | 19.739033 | 8.91e-6 | 8.9e-6 | 6.72e-3 | 0.0002 | 0.082 | 0.0002 | ok |
| (1,2) | 49.348022 | 49.385078 | 7.51e-4 | 7.5e-4 | 1.16e-1 | 0.0371 | 0.341 | 0.004 | ok (Weinstein) |
| (2,2) | 78.956835 | 78.962188 | 6.78e-5 | 6.78e-5 | 1.26e-1 | 0.0054 | 0.355 | 0.006 | ok; "≈ 78.956" should be **78.957** |
| (1,3) | 98.696044 | 98.594123 | 1.03e-3 | 1.03e-3 | **1.40e-3** | **0.1019** | **0.037** | 0.0001 | **Weinstein violated** |

- **T1-a (C), L361.** Residual of row (1,3): v3 `1.40e-3`, rev-2 and v2 `1.40e-4`. Both contradict
  dist(E,σ) ≤ √Λ (L266) by a factor 2.7 / 8.6. With neighbours at 1.2e-1 the true value is most likely
  ~1e-1 or larger (mantissa kept, exponent mistyped), but it cannot be reconstructed: **re-run or recover the log**.
  Until then the row cannot be published. (The v3 change 1e-4 → 1e-3 is an undocumented edit.)
- **T1-b (M), L358–361.** 5π² and 10π² are double eigenvalues in the square (eigenfunctions (1,2),(2,1) and (1,3),(3,1)).
  rev-2 listed both index pairs with identical computed values; v3 silently kept one. Neither is right: the method
  returns one number per minimum and cannot see multiplicity. Fix: one row per distinct eigenvalue plus a
  "mult." column, and one sentence saying multiplicities are not detected.
- **T1-c (M).** The run parameters of Table 1 are not in the text. Text gives μ = 3000 only for the [44,55] scan (L323);
  the figure file is `…mu=450_cuadrado.png`, eigenfunction files `mu=150`, notebook μ = 500, a commented block L326–336 says μ = 500.
  From Fig. 7 at E = 0: plotted value ≈ 222 while d = 389.6; d − d²/(2μ) = 221 for μ = 450. So Fig. 7 used μ = 450.
- **T1-d (m).** Inconsistent significant digits (8.9, 7.5 vs 6.78, 1.03). Column header "Residual" undefined (see M-9).

Proposed replacement (residual of last row to be filled after re-run):

```latex
\begin{tabular}{ccccccc}
\hline
$(m,n)$ & mult. & $E_{(m,n)}=\pi^2(m^2+n^2)$ & Computed & Rel. error & $\bar\Lambda_{\rm res}$ & $\bar\Lambda_{\rm res}^{1/2}$\\
\hline
$(1,1)$ & 1 & $2\pi^2\approx 19.7392$ & 19.739033 & $8.91\times10^{-6}$ & $6.72\times10^{-3}$ & 0.082\\
$(1,2),(2,1)$ & 2 & $5\pi^2\approx 49.3480$ & 49.385078 & $7.51\times10^{-4}$ & $1.16\times10^{-1}$ & 0.341\\
$(2,2)$ & 1 & $8\pi^2\approx 78.9568$ & 78.962188 & $6.78\times10^{-5}$ & $1.26\times10^{-1}$ & 0.355\\
$(1,3),(3,1)$ & 2 & $10\pi^2\approx 98.6960$ & 98.594123 & $1.03\times10^{-3}$ & \textbf{[re-run]} & \textbf{[re-run]}\\
\hline
\end{tabular}
```

**Table 2 (3-D ball), L414–422** — all exact values and relative errors confirmed.

| (m,n) | exact j²_{m+1/2,n} | mult. 2m+1 | computed | rel.err (mine) | Res | \|err\| | √Res | KT |
|---|---|---|---|---|---|---|---|---|
| (0,1) | 9.869604 | 1 | 9.885022 | 1.56e-3 | 8.61e-2 | 0.0154 | 0.293 | 0.008 |
| (1,1) | 20.190729 | 3 | 20.182569 | 4.04e-4 | 4.59e-2 | 0.0082 | 0.214 | 0.005 |
| (2,1) | 33.217462 | 5 | 33.243410 | 7.81e-4 | 1.00e-1 | 0.0259 | 0.316 | 0.016 |
| (0,2) | 39.478418 | 1 | 39.494155 | 3.99e-4 | 1.81e-1 | 0.0157 | 0.425 | 0.029 |
| (3,1) | 48.831194 | 7 | 48.826976 | 8.64e-5 | 2.24e-1 | 0.0042 | 0.473 | 0.024 |
| (1,2) | 59.679516 | 3 | 59.734069 | 9.14e-4 | 2.04e-1 | 0.0546 | 0.452 | 0.028 |
| (4,1) | 66.954312 | 9 | 66.991853 | 5.61e-4 | 4.82e-1 | 0.0375 | 0.694 | 0.066 |

- **T2-a (m).** No inconsistency with rev-2. Weinstein holds in all rows with a factor 10–100 of slack.
- **T2-b (M).** Multiplicities (1,3,5,1,7,3,9) are not mentioned; add a column. The list is complete up to 66.95 (next: 82.72).
- **T2-c (M), code.** The notebook's reference list `EXACT_3D = [9.8696, 20.1907, 33.2251, 48.9737, 59.6946, 66.9489]`
  is wrong in four entries (correct: 33.2175, 48.8312, 59.6795, 66.9543). The paper's table has the correct values, so the
  notebook does not reproduce the table's error column and the orange curve it draws is slightly off.
- **T2-d (M).** Text says μ = 200 for both balls (L382); notebook says 400; figure file says `mu=100`, and the 3-D curve at E = 0
  (≈ 45–50 vs d = 97.4) matches d − d²/(2μ) = 50.0 for μ = 100 (μ = 200 would give 73.7). Figure 6 (top) stops at E = 55 while
  Table 2 goes to 67: table and figure come from different runs.

**Table 3 (4-D ball), L454–459**

| (m,n) | exact j²_{m+1,n} | mult. (m+1)² | computed | rel.err (mine) | paper | Res | \|err\| | √Res |
|---|---|---|---|---|---|---|---|---|
| (0,1) | 14.681971 | 1 | 14.641796 | 2.74e-3 | 2.73e-3 | 3.84e-3 (v3) / 3.84e-2 (rev-2, v2) | 0.0402 | 0.062 / 0.196 |
| (1,1) | 26.374616 | 4 | 26.257056 | 4.46e-3 | 4.47e-3 | 2.13e-1 | 0.1176 | 0.462 |
| (2,1) | 40.706466 | 9 | 40.794700 | 2.17e-3 | 2.16e-3 | 1.98e-1 | 0.0882 | 0.445 |
| (0,2) | 49.218456 | 1 | 49.175395 | 8.75e-4 | 8.74e-4 | 6.04e-1 | 0.0431 | 0.777 |

- **T3-a (C), L456.** Residual of row (0,1) changed 3.84e-2 → 3.84e-3 between rev-2 and v3 with the same computed eigenvalue.
  Both satisfy Weinstein, so mathematics cannot decide; 3.84e-2 is the one consistent with v2 and with the magnitude of the other rows.
  Recover from logs / re-run.
- **T3-b (m).** Rounding: 14.681971 → **14.6820** (paper 14.6819), 40.706466 → **40.7065** (40.7064), 49.218456 → **49.2185** (49.2184);
  the paper truncates. Relative errors recomputed with full-precision exact values: 2.74e-3, 4.46e-3, 2.17e-3, 8.75e-4.
  The same truncated value 14.6819 appears at L153 (E_2 of the disk).
- **T3-c (m), L454/L405.** `y_{m,n}` is the standard symbol for zeros of the Bessel function of the *second* kind Y_m, and `j_{m,n}` for zeros of
  J_m (not J_{m+1/2}). Use `j_{m+\frac12,n}` (d = 3) and `j_{m+1,n}` (d = 4), or `j_{\nu,n}` with ν = m + (d−2)/2.
- **T3-d (m).** Mathematical content of L405 is right (ν = m + ½ for d = 3, ν = m + 1 for d = 4), but m ranges over N₀, not N.

**Table 4 (right isosceles triangle), L533–542**

| mode | exact | computed | rel.err (mine) | Res | \|err\| | √Res | KT |
|---|---|---|---|---|---|---|---|
| (2,1) | 5π² = 49.348022 | 49.369942 | 4.44e-4 | 2.42e-1 | 0.0219 | 0.492 | 0.005 |
| (3,1) | 10π² = 98.696044 | 98.892161 | 1.99e-3 | 1.26 | 0.1961 | 1.122 | 0.043 |
| (3,2) | 13π² = 128.304857 | 128.048509 | 2.00e-3 | 1.98 | 0.2563 | 1.407 | 0.068 |
| (4,1) | 17π² = 167.783275 | 167.687185 | 5.73e-4 | 2.16 | 0.0961 | 1.470 | 0.073 |

- **T4-a (C), L535–542.** The Dirichlet eigenfunctions of the triangle (0,0),(1,0),(0,1) are
  u_{m,n} = sin(mπx)sin(nπy) − (−1)^{m+n} sin(nπx)sin(mπy), m > n ≥ 1 (checked: vanishes on x + y = 1), eigenvalue π²(m²+n²), **simple**
  (in this range). u_{n,m} = ±u_{m,n}. The 8-row table duplicates each of the 4 eigenvalues. Delete the four duplicate rows
  and state the index set m > n ≥ 1. The spectrum is complete on [0,180] (next: 20π² = 197.4).
- **T4-b (m), L533.** Header `E_{m,n}` (Tables 1–3 use `E_{(m,n)}`), "residual" lower case.
- **T4-c (M).** Fig. 11 panel titles read E = 127.7529 and E = 168.5851; Table 4 gives 128.048509 and 167.687185. Figures and table
  come from different runs (differences 0.30 and 0.90, larger than the table's errors).
- **T4-d (m).** Commented block L517–526 says μ = 1500; text L513 says 15000 (file name agrees with 15000); notebook uses 4000 samples, 2000 epochs
  (text: 5000 / 3000).

```latex
\begin{tabular}{cccccc}
\hline
$(m,n)$ & $E_{(m,n)}=\pi^2(m^2+n^2)$ & Computed & Rel. error & $\bar\Lambda_{\rm res}$ & $\bar\Lambda_{\rm res}^{1/2}$\\
\hline
$(2,1)$ & $5\pi^2\approx 49.3480$   & 49.369942  & $4.44\times10^{-4}$ & $2.42\times10^{-1}$ & 0.49\\
$(3,1)$ & $10\pi^2\approx 98.6960$  & 98.892161  & $1.99\times10^{-3}$ & $1.26$ & 1.12\\
$(3,2)$ & $13\pi^2\approx 128.3049$ & 128.048509 & $2.00\times10^{-3}$ & $1.98$ & 1.41\\
$(4,1)$ & $17\pi^2\approx 167.7833$ & 167.687185 & $5.73\times10^{-4}$ & $2.16$ & 1.47\\
\hline
\end{tabular}
\caption{Dirichlet eigenvalues of the triangle with vertices $(0,0)$, $(1,0)$, $(0,1)$. The eigenvalues are
$\pi^2(m^2+n^2)$, $m>n\ge1$, with eigenfunctions $\sin(m\pi x)\sin(n\pi y)-(-1)^{m+n}\sin(n\pi x)\sin(m\pi y)$; all those listed are simple.}
```

**Table 5 (p = 2.2, unit disk), L730–736**

| k | Horák | computed | rel.err (mine) | Res | independent check |
|---|---|---|---|---|---|
| 1 | 6.5320 | 6.679258 | 2.25e-2 | 1.81 | radial shooting: λ1 = **6.53174** (confirms Horák) |
| 2 | 18.395 | 19.441806 | 5.69e-2 | 0.965 | non-radial, not checked |
| – | – | 37.621714 | – | 1.82 | second *radial* eigenvalue is **41.7886**; 37.62 is not it |

- **T5-a (M).** Arithmetic of the error column is right. Reference λ1 confirmed independently.
- **T5-b (C).** Third row: the text (L692) says "the first two eigenvalues"; the table lists a third number with no reference and no
  comment. It is not the second radial eigenvalue (41.79). If it is claimed as λ3 that is a new, unvalidated claim (and for p ≠ 2 it is not even known
  that the spectrum is discrete beyond λ2). Either delete the row or label it "further local minimum of the loss curve, not validated".
- **T5-c (C).** Which equation produced these numbers? The equation in `pinn_eigen.py` (−Δ_p u = E u, mean(u²) = 1) has first radial
  "eigenvalue" 6.935 on the disk; the paper's 6.679 lies between that and the true 6.532. For the 3-D case the match with the buggy
  equation is unambiguous (see P-1). The table must be regenerated with the corrected loss.
- **T5-d (m), L730.** Column spec `c| c|c|c|c|c|c` declares 7 columns, 5 are used (Table 6, L769: 6 declared). "planar unit ball" → "unit disk".
  Which table of Horák is used is given for Table 6 (`[Table 2]`) but not for Table 5.
- **T5-e (M).** Weinstein does not apply, but note that for k = 2 |err| = 1.05 > √Res = 0.98 and for Table 6, k = 1, 0.37 > 0.25: the residual gives
  no control here, contrary to the impression left by L285–288 ("accurate eigenvalues").

**Table 6 (p = 2.2, square [0,2]²), L769–774**

| k | Horák | computed | rel.err (mine) | Res |
|---|---|---|---|---|
| 1 | 5.4952 | 5.861383 | 6.66e-2 | 6.22e-2 |
| 2 | 15.144 | 15.144490 | 3.24e-5 | 1.64 |

- **T6-a (M).** Arithmetic right. The pattern is implausible as a measure of accuracy: the *first* eigenvalue is off by 6.7 % with the *smallest*
  residual, the second agrees to 3e-5 with a residual 26 times larger, on a grid of step 20/49 = 0.41 and with a noisy refine step. The agreement for k = 2
  is a coincidence and should not be quoted with 3 significant digits of error. (Plausibility of the references: for p = 2 the values are π²/2 = 4.935 and
  5π²/4 = 12.337.)
- **T6-b (M).** μ: text L720 says 4000 for "the p-Laplacian experiments"; notebook uses μ(E) = 1000 + E² for the square; the commented block L752 says
  100 + λ²; the disk figure file is named `mu=150_+_lam_x_lam`.
- **T6-c (m), L775.** "lenght" → "side length".

### 1.2 Other numbers

- **N-1 (m), L153.** Disk: 5.7832, 14.6820 (paper 14.6819), 26.3746, 30.4713 – correct up to the rounding of E_2. E_2 and E_3 are double; say so.
- **N-2 (m), L323.** 5π² = 49.348 is the only eigenvalue in [44,55] – correct, but it has multiplicity two ("one eigenvalue" → "one (double) eigenvalue").
  The symbol `E_{(1)}` is used nowhere else; use `E_{(1,2)}`.
- **N-3 (C), L804.** "E_1 = 17.4513, E_2 = 45.8908" – see P-1. Four decimals on a grid of step ≈ 1.7 (read from the figure) are not justified in any case.

---

## 2. Mathematics

- **M-1 (M), L103–125. The identity Λ(E) = min_k (E−E_k)² is correct, but the presentation is circular and duplicated.**
  L106–110 already prove "≥" and equality at u = u_k, i.e. the identity. L112–121 then re-derive "≤" ("Moreover…the following theoretical upper bound"),
  and L122–125 conclude the identity again (no full stop after (4)). Afterwards every caption still calls the curve "theoretical *upper bound*".
  Hypotheses are missing: the expansion needs a complete orthonormal eigenbasis, i.e. L self-adjoint with compact resolvent (bounded Ω with Dirichlet
  condition, or V → ∞), u in the domain of L, and Parseval for Lu. For problem (1) on R^n with a decaying potential there is continuous
  spectrum; then Λ(E) = dist(E,σ(L))² and the infimum is not attained. "Λ(E) = 0 iff E is an eigenvalue" (L104–105) also needs attainment.
  Replacement for L103–125:

  ```latex
  Assume that $L=-\Delta+V$ (with the boundary condition built into its domain $D(L)$) is self-adjoint with compact
  resolvent, as is the case for a bounded domain with Dirichlet boundary condition and bounded $V$, and let
  $\{u_k\}$ be an orthonormal basis of eigenfunctions, $Lu_k=E_ku_k$. Writing $u=\sum_k c_ku_k\in D(L)$,
  \[
  \Lambda(u,E)=\frac{\sum_k c_k^2(E_k-E)^2}{\sum_k c_k^2}\ \ge\ \min_k (E_k-E)^2 ,
  \]
  with equality if and only if $u$ lies in the eigenspace of the eigenvalue(s) closest to $E$. Hence
  \begin{equation}\label{cota.igual}
  \Lambda(E)=\min_k (E_k-E)^2=\operatorname{dist}\big(E,\sigma(L)\big)^2 .
  \end{equation}
  In particular $\Lambda\ge0$, the zeros of $\Lambda$ are exactly the eigenvalues, every local minimum of $\Lambda$ is a zero,
  and $\Lambda$ is a parabola of unit curvature around each eigenvalue. The infimum is attained (Theorem~\ref{teo.min} gives
  a direct proof that does not use the spectral decomposition). Note that \eqref{cota.igual} does not see multiplicities.
  ```
  This also answers round-1 R1 #1 properly (all local minima of the exact curve are eigenvalues) – the round-1 reply only asserted it.

- **M-2 (M), L229–233, L312–320. The penalised loss has an exact landscape; the paper's μ ∝ E² heuristic is not what governs it.**
  For any u ∈ D(L): Λ̄(u,E) = ‖(L−E)u‖² + μ(‖u‖²−1)² ≥ d·s + μ(s−1)², s = ‖u‖², d = d(E) = min_k (E−E_k)², with equality for u ∝ u_k. Minimising over s ≥ 0:

  ```latex
  \[
  \bar\Lambda(E):=\inf_u\bar\Lambda(u,E)=
  \begin{cases}
  d(E)-\dfrac{d(E)^2}{4\mu}, & d(E)\le 2\mu,\quad\text{attained at } \|u\|_2^2=1-\dfrac{d(E)}{2\mu},\\[2mm]
  \mu, & d(E)\ge 2\mu,\quad\text{attained only at } u\equiv 0 ,
  \end{cases}
  \qquad d(E)=\min_k(E-E_k)^2 .
  \]
  ```
  At the minimiser the *residual part* (what `sweep` returns and the figures plot) equals d − d²/(2μ) ≤ μ/2.
  Verified numerically on a 4-mode model (script in this audit) and against the paper's own figures:
  disk Fig. 3, E = 0: d = 33.4 → 27.8 (plot ≈ 27); square Fig. 7, E = 0: d = 389.6, μ = 450 → 221 (plot ≈ 222);
  4-D Fig. 6, E = 0: d = 215.6, μ = 200 → 99 (plot ≈ 103); harmonic ω = 1, E = 0: d = 34.7, μ = 100 → 28.7 (plot 28.6);
  harmonic ω = 10, μ = 100: peaks predicted 40 and 43, observed 40 and 41, and at E = 0 (d = 204 > 2μ) the network collapsed to u ≡ 0 (loss 0).
  Consequences:
  1. the zeros and their unit curvature are unchanged for every μ > 0 – the *location* of the minima does not depend on μ in exact arithmetic;
  2. the trivial state u ≡ 0 is the global minimiser iff d(E) ≥ 2μ. The rigorous requirement is therefore **μ > ½·max_{E in the scan window} dist(E,σ)²** – (half the largest
     spectral gap)²/2, plus the distance of the window ends to the spectrum – and not μ ∝ E². By Weyl's law gaps do not grow with E in d ≥ 2, so μ does not need to grow either.
     A practical rule: μ ≈ d_max to 2·d_max keeps ‖u‖² ≥ ½–¾ on the whole window;
  3. the heuristic at L312–320 is about the *initialisation* (residual of a random network ∝ E²), i.e. about optimisation dynamics, and should be
     presented as such; "it is necessary to scale μ ∝ E²" (L320) is false as stated and is contradicted by the paper's own values (μ = 100 up to E = 33, μ = 200 up to E = 67,
     μ = 400–1000 for the same E-range as μ = 200, μ = 15000 up to E = 180);
  4. the curve labelled "theoretical upper bound" is not an upper bound for the plotted quantity near the peaks.

- **M-3 (M), L163–169.** "we look for zeros of the loss curve Λ(E^i). By construction, these zeroes correspond with eigenvalues E^i" – grid points are not eigenvalues and
  the trained loss is never zero. Say: local minima of the discrete curve below a threshold; the eigenvalue is then located within one grid step and refined.

- **M-4 (M), L257–282. Weinstein / Krylov–Bogoliubov.** The statement dist(E,σ(L)) ≤ ‖(L−E)u‖/‖u‖ is correct for self-adjoint L and u ∈ D(L), u ≠ 0;
  attribution (Krylov–Bogoliubov 1929, Weinstein 1934) is fine, and it holds for the PINN ansatz because B(x)·NN(x) satisfies the boundary condition exactly
  (worth saying – it is the reason for the hard constraint). Problems:
  1. "Residual" in the tables is `loss_res` = MC mean of the squared residual ≈ Λ = r², not r. A reader comparing the table with "dist ≤ r" will take r = 0.24 instead of 0.49 etc.
     Also it is not divided by mean(u²) (fine at the minima, where the norm is ≈ 1, but it must be said).
  2. "in practice the actual error is well below r(u,E)" (L273–274) is contradicted by Table 1 row (1,3) as printed (T1-a).
  3. The explanation offered (flat floor η does not move the minimiser, L274–282) is heuristic. The true reason for "error ≪ residual" is quantitative and classical:
     for fixed u, E ↦ ‖(L−E)u‖² is minimised at the **Rayleigh quotient** ρ(u) = ⟨u,Lu⟩/⟨u,u⟩ – this is exactly what `refine` computes (C-3) – and the Kato–Temple
     inequality gives |ρ(u) − E_k| ≤ r(u,ρ)²/δ, δ = dist(ρ, σ(L)∖{E_k}). The error is quadratic in the residual. Suggested text to append after L282:

     ```latex
     This can be made quantitative. For fixed $u$ the map $E\mapsto\|(L-E)u\|_2^2$ is a parabola whose minimum is attained at the
     Rayleigh quotient $\rho(u)=\langle u,Lu\rangle/\langle u,u\rangle$, and if $E_k$ is the only point of $\sigma(L)$ in
     $(\rho-\delta,\rho+\delta)$, the Kato--Temple inequality \cite{Kato1949} gives
     \[
     |\rho(u)-E_k|\le \frac{r(u,\rho(u))^2}{\delta}.
     \]
     Thus the eigenvalue error is quadratic in the residual: with $r^2\approx 10^{-1}$ and spectral gaps $\delta\approx 10$ one expects
     errors of order $10^{-2}$, in agreement with Tables~\ref{table.1}--\ref{table.4}.
     ```
     Caveat for the authors: the *observed* errors in Tables 2–4 are mostly 2–5 times **larger** than the KT radius (columns above). In exact arithmetic that is impossible if the
     reported value were the Rayleigh quotient; the excess is the Monte-Carlo noise of `refine` (C-3). This is evidence that the published digits are limited by the refine step,
     not by the network.
  4. L285–288: "Tables 5 and 6 show the same behavior as in the linear case: accurate eigenvalues" – errors of 2.3 %, 5.7 %, 6.7 % are not "accurate", and this is a forward
     reference to Section 5 from Section 3. Delete or move to Section 5 and rephrase.

- **M-5 (M), L869–897. Theorem 1 and its proof.**
  1. Arithmetic slip, L893–896. From Λ(u_j,E) = ∫(Δu_j)² + 2E∫u_jΔu_j + E² and |∫u_jΔu_j| ≤ ε∫(Δu_j)² + 1/(4ε), with ε = 1/(4E):
     2E·ε = ½ and 2E·1/(4ε) = 2E², hence Λ(u_j,E) ≥ ½∫(Δu_j)² + E² − 2E² = **½∫(Δu_j)² − E²**, not "+ E² − E". The conclusion (bounded ‖Δu_j‖₂) survives.
  2. ε = 1/(4E) is undefined for E = 0 (allowed by "E ≥ 0"); write `1/(4E)` not `1/4E`; the case E = 0 is trivial.
  3. E ≥ 0 is not needed: with ε = 1/(4|E|) the same estimate gives ½∫(Δu)² − E² for every E ≠ 0. State the theorem for E ∈ R.
  4. "Ω ⊂ R^n bounded" is not sufficient for "bounded in H²": the estimate ‖u‖_{H²} ≤ C‖Δu‖₂ on H² ∩ H¹₀ requires ∂Ω ∈ C^{1,1} or Ω convex.
     All domains in the paper qualify (disk, balls, annulus: smooth; square, triangle: convex), so add the hypothesis; or work in D(Δ_D) = {u ∈ H¹₀ : Δu ∈ L²} with the graph norm, which needs no regularity.
  5. `a, b > 0` → `a, b ≥ 0`; "immersion" (L909) → "embedding"; ϵ here clashes with the threshold ϵ (L243) and ε (early stopping).
  6. The theorem covers only V of the form (5) (pure Dirichlet Laplacian) while L101 invokes it for the general problem (1). Either extend (V ∈ L^∞: same proof, the extra term is lower order) or say so at L101.
  7. Given M-1 the theorem is a corollary (the minimiser is any eigenfunction of the nearest eigenvalue). Its value is that the proof does not use the spectral theorem – say so, otherwise a referee will ask why it is there (R2 round 1 #6 already did).

  Replacement for L885–897:

  ```latex
  Now, since $\|u_j\|_2=1$,
  \[
  \Lambda(u_j,E)=\int_\Omega(\Delta u_j+Eu_j)^2\,dx=\int_\Omega(\Delta u_j)^2\,dx+2E\int_\Omega u_j\Delta u_j\,dx+E^2 .
  \]
  If $E=0$ there is nothing to do. Otherwise, by Young's inequality $ab\le\varepsilon a^2+\frac{b^2}{4\varepsilon}$ with $\varepsilon=\frac{1}{4|E|}$,
  \[
  2|E|\int_\Omega|u_j||\Delta u_j|\,dx\le\frac12\int_\Omega(\Delta u_j)^2\,dx+2E^2 ,
  \]
  and therefore
  \begin{equation}\label{cota.Lambda2}
  \Lambda(u_j,E)\ge\frac12\int_\Omega(\Delta u_j)^2\,dx-E^2 .
  \end{equation}
  Combining \eqref{cota.Lambda} and \eqref{cota.Lambda2} we get $\|\Delta u_j\|_2^2\le 2(C+E^2)$. Since $\Omega$ is convex or of class
  $C^{1,1}$, $\|u\|_{H^2(\Omega)}\le C_\Omega\|\Delta u\|_2$ for every $u\in H^2(\Omega)\cap H^1_0(\Omega)$, so $\{u_j\}$ is bounded in $H^2(\Omega)\cap H^1_0(\Omega)$.
  ```
  and the statement: "Let Ω ⊂ R^n be a bounded domain which is convex or has C^{1,1} boundary, and V as in (5). For every E ∈ R there exists …".

- **M-6 (M), L663–688. p-Laplacian bound.**
  1. The computation is correct: for ‖u_k‖_p = 1, Λ_p(u_k,E) = (E−E_k)² ∫|u_k|^{2(p−1)} dx, so Λ_p(E) ≤ min_k c_k (E−E_k)², c_k = ‖u_k‖_{2(p−1)}^{2(p−1)} < ∞ (eigenfunctions are bounded),
     and c_k = 1 for p = 2. For the first radial eigenfunctions (‖u‖_p = 1 w.r.t. Lebesgue measure) the constants can be computed from the shooting solution if wanted.
  2. "min_k" presupposes an enumeration of the spectrum. For p ≠ 2 only the variational (Ljusternik–Schnirelmann) sequence is known; whether it exhausts the spectrum, and whether
     the spectrum is discrete above λ2, is open. Write "for every eigenpair (E_*,u_*)": Λ_p(E) ≤ inf over eigenpairs.
  3. "Λ_p(u,E) = 0 with ‖u‖_p = 1 holds only if u is an eigenfunction" – Λ_p = 0 already forces ‖u‖_p = 1; and one must fix the function class (u ∈ W^{1,p}_0 with Δ_p u ∈ L²).
     Whether the infimum defining Λ_p(E) is attained is not known – say "Λ_p(E_k) = 0" (true) but do not claim "Λ_p(E) = 0 only at eigenvalues" without attainment.
  4. Naming: Λ(u,E) (L95) is a quotient without penalty, Λ̄ (L231) has the penalty, Λ_p (L665) has the penalty again; the p-Laplacian captions (L787, L804) say Λ̄. Use Λ̄_p for the penalised
     functional. A cleaner object, scale invariant and with constant 1, is
     `\Lambda_p(u,E)=\int(\Delta_pu+E|u|^{p-2}u)^2dx\,/\int|u|^{2(p-1)}dx`, for which Λ_p(u_k,E) = (E−E_k)² exactly.
  5. Not said and important for R1 #5: for p > 2 eigenfunctions are only C^{1,α}; near a non-degenerate critical point x₀ the eigenfunction behaves like u(x₀) − c|x−x₀|^{p/(p−1)},
     whereas a tanh network is smooth, so its Δ_p vanishes like |x−x₀|^{p−2} at its critical points while the right-hand side −E|u|^{p−2}u does not. The strong-form residual of a smooth
     ansatz therefore cannot vanish near critical points: there is an approximation floor (visible in Fig. 14, where the minima sit at 1–2, not near 0). For p < 2 the strong form is singular.
     `_p_laplacian` (code L112) regularises |∇u| by +1e-12 only.
  6. L661 "p > 1": only p = 2.2 and 2.5 are tried; restrict the claim to p ≥ 2 or say what happens for p < 2.

- **M-7 (m), L129–135.** Fine in substance; "section \ref{sec.nolin}" → "Section~\ref{sec.nolin}". The sentence "the loss vanishes exactly at the eigenvalues" needs the attainment caveat of M-6.3.

- **M-8 (M), L137–138 and L181, L847–849. "No need to know the lower ones / does not rely on the Rayleigh quotient".** True for the scan. But (i) the idea of scanning a
  spectral parameter and locating the minima of a residual / smallest singular value is classical and must be credited: the method of particular solutions (Fox–Henrici–Moler 1967;
  Betcke–Trefethen, SIAM Rev. 47 (2005)), variance minimisation ⟨(H−E)²⟩ in quantum Monte Carlo, and the folded-spectrum method (Wang–Zunger 1994), whose stated purpose is
  precisely interior eigenvalues without the lower ones; among PINN works, Ben-Shaul et al. (Neural Comput. 2023) and the scanning of Jin et al. are close. The authors should check and cite
  the closest of these (bibliographic details to be verified by the writer); (ii) the refined eigenvalue *is* a Rayleigh quotient (M-4.3).

- **M-9 (M). "Residual" is never defined.** Define once, e.g. after L243:
  ```latex
  In the tables we report, besides the computed eigenvalue $E$, the value
  $\bar\Lambda_{\rm res}=\frac1N\sum_{i=1}^N\big(\Delta u(x_i)-V(x_i)u(x_i)+Eu(x_i)\big)^2$ (the first term of \eqref{lossmu}, evaluated on $N=10^4$ fresh points).
  Since $\frac1N\sum u(x_i)^2\approx1$ at the minima, $\bar\Lambda_{\rm res}\approx\Lambda(u,E)=r(u,E)^2$.
  ```

- **M-10 (m), L78–82.** L(u) for the Rayleigh quotient clashes with L = −Δ + V (L259). Rename the quotient R(u). The domain of integration is missing in (2), (3), L80, L665.

- **M-11 (m), L55–58 vs L214–227.** Problem (1) is posed in R^n, everything afterwards is on a bounded Ω with Dirichlet data, including the "potentials" (harmonic, wells), which are
  in fact −Δ + V on the unit disk/ball with u = 0 on the boundary. Say this explicitly; it also means the harmonic example is *not* the harmonic oscillator.
  n is the dimension at L57, L74, L864; d at L198, L375; n is also a mode index in Tables 1–4. Use d for the dimension throughout.

---

## 3. Code vs. text

- **C-1 (C), code L163–171 vs L663–667. p-Laplacian loss.** The paper: (Δ_p u + E|u|^{p−2}u)² and (∫|u|^p − 1)². The code: `Lu - V*u + E_val*u` (comment: "linearised: |u|^{p-2}u ≈ u near p=2")
  and `((u**2).mean() - 1)**2`. The equation −Δ_p u = E u is not homogeneous: if (E,u) solves it then (c^{p−2}E, cu) does too, so "E" depends on the normalisation and is not an eigenvalue of (6).
  - **P-1.** Radial shooting (`audit/scripts/plap_radial_check.py`, validated at p = 2: 5.78319, 9.86960, and against Horák at p = 2.2: 6.53174):

    | case | true λ1 (eq. (6)) | code equation, mean(u²)=1 | paper |
    |---|---|---|---|
    | disk, p = 2.2 | 6.5317 (2nd radial 41.789) | 6.9350 (2nd radial 44.43) | 6.679 (Table 5) |
    | 3-D ball, p = 2.5 | **14.1112** (2nd radial 87.49) | **17.5899** | **17.4513** (Fig. 16) |
    | 3-D ball, p = 2.2 | 11.4699 | 12.4389 | – |

    For the 3-D figure the paper's value is 0.8 % from the buggy equation and 23.7 % from the true eigenvalue. Unless the original (amsalort) code differs, Fig. 16's "E_1, E_2" are not
    eigenvalues of the p-Laplacian. The 2-D numbers are inconclusive (between the two). All p-Laplacian results must be regenerated with
    `E_val * u.abs()**(p-2) * u` and `(u.abs()**p).mean()`; the 3-D p = 2.5 case can then be validated against 14.1112 (a reference value the paper says does not exist, L722).
  - `refine` with p ≠ 2 minimises mean((Δ_p u + E u)²) over E, i.e. returns ∫|∇u|^p/∫u², not ∫|∇u|^p/∫|u|^p.
  - `loss_fn` keeps `- V*u` in the p-branch although the docstring says V = 0 is assumed.

- **C-2 (M), code L170–172 vs L231. Monte-Carlo normalisation.** The code uses sample means: (1/|Ω|)∫res² + μ((1/|Ω|)∫u² − 1)². So the constraint is ∫u² = |Ω| (π, 4π/3, π²/2, ½, 1, 4, 3π/4), not 1, and the
  residual term carries 1/|Ω|. For the linear problem this is a harmless rescaling of u (the ratio Λ is unchanged); for the p-Laplacian with the wrong homogeneity (C-1) it changes the answer.
  Fix the text: write the loss with averages `\fint_\Omega` (or `\frac{1}{|\Omega|}\int_\Omega`), or put |Ω| in the code.

- **C-3 (C), code L291–351 vs L251. The refinement step is not what the paper says, and it is noisy.**
  Paper: "refined locally by using a finer grid in [E^{i0−1}, E^{i0+1}]". Code: the network trained at the grid point E^{i0} is **frozen** and `scipy.optimize.minimize_scalar(bounded)` is applied to
  E ↦ mean((Δu − Vu + Eu)²) with μ = 0, on 10 000 points **re-sampled at every function evaluation**. Consequences:
  1. the objective is the parabola (E − ρ)²·mean(u²) + const, so the output is a Monte-Carlo estimate of the Rayleigh quotient ρ(u) = −∫u(Δu − Vu)/∫u² – available in closed form with one batch;
  2. a deterministic bracketing minimiser on a stochastic objective cannot locate the minimum better than ≈ √(noise of the objective) ≈ (Λ_res/√N)^{1/2}·O(1) ≈ 0.03–0.1 for the residual levels
     in the tables – the size of the errors actually reported. The six decimals printed in every table are not significant;
  3. the "Residual" column is `result.fun`, one noisy evaluation at the returned E, on a different sample than training;
  4. the conclusions claim independence from the Rayleigh quotient (L847).
  Fix: compute ρ in closed form on a fixed large sample (or fix the seed inside `objective`), report 3–4 digits, describe the step truthfully:
  ```latex
  Once a local minimum $E^{i_0}$ of the discrete curve is detected, the network $u^{i_0}$ is kept fixed and the eigenvalue is taken as the minimiser of
  $E\mapsto\Lambda(u^{i_0},E)$ on $[E^{i_0-1},E^{i_0+1}]$, that is, the Rayleigh quotient
  $E=-\int_\Omega u^{i_0}(\Delta u^{i_0}-Vu^{i_0})\,dx\big/\int_\Omega (u^{i_0})^2dx$, evaluated with $10^4$ Monte Carlo points.
  ```

- **C-4 (M), L229. B(x).** Only B = 1 − |x|² (disk) is given. Code: square (x−x0)(x1−x)(y−x0)(x1−y); triangle x·y·(1−x−y); annulus (|x|−r_i)(r_o−|x|); balls 1 − |x|².
  R2 round 1 #3 asked for these; the reply pointed to the repository. Add a table (one line each). "a smooth approximation to the distance to the boundary" → "a smooth function vanishing exactly on ∂Ω
  and comparable to the distance to the boundary" (1 − |x|² ≈ 2 dist).
  ```latex
  \begin{tabular}{ll}
  \hline Domain & $B(x)$\\ \hline
  Ball $\{|x|<1\}\subset\R^d$, $d=2,3,4$ & $1-|x|^2$\\
  Square $(0,a)^2$, $a=1,2$ & $x_1(a-x_1)\,x_2(a-x_2)$\\
  Triangle $(0,0),(1,0),(0,1)$ & $x_1x_2(1-x_1-x_2)$\\
  Annulus $\{r_0<|x|<r_1\}$ & $(|x|-r_0)(r_1-|x|)$\\ \hline
  \end{tabular}
  ```

- **C-5 (M). μ: text vs code vs figures.**

  | experiment | text | notebook | file name | read from figure (M-2) |
  |---|---|---|---|---|
  | disk, V = 0 (Figs. 3–4) | 100 (L233) | not in notebook | – | ≈ 100 |
  | square [44,55] (Fig. 5) | 3000 (L323) | not in notebook | – | – |
  | square [0,105] (Table 1, Fig. 7) | not stated | 500 | 450 (loss), 150 (eigenfunctions) | 450 |
  | 3-D ball (Table 2, Fig. 6 top) | 200 (L382) | 400 | 100, "reseteo cada 15 iteraciones" | 100 |
  | 4-D ball (Table 3, Fig. 6 bottom) | 200 | 200 | 200 | 200 |
  | triangle | 15000 (L513) | 15000 | 15000 (comment L521: 1500) | – |
  | annulus | not stated | not in notebook | – | – |
  | harmonic ω = 1 / ω = 10 | "400" (L617, ambiguous) | 100 (ω = 1); ω = 10 absent | – | 100 / 100 |
  | double well | 400 | 400 | – | – |
  | four well | 1000 | 1000 (2000 samples) | – | – |
  | p-Lap disk | 4000 (L720) | 4000 | "150 + λ·λ" | – |
  | p-Lap square | 4000 | 1000 + E² | (comment L752: 100 + λ²) | – |
  | p-Lap 3-D, p = 2.5 | 4000? | not in notebook | – | – |

  A per-experiment hyper-parameter table (domain, B, E-window, grid points, μ, samples, max epochs, reset period, threshold ϵ, wall time) is needed; it would also answer R1 #1 and #3.

- **C-6 (M), L238. Training details vs code.** Matches: Adam, patience 300, tolerance 1e-5, 5000 points re-sampled each step, Xavier-uniform with tanh gain, zero biases, 3×128 tanh.
  Missing or different: learning rate 1e-3, constant (round-1 reply to R1 #3 / R2 #5 claimed the "learning rate schedule" had been added – it was not); "epoch" = one Adam step;
  max epochs is 2000–5000 in the notebook, not always 3000; samples 4000 (triangle) and 2000 (four-well); periodic **re-initialisation** of the network every 15–60 grid points
  (`reset_every`) is never mentioned although it visibly shapes the curves; the value returned is the loss of the *last* step, not the best; seed 0.

- **C-7 (M), L243. Threshold ϵ.** Never given. Notebook: 0.3 (harmonic), 2.0 (square), 2.9 (p-Lap disk), 3.5 (p-Lap square), 6.0 (triangle), 2.0 (4-D), 1.3 (3-D and wells) –
  chosen per example, evidently after looking at the curve, and as large as 6 for a quantity that is supposed to vanish. State the values and the rule (e.g. local minimum whose
  value is below a fraction of the neighbouring maxima), or drop the threshold and report all local minima with their certificate √Λ_res.

- **C-8 (M), L307, L838–843. Reproducibility claim.** "All computations and plots … were generated using the code provided" – the linked repository is `amsalort/PINN-eigenvalues`; this
  repository (`PINN-eigenvalues-v2`) says it "reproduces all experiments" but has no cell for: disk with V = 0 (Figs. 3–4), square [44,55] (Fig. 5), annulus (Fig. 9), harmonic ω = 10 (Fig. 12),
  p-Laplacian p = 2.5 in 3-D (Fig. 16). No notebook cell has stored outputs. Decide which repository is cited, tag a release, and make it produce every table.

- **C-9 (m), L602 vs code L471–485. Double well.** Text exp(−2|x ∓ x₀|²), x₀ = (1,0,0); code exp(−|x ∓ a e₁|²/(2σ²)), a = 1, σ = 0.5, V₀ = 1 → identical (1/(2σ²) = 2). **Consistent**
  (the coordinator's item 5 is not a discrepancy). What is missing in the text: the domain (unit ball of R³, Dirichlet), and the fact that the wells are centred *on the boundary*,
  so that inside the ball −1.0002 ≤ V ≤ −0.037.

- **C-10 (m), L604–615 vs code L488–508. Four wells.** a = 0.22, σ = 0.2, centres (±a,±a,0), bump exp(−1/(1−r̃²)): consistent. Text defects: "V(w) is defined as V(x) = …" (x is also the first
  coordinate of w = (x,y,z)); ", V₀ = 1" refers to a constant that is not in the formula; domain not stated. Depth: max|V| = e^{−1} = 0.368.

- **C-11 (M). The "complicated potentials" are tiny perturbations.** First-order perturbation of the ground state of the unit ball (π² = 9.8696): double well ⟨u₁,Vu₁⟩ = −0.292 → E₁ ≈ 9.58;
  four well −0.0153 → E₁ ≈ 9.854. The loss curves of Figs. 17–18 indeed show minima at ≈ 9.6, 19.9, 33.0, 39.3, 48.6, 59.5, 66.7 and ≈ 9.8, 20.1, 33.2, 39.4, 48.7, 59.6, 67.0, i.e. the free
  spectrum 9.87, 20.19, 33.22, 39.48, 48.83, 59.68, 66.95 shifted by ≤ 0.3. These examples do not test "robustness with respect to V" (R1 #2); the paper reports no numbers for them at all.

- **C-12 (M). Harmonic potential in the disk: exact reference values exist** (the text says they do not, L341). −Δ + (ω²/2)|x|² on the unit disk separates; radial FD + Richardson
  (`audit/scripts/potentials_reference.py`, checked at ω = 0 against Bessel zeros to 4 decimals):

  | | E (angular m, radial k; mult.) |
  |---|---|
  | ω = 1 | 5.8919 (0,1;1), 14.8484 (1,1;2), 26.5790 (2,1;2), 30.6271 (0,2;1), 40.9385 (3,1;2), 49.3852 (1,2;2) |
  | ω = 10 | 14.2847 (0,1;1), 29.1349 (1,1;2), 45.0887 (2,1;2), 46.2907 (0,2;1), 62.5793 (3,1;2), 66.2805 (1,2;2) |

  Fig. 13 (ω = 1) shows minima at all six values. Fig. 12 (ω = 10) shows three minima on [0,55]: 45.09 and 46.29 are merged (grid step 0.7, gap 1.2) – see finding 8.

- **C-13 (C), L570–577. Annulus.** Cross-product Bessel zeros for 0.5 < r < 1: m = 0: 39.0133; m = 1: 40.8725 (double); m = 2: 46.4285 (double); m = 3: 55.6179. The figure's orange curve
  has a single zero near 39–40 and the PINN curve rises monotonically to ≈ 120 at E = 50, where the true Λ is (50 − 46.43)² = 12.7. So (i) the reference curve is drawn with one eigenvalue only,
  (ii) the PINN followed the radial branch and missed the m = 1 and m = 2 eigenvalues, (iii) with ~30 grid points on [0,50] (step ≈ 1.7) 39.01 and 40.87 could not have been separated anyway.
  No eigenvalue is quoted for the annulus in the text. Redo the experiment with a finer grid and re-initialisation, give a table, or present it honestly as an example of a missed eigenvalue.

- **C-14 (M). Evidence in the paper's own figures of optimisation artefacts** (relevant to R1 #6): peaks above the exact curve and shifted to the right (triangle near E = 76: ≈ 750 vs 600;
  3-D near E = 28: 48 vs 42; 4-D peaks shifted by ≈ 1–2) – warm-start hysteresis, the network stays on the previous eigenfunction past the midpoint; small spurious local minima
  (3-D near E ≈ 49.5, harmonic ω = 10 near E ≈ 49–50); a discontinuity where the network is re-initialised (p = 2.5 curve at E ≈ 63). None is discussed.

- **C-15 (m).** `sweep` docstring says "Tip: use mu ∝ E²"; `loss_fn` docstring states the p-problem as −div(|∇u|^{p−2}∇u) = E u. `generate_report` pairs the k-th detected minimum with the k-th exact
  value, which mislabels everything after a missed or spurious minimum.

---

## 4. Presentation

### 4.1 Captions, figures, cross-references

| ID | Sev | Line | Problem | Fix |
|---|---|---|---|---|
| F-1 | M | L804 | "p=2.5" (not math) inside the section whose text and tables are all p = 2.2; the text L722 does not state p; "unit disk" for a 3-D domain; quotes 4 decimals; values wrong (P-1) | "… for the $p$-Laplacian with $p=2.5$ in the unit ball of $\R^3$." State p in the text; move numbers to the text with validation against 14.111 |
| F-2 | M | L824, L834 | "in three-dimensional space … in the unit disk"; "Compactly" capitalised; left panel is the potential, not said | "Double-well Gaussian potential (left) and loss curve $\bar\Lambda(E)$ (right) in the unit ball of $\R^3$." |
| F-3 | M | L820–836 | Figures 17–18 (linear problems of §4.4) are placed after the tables of §5 and inside §5 | move to §4.4 after L617 |
| F-4 | M | L550–568 | Figures 7–8 (square) are placed in §4.3 after Table 4 although discussed in §4.1 (L339) | move next to Table 1 |
| F-5 | m | L602 | Fig. 13 (ω = 1) is cited before Fig. 12 (ω = 10) | swap the two figure blocks |
| F-6 | m | L575 | "unit annulus of radii 0.5 and 1"; "Below it is displayed the first eigenfunction" but the panels are side by side | "in the annulus $\{0.5<|x|<1\}$ … (left); first computed eigenfunction (centre, right)." |
| F-7 | m | L482 | "(above)/(below)" depends on line breaking of two `.5\columnwidth` images; "the four-dimensional space" | use `(left)/(right)` with `.48\columnwidth`, or two subfigures |
| F-8 | m | L158, L247 | "$2-$dimensional"; "confined potential"; Fig. 2 is the exact Λ(E) but the text L151 calls it "the upper bound" | "two-dimensional"; "confining potential"; "The exact loss curve $\Lambda(E)=\min_k(E-E_k)^2$ for the unit disk." |
| F-9 | m | all loss-curve captions | "compared with the theoretical upper bound" while the legends say Λ(E) | "compared with the exact curve $\Lambda(E)=\min_k(E-E_k)^2$" (and see M-2 for why they differ at the peaks) |
| F-10 | m | L797 | caption without final period (editor's requirement) | add "." |
| F-11 | m | L787, L797, L804 | "p-Laplacian" in text mode; $\bar\Lambda$ for the p-functional | `$p$-Laplacian`, `$\bar\Lambda_p(E)$` |
| F-12 | m | L637, L646 | "in two-dimensional disk" | "in the unit disk" |
| F-13 | m | L592 | "Third and fourth computed eigenfunctions" – panel titles 127.75 and 168.59 disagree with Table 4 (T4-c) | regenerate from the run of the table |
| F-14 | m | L346 | "[44,55]" in text mode | `$[44,55]$` |
| F-15 | m | L513 | "Fig.~\ref{…} and \ref{…}" | "Figs.~\ref{…} and \ref{…}" |
| F-16 | m | L132 | "the \emph{Nonlinear Problems} section \ref{sec.nolin}" | "Section~\ref{sec.nolin}" |

All tables and figures are cited in the text (checked: 18 figures, 6 tables).

### 4.2 Notation clashes

- **ϵ / ε (M).** `\varepsilon` = early-stopping tolerance 1e-5 (L238 and five repeats); `\epsilon` = detection threshold (L243); `\epsilon` = Young parameter (L889–893). Use `\mathrm{tol}` or δ for the
  tolerance, τ for the threshold, ε only in the proof.
- **n / d** for the dimension (M-11); **j** as last grid index (L165) and as sequence index (L877); **E^i** superscripts look like powers (L165–169, L235) – use E_i or E^{(i)};
  **u_k** (eigenfunction) vs **u^i** (trained network) vs **u_j** (minimising sequence); **NN(x)** (L198) vs **ũ(x)** (L201, L207) for the raw output; **L(u)** vs **L** (M-10);
  **E_{(m,n)}** vs **E_{m,n}** vs **E_{(1)}** vs **E_k**; **D** (L661) vs **Ω** for the domain; **x** as point and as coordinate (L604–614).
- `$p-$Laplacian`, `$n-$th`, `$k-$th`, `$E^i-$grid`, `$2-$dimensional` typeset a minus sign: write `$p$-Laplacian` etc. (L158, L251, L298, L405, L654, L690, L720, L722, L737, L775).

### 4.3 Typos and grammar (exact replacements)

| Line | Text | Replace by |
|---|---|---|
| L138 | "This is the powerfulness of the proposed method." | "This is the main strength of the proposed method." |
| L151 | "a confining potential given by" / "the upper bound of Λ(E)" | "the hard-wall potential" / "the curve Λ(E)" |
| L169 | "zeroes correspond with eigenvalues" | see M-3 |
| L172 | "whose implementation is simple and no need of previous eigenvalues is required" | "whose implementation is simple and which does not require the previously computed eigenvalues" |
| L173 | "there is efficient and well-studied methods" | "there are efficient and well-studied methods" |
| L173 | "several adventages over other method of the literature as explained as follows" | "several advantages over other methods in the literature, which we now list" |
| L179 | "Advantages of our method:" | fold into the previous sentence |
| L188 | "shown ir our examples" | "shown in our examples" |
| L198 | "a  fully connected neural network,  implemented as" (double spaces); "$x\in\R^d$" | fix spacing |
| L214 | "the simpler case, where confining potentials, as described in the previous section, are considered. So, we let" | "the simplest case, the hard-wall potential" |
| L239 | "we obtain the discrete loss curve." (period before the display) | remove the period; put it after the display |
| L253 | "displayed in Fig.~\ref{func.fig}" | add final period |
| L285 | leading space / forward reference | see M-4.4 |
| L323, L382, L513, L617, L720 | sentence "an early stopping mechanism terminates training if …" repeated verbatim six times | state once at L238, delete the repeats |
| L339 | "For illustrative purposes, we compute…" (after the [44,55] paragraph, same domain) | merge; give the window [0,105] |
| L375 | "with $d=3,4$ where" | "with $d=3,4$, where" |
| L405 | "exact formulas for the eigenvalues are known, for $m,n\in\N$, the eigenvalues are computed as" | "the eigenvalues are known explicitly: for $m\in\N_0$, $n\in\N$, $E_{(m,n)}=j_{m+\frac12,n}^2$ with multiplicity $2m+1$, where $j_{\nu,n}$ is the $n$-th positive zero of $J_\nu$. For $d=4$, $E_{(m,n)}=j_{m+1,n}^2$ with multiplicity $(m+1)^2$." |
| L461 | "compared wih" | "compared with" |
| L513 | "more complex geometries, such as squares, annuli, or triangles" | "other geometries: squares, annuli and triangles" (none is complex) |
| L515 | "in a triangle" | "in the triangle with vertices (0,0), (1,0), (0,1)" |
| L602 | "displays loss curve" | "displays the loss curve" |
| L604 | "More precisely, given $w=(x,y,z)$, and define the centers" | "More precisely, for $x\in\R^3$ let the centers be" |
| L612–615 | "the potential $V(w)$ is defined as $V(x)=…, \quad V_0=1.$" | "$V(x)=-\sum_{i=1}^4\phi(|x-c_i|/\sigma)$." |
| L617 | "In four  three examples we have chosen μ=400, μ=400 and μ=1000, respectively" | "In the harmonic, double-well and four-well examples we used μ = …, …, … respectively" (with the true values, C-5) |
| L692 | "values of E" | "values of $E$" |
| L737 | "planar unit ball" | "unit disk" |
| L775 | "planar square of lenght 2" | "square $[0,2]^2$" |
| L857 | "which inspired the author" | "which inspired this work" |
| L893 | "$\epsilon=1/4E$" | "$\varepsilon=1/(4|E|)$" |
| L909 | "immersion" | "embedding" |
| (word "lineal" from the coordinator's list does not occur in v3; "wih", "lenght", "adventages", "shown ir", "In four three" do) | | |

### 4.4 Structure, duplication

- **S-1 (M), L103–125.** Duplicated derivation (M-1). **S-2 (m), L141–149.** Dead commented derivation; L326–336, L389–401, L432–446, L470–499, L517–526, L618–625, L695–714, L744–764
  are commented blocks with parameters that contradict the text (μ = 500, 1500, 100 + λ²; "Example 6/7"; an arXiv URL). Remove before submission – the source is public on arXiv.
- **S-3 (M).** §2 ends with a list of "advantages" (L179–192) before any result; §3 "Implementation details" contains the first experiment and a long remark on Weinstein; §4.1 "Higher
  eigenvalues" contains the μ discussion and the square; §4.3 "Different geometries" shows the square again. Suggested order: 1 Introduction (with roadmap) – 2 Method (loss curve, identity,
  penalised landscape, certificate) – 3 Implementation (architecture, B table, training, detection/refinement, cost) – 4 Linear experiments (hyper-parameter table; disk; square; balls; triangle;
  annulus; potentials with reference values) – 5 p-Laplacian – 6 Limitations – 7 Conclusions – Declarations – Appendix.
- **S-4 (m), L838–843.** "Code availability" sits before the Conclusions as an unnumbered section; move into Declarations.
- **S-5 (M). Overstatements to tone down.**
  - L38 "independent of the space dimension" → tested for d ≤ 4 only; MC with 5000 points and a product-form B do not make it dimension independent.
  - L184 "our approach has been shown to be effective regardless of the dimensionality" → "we report experiments in dimensions two, three and four".
  - L186 "robust and works effectively independent of the structure of the potential function V or the geometry" → contradicted by the annulus (C-13) and ω = 10 (C-12); potentials tested are small perturbations (C-11).
  - L188 "producing novel results in several contexts" → no new validated value is reported.
  - L191 "readily adapted to more general operators beyond the p-Laplacian" → "in principle".
  - L310 "we can directly recover eigenvalues that are far from the principal one" → highest reported: 17π² = 167.8 (4th eigenvalue of the triangle); the 7th distinct eigenvalue in the 3-D ball.
  - L341 "exact eigenvalues are not available" → reference values are cheap (C-12).
  - L722 "To the best of our knowledge, no previous numerical results are available" → false for balls (radial ODE; inverse-iteration computations of λ1 for balls in several dimensions exist, e.g. Biezuner–Ercole–Martins, J. Funct. Anal. 2009 – to be verified by the writer). Replace by a comparison with the shooting value.
  - L847 "does not rely on the Rayleigh quotient" → see C-3.
  - L849 "naturally extends to irregular domains and high-dimensional settings, where traditional mesh-based methods become increasingly difficult" → all domains have polynomial B; d ≤ 4, where FEM/spectral methods are far more accurate (the authors concede this in the round-1 reply).
  - L61 "The eigenvalue problem itself has been more recently addressed: first in one dimension [2020]" → NN eigen-solvers go back to Lagaris–Likas–Fotiadis (1997, quantum eigenproblems) and the Deep Ritz paper (2018) treats eigenvalues; R2 round 1 #7 asked for Deep Ritz and it is still not cited.

### 4.5 References (L916–1019)

- **R-1 (M). Order.** The list is alphabetical; the editor requires numbering in order of first citation. Order of first citation:
  1 Lu–Meng–Mao, 2 Rackauckas et al. (UDE), 3 Rackauckas–Innes et al. (DiffEqFlux), 4 Blechschmidt–Ernst, 5 Berg–Nyström, 6 Ghazi–Tawfiq, 7 Kumar–Kumar–Goswami, 8 Madani–Hammouch–Özdemir,
  9 Lagaris–Likas–Fotiadis, 10 Raissi–Perdikaris–Karniadakis, 11 Karniadakis et al., 12 Jin–Mattheakis–Protopapas (2020), 13 idem (2022), 14 Sarkar, 15 Holliday–Lindner–Ditto, 16 Rowan et al.,
  17 Weinstein, 18 Parlett, 19 Horák. All 19 `\bibitem`s are cited and all citations resolve. (A `\cite{a,b,c}` list should itself be in order: L47 cites Lu, Rackauckas, Rackauckas–Innes.)
- **R-2 (M), L1009.** `\href{arXiv:2504.05367} (2025)` has one argument; hyperref eats " (" as link text and the identifier is **not printed**: the PDF shows "[18] S. Sarkar, …, arXiv preprint (2025)."
  Fix: `\href{https://arxiv.org/abs/2504.05367}{arXiv:2504.05367} (2025).` Key `Sakar` is a misspelling of Sarkar.
- **R-3 (m).** Mixed styles: journal names italic in one entry (L937) and roman elsewhere; Rowan et al. (L1005) in a different format and without year; Raissi et al. without final period (L999);
  Kumar et al. without pages (L966); Horák without volume/year structure ("Electron. J. Differential Equations 2011, No. 132, 1–30"); proceedings without pages; no DOIs.
  Bibitem keys `Raissi-Perdiakaris-Karniadakis`, `Rowan-Evans-Maute-Dootsan` misspell Perdikaris, Doostan (harmless).
- **R-4 (M). Relevance** (editor: "only very very close papers"). [GT24], [KKG], [MHO] (L929–932, L963–966, L978–981; cited once at L53 for "three-dimensional PDEs, Emden–Fowler systems and
  fractional-order predator–prey models") are unrelated to eigenvalue problems or PINNs proper; the two software references (DiffEqFlux, UDE) are marginal. Candidates to drop.
  Missing close work: Deep Ritz (E–Yu 2018); Lagaris et al. 1997 (quantum eigenproblems); Ben-Shaul et al. 2023; method of particular solutions / folded spectrum (M-8); Kato–Temple; Krylov–Bogoliubov is named (L263) but not cited;
  p-Laplacian eigenvalue numerics beyond Horák. Bibliographic data of every addition must be checked against the publisher – none was verifiable from this sandbox (arxiv.org blocked).
- **R-5 (m).** `\cite{HJ11}` key suggests two authors; the paper is by J. Horák alone (entry is right).

---

## 5. Editorial requirements of IJMCE (round-1 editor list) – status in v3

| Requirement | Status in v3 | Evidence / action |
|---|---|---|
| IJMCE LaTeX format | **unmet** | L1 `\documentclass[11pt]{amsart}` with custom geometry; file name still "rev2-arxiv". Port to the journal template (not available in this sandbox – obtain it). |
| PDF + zip of sources | n/a here | 43 PNGs in `paper/v3/`; every one is used by an active `\includegraphics` and every target exists (checked). |
| Longer abstract | **unmet** | L37–39: three sentences, unchanged since v1, no results, contains the overclaim "independent of the space dimension". |
| AMS 2020 codes | met | L34: 35P05, 35P30, 35J92, 65N25, 68T07 (35P15 "estimates of eigenvalues" would fit better than 35P05). Keywords present (L35). |
| References cited in numerical order | **unmet** | R-1. |
| Numbered sections | partly | Sections 1–6 numbered; "Code availability", "Funding", "Acknowledgements" unnumbered and scattered; appendix "Appendix A". |
| Roadmap at the end of the Introduction | **unmet** | Introduction ends at L88 with the list of limitations; no outline paragraph anywhere. |
| All tables/figures cited in the text | met | checked; but citation order ≠ appearance order (F-3, F-4, F-5). |
| Period at the end of captions | **unmet** (1) | L797. |
| Declarations section per journal format | **unmet** | only Funding (L851) and Acknowledgements (L855). Needed (typical IJMCE/Sciendo list – confirm with the template): Conflict of interest, Funding, Author's contributions, Acknowledgement, Data availability statement, Using of AI tools. |
| References in journal format | **unmet** | R-2, R-3. |
| Remove irrelevant / self citations | partly | no self-citations; [GT24], [KKG], [MHO] weakly related (R-4). |
| Check for retracted papers | not verifiable offline | do a Retraction-Watch/Crossref check on the final list. |
| Tables in IJMCE format | **unmet** (likely) | vertical rules, `\begin{small}` environment wrapping `\centering`, captions below the tables, column specs with phantom columns (T5-d). Use captions above, horizontal rules only, as in the template. |

---

## 6. Reviewer #1, round 2 – what v3 already contains, and what is missing

| R1 point | Existing text in v3 that partially addresses it | Gap | Material from this audit that can be used |
|---|---|---|---|
| 1. Cost of the scan; complexity / runtime vs Rayleigh PINNs, many eigenvalues in a large interval | L235 (warm start); L172–173 ("our goal is not to obtain the best estimates"); L382 ("despite the increased computational cost"); L251 (local refinement) | No count of trainings, no wall time, no comparison. Round-1 reply conceded Rayleigh PINNs are cheaper for E1, E2 – not in the paper. | Cost = J trainings × (≤ n_epochs Adam steps) × N points × cost of one Laplacian by autograd (d second-order backward passes, `_laplacian`). From the notebook: J = 50–120, n_epochs = 2000–5000, N = 2000–5000, i.e. up to 80 × 5000 = 4·10⁵ Adam steps for Table 1 (4 eigenvalues) – roughly J/(number of eigenvalues wanted) times a sequential Rayleigh/deflation PINN. Honest scaling: to resolve eigenvalues the step must be below the smallest gap; by Weyl's law N(E) ~ c_d|Ω|E^{d/2}, so a full scan of [0,E] needs J ≳ (several)×N(E) trainings – no asymptotic gain over deflation when *all* eigenvalues up to E are wanted; the gain is for a narrow window at high E (cost independent of the number of lower eigenvalues), which is what Fig. 5 illustrates. `generate_report` already records `elapsed_time`: the numbers just need to be put in a table. |
| 2. Independent validation for the double-well / four-well cases (e.g. FEM) | L341 ("exact eigenvalues are not available for comparison"); Figs. 12, 13, 17, 18 (curves only, no numbers) | No eigenvalue is reported for any potential, hence nothing is validated. | C-12: exact reference values for the harmonic disk (ω = 1, 10). C-11: first-order estimates E₁ ≈ 9.58 (double well), 9.854 (four well); both potentials are small perturbations of the free ball. A proper reference (FEM or a spectral Galerkin in the ball eigenbasis) is still to be computed by the experiments agent; then add a table "PINN vs reference" for the first 5–7 eigenvalues of each potential. Also disclose finding 8 (merged pair for ω = 10). |
| 3. μ: no guideline; sensitivity study | L312–320 (heuristic μ ∝ E²); values at L233, L323, L382, L513, L617, L720 | No rule that explains 100/200/400/3000/4000/15000; no sensitivity experiment; round-1 reply declined one. Stated values partly wrong (C-5). | M-2: exact formula for Λ̄(E), threshold μ > d_max/2 for avoiding the trivial state, independence of the minimisers from μ, and the prediction "plotted residual = d − d²/(2μ) ≤ μ/2", already confirmed on five of the paper's figures. A sensitivity table (same window, μ ∈ {d_max/4, d_max/2, d_max, 10 d_max, 100 d_max}: detected eigenvalues, ‖u‖², residual) is the experiment to run; the theory predicts collapse to u ≡ 0 below d_max/2 and slower/ill-conditioned training for very large μ. |
| 4. Update the reference in the third paragraph of the Introduction (generalised higher-order NLS, multi-peak/breather solitons) | L49–53 is the paragraph meant (Burgers, Eikonal, heat, diffusion, 3-D PDEs, Emden–Fowler, predator–prey) | Not cited; round-1 reply (R1 #9) declined as out of scope. | Editorial decision. It conflicts with the editor's "only very very close papers" rule. Options: (a) one neutral clause at L53 ("…and nonlinear Schrödinger-type models [ref]") – bibliographic data must be looked up, I could not verify authors/journal from here; (b) decline again, citing the editor's instruction, and instead add one sentence in a Limitations paragraph on NLS standing waves −Δu + Vu − |u|^{2σ}u = Eu (the residual loss applies verbatim; bifurcating branches make "eigenvalue" normalisation dependent – exactly the phenomenon of C-1). |
| 5. p-Laplacian errors > 5 %; dominant error sources; convergence under grid refinement / network capacity | L685–688 (only an upper bound; shape between eigenvalues not determined); L285–288 (same behaviour as linear case); L692 | No analysis, no convergence study. | Error sources identified here: (i) C-1 – the code does not solve equation (6) (wrong zero-order term and normalisation); decisive in 3-D (P-1), possibly in Tables 5–6; (ii) M-6.5 – smooth ansatz vs C^{1,α} eigenfunctions: non-vanishing strong residual near critical points, loss floor ≈ 1–2 in Fig. 14; (iii) coarse grids (step 1.0 on the disk, 0.41 on the square, ≈ 1.7 in 3-D) and the noisy refine (C-3); (iv) c_k ≠ 1 – the curvature of the loss curve at an eigenvalue is c_k, so the threshold/refinement behave differently; (v) the reference values are themselves numerical (Horák; λ1 = 6.5320 vs shooting 6.5317). New reference values available for convergence plots: radial λ1 for any (p,d) by shooting – disk p = 2.2: 6.53174; 3-D ball p = 2.2: 11.46991; 3-D ball p = 2.5: 14.11123; second radial: 41.7886, 54.6063, 87.4923. |
| 6. Theorem vs finite-dimensional NN; effect of optimisation/approximation errors on Λ(E) = min(E−E_k)²; spurious minima | L274–282 (floor η(E) "does not shift the minimiser"); L243 (threshold); Appendix | No statement of what survives for a network class; no discussion of spurious or missed minima. Round-1 reply to R2 #6: the theorem "is not intended to establish full mathematical rigor". | A rigorous one-sided statement is available and cheap: every admissible network function u = B·NN lies in D(L), hence Λ(u,E) ≥ dist(E,σ(L))² for *every* u. Therefore Λ_NN(E) := inf over the network class ≥ Λ(E) and equals Λ(E) + approximation error ≥ 0; the trained value adds an optimisation error ≥ 0. Consequences: (a) with exact integrals a small value Λ(u,E) = r² *certifies* an eigenvalue within r of E (no spurious eigenvalue below the threshold: "no spectral pollution"); only Monte-Carlo error can violate this, of size O(N^{−1/2}); (b) errors are not uniform in E, so spurious *local minima with positive value* and *missed* eigenvalues can occur – and do: C-14 (hysteresis peaks, small spurious dips), C-13 (annulus: two eigenvalues missed), C-12 (ω = 10: merged pair), M-2 (collapse to u ≡ 0 when d > 2μ); (c) multiplicity is invisible. Same for the penalised loss: Λ̄(u,E) ≥ d − d²/(4μ). Suggested paragraph: ```Let $\mathcal N$ be the set of functions $u=B\,\mathrm{NN}_\theta$. Since $\mathcal N\subset D(L)$, \eqref{cota.igual} gives $\Lambda_{\mathcal N}(E):=\inf_{u\in\mathcal N}\Lambda(u,E)\ge\Lambda(E)$ for every $E$: approximation and optimisation errors can only raise the loss curve. Consequently any pair $(E,u)$ produced by the algorithm with $\Lambda(u,E)=r^2$ certifies an eigenvalue in $[E-r,E+r]$, whatever the quality of the training; false detections can only come from the Monte Carlo quadrature. The converse fails: an eigenvalue is missed whenever the optimiser does not reach the corresponding eigenfunction, and local minima with positive value may appear; this is why a threshold is used and why the scan is restarted from random weights periodically.``` |
| 7. Summary (more validation, parameter analysis, efficiency, nonlinear accuracy) | – | – | Items above; plus fix the table errors (T1-a, T3-a, T4-a) before any new experiment is added, since the reviewer will re-read the tables. |

Round-1 commitments that v3 does not honour (the reviewer may check): learning rate / schedule "added" (not in the text, C-6); explicit B(x) for triangle and annulus (only in the repository, C-4);
"the zeros correspond to absolute minima" (now correct in substance, L103–125, but circular, M-1); Deep Ritz comparison (not cited).

---

## 7. Things I could not verify

- Horák's tables (EJDE 2011, No. 132): only λ1 of the disk at p = 2.2 was cross-checked (by shooting). 18.395, 5.4952, 15.144, and that his square is [0,2]², were not checked against the source.
- Bibliographic details of [GT24], [KKG], [MHO], Rowan et al., Sarkar, and of every reference suggested in M-8 / R-4 / S-5.
- The IJMCE template and its exact Declarations headings.
- Which code produced the published numbers (the `amsalort/PINN-eigenvalues` repository was not inspected); statements about μ "read from figures" rely on the closed form of M-2.
- No PINN run was repeated; all statements on tables are consistency checks, not reproductions.
