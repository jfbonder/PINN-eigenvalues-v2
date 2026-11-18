# Environment (recorded by infrastructure agent)

- Python 3.13.18, Linux, 4 CPUs, no GPU.
- `pip install torch numpy scipy matplotlib scikit-fem meshio pandas` works with plain PyPI
  (torch 2.15.0+cu132 wheel is pulled with nvidia-* deps but runs on CPU; ~GBs of unused packages).
  Versions: torch 2.15.0, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, scikit-fem 12.0.2,
  meshio 5.3.5, pandas 3.0.5. All imports verified.
- `torch.get_num_threads()` default = 4. Reference op: 10 x (2000x2000 @ 2000x2000) matmul:
  1 thread 1.03 s, 2 threads 0.60 s, 4 threads 0.56 s (i.e. memory/overhead bound beyond 2 threads).
- LaTeX: `pip` has no usable TeX (tectonic not on PyPI). `apt-get update` then
  `apt-get install -y --no-install-recommends texlive-latex-base texlive-latex-recommended
  texlive-latex-extra texlive-science texlive-fonts-recommended texlive-publishers latexmk` WORKS
  (TeX Live 2023). `pdflatex`, `latexmk`, `bibtex` available. paper/v3/FBS-rev2-arxiv.tex (amsart)
  compiles to 17 pages with `latexmk -pdf` (tested in a scratch copy, not in paper/).
- ctan.org / arxiv.org blocked (403); GitHub clone works.

## Cost per epoch (3x128 tanh, 5000 points, Adam, no early stop; experiments/smoke/bench_epoch.py)
ms/epoch (3000 epochs = x3 seconds); noisy +-15%:

| case | 1 thr | 2 thr | 4 thr |
|---|---|---|---|
| 2D disk, Laplacian | 135 | 98 | 54 |
| 2D disk, p=2.2 | - | 94 | 48 |
| 3D ball, Laplacian | 184 | 89 | 77 |
| 4D ball, Laplacian | 233 | 136 | 112 |

One E-point at the paper settings (3000 epochs, no early stop): 2D ~ 2.5-5 min (2 thr: ~290 s,
4 thr: ~160 s), 3D ~ 4-5 min, 4D ~ 5.5-7 min (2 thr). Early stopping (patience 300, tol 1e-5)
may cut this. With 2000 epochs (3D/4D examples in the notebook) multiply by 2/3.
Scaling: 4 threads gives only ~1.2-1.8x over 2 threads, and 2 threads ~1.4-2x over 1,
so 4 single-thread or 2x2-thread concurrent jobs give the best throughput.
Smaller net (3x64, 2000 pts, 2D square): ~11 ms/epoch (2 thr), 600 epochs = 7 s per E.
Eval on 20000 fresh points adds ~0.3 s per E.
A full sweep of e.g. 80 E-points x 3000 epochs in 2D at 2 threads ~ 6-7 h; the sweeps are
sequential in E (warm start), so parallelise ACROSS sweeps/seeds/problems, not across E.
