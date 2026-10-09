# A sharp bound for the least signless Laplacian eigenvalue of triangle-free graphs

This repository contains the readable Python calculation programs, exact integer data, precomputed outputs, and manuscript for the bound

$$q_{\min}(G)\leq \frac{7}{50}|V(G)|$$

for every finite simple triangle-free graph. Here $q_{\min}(G)$ is the least eigenvalue of the signless Laplacian $Q_G=\operatorname{diag}(\deg_G(v))+A_G$. The constant is sharp for the Higman–Sims graph and its balanced blow-ups.

- **Paper:** [PDF](arxiv/main.pdf) · [LaTeX source](arxiv/main.tex).
- **Complete program guide:** [programs/README.md](programs/README.md), including inputs, outputs, data structures, and execution order.
- **Exact data:** [spectral_data.py](programs/spectral_data.py) and [sigma_data.py](programs/sigma_data.py).

## Quick verification using the supplied outputs

Python 3.10 or later is required. Install NumPy and NetworkX in a virtual environment:

```bash
git clone https://github.com/Diudiutse/triangle-free-signless-laplacian.git
cd triangle-free-signless-laplacian
python -m venv .venv
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then run:

```bash
python -m pip install -r requirements.txt
python programs/verify_39.py
python programs/verify_40.py
```

The six generated pickle files are included, so these commands do not require regenerating graphs or matrices. Equation (39) checks strict diagonal dominance of `qI + X_star` and all `qI + X_sigma`. Equation (40) checks the exact trace identity for each of the 410 eight-vertex triangle-free graphs. The success messages are:

```text
X_star and all X_sigma satisfy (39).
(40) passed: all 410 totals are exactly zero.
```

The second check prints progress every ten graphs and writes [programs/results/verify_40.json](programs/results/verify_40.json). It performs arbitrary-precision integer and rational arithmetic, so allow it to finish; the calculation is not a floating-point tolerance test. The checked matrices have 1,650 rows in total across 67 matrices.

## What each file does

| File | Task | Output or supplied data |
| --- | --- | --- |
| [01_expand_F.py](programs/01_expand_F.py) | Expand all 25 entries of $\mathsf F$ and collect coefficients of isomorphic graph monomials. | `programs/results/01/c_ij_J.pkl` |
| [02_generate_tau8.py](programs/02_generate_tau8.py) | Enumerate the 410 ordinary isomorphism classes of triangle-free graphs on eight vertices. | `programs/results/02/tau_8.pkl` |
| [03_compute_F_H.py](programs/03_compute_F_H.py) | Evaluate $t_{\mathrm{inj}}(J,H)$ and use the expansion coefficients to compute each $F_H$. | `programs/results/03/F_H.pkl` |
| [04_generate_sigma.py](programs/04_generate_sigma.py) | Enumerate the 66 type graphs of orders zero through six. | `programs/results/04/sigma_list.pkl` |
| [05_generate_flags.py](programs/05_generate_flags.py) | Enumerate flags for each type, with the first $k$ vertices fixed under isomorphism. | `programs/results/05/F_sigma.pkl` |
| [06_compute_C.py](programs/06_compute_C.py) | Count all 27,060 matrices $C_H^\sigma$ in those flag orders. | `programs/results/06/C.pkl` |
| [spectral_data.py](programs/spectral_data.py) | Supply the fixed integers $L,q,\Delta,X_*$. | Python data module |
| [sigma_data.py](programs/sigma_data.py) | Supply $V_\sigma,X_\sigma,a_\sigma$ for every type. | Python data module |
| [prepare_readable_data.py](programs/prepare_readable_data.py) | Convert the original integer data to the generated type and flag orders. | The preceding two data modules; `programs/results/data_conversion.json` |
| [verify_39.py](programs/verify_39.py) | Check all strict row inequalities (39). | Console verdict |
| [verify_40.py](programs/verify_40.py) | Check all 410 exact identities (40). | `programs/results/verify_40.json` |

To regenerate every output, follow the commands in the [program guide](programs/README.md#recompute-all-six-stages-and-verify). The fixed integer data modules are supplied; `prepare_readable_data.py` only rewrites their representation and ordering, and does not rerun the SDP.

## Repository layout

```text
requirements.txt
programs/
  01_expand_F.py ... 06_compute_C.py
  spectral_data.py
  sigma_data.py
  prepare_readable_data.py
  verify_39.py
  verify_40.py
  results/01/ ... results/06/
arxiv/
  main.tex
  main.pdf
  anc/                 original exact data and separate portable verifiers
```

`programs/results/06/C.pkl` contains complete dense matrices and is approximately 91 MB. The code can regenerate it. Allow memory for these Python lists and the temporary matrix arrays. Load only trusted pickle files, such as the supplied files or outputs you generate yourself.

## Separate verification implementation

The original exact data and additional portable verifiers are supplied in [arxiv/anc/](arxiv/anc/). Run them from the repository root:

```bash
python arxiv/anc/verify_all.py
python arxiv/anc/verify_collisions.py
```

Do not pass Python's `-O` or `-OO` options. `verify_all.py` rebuilds expansions, flags and counts and checks the original data. `verify_collisions.py` checks the catalog and exact sampling identities including repeated sampled values. Their input hashes are listed in [manifest.json](arxiv/anc/manifest.json). These are additional checks; the readable step-by-step pipeline is in `programs/`.

## Manuscript and verification scope

The paper gives the mathematical reductions and sampling arguments. Its Sections 1–3 are Introduction, Preliminary, and Proof of the main theorem. Lemmas 8–10 connect the matrix inequality, the finite trace identities, and their exact verification; equations (39) and (40) correspond to the two readable verification programs.

A numerical semidefinite program was used to find candidate matrices. Rational reconstruction and exact linear correction produced the supplied integer data. Running the published verification programs requires no SDP solver, CVXPY, Clarabel, FLINT, or proof assistant. The checks are exact finite computations, not a completed Lean formalization.

To build the paper with an existing LaTeX installation:

```bash
cd arxiv
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The current manuscript is a review draft, with the author field left for the submitting author. This repository publication is not an arXiv submission.
