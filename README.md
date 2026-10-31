# A sharp bound for the least signless Laplacian eigenvalue of triangle-free graphs

This repository contains the readable Python calculation programs, exact integer data, precomputed outputs, and paper proving the sharp bound conjectured by [Balogh, Clemen, Lidický, Norin and Volec (2023)](https://doi.org/10.1137/22M150767X):

$$q_{\min}(G)\leq \frac{7}{50}|V(G)|$$

for every finite simple triangle-free graph. Here $q_{\min}(G)$ is the least eigenvalue of the signless Laplacian matrix $Q_G=D_G+A_G$, where $D_G$ is the diagonal matrix of vertex degrees and $A_G$ is the adjacency matrix. The constant is sharp, since the Higman–Sims graph and its balanced blow-ups attain equality. The paper also proves that these are the only graphs attaining equality, up to isomorphism.

- **Paper:** [PDF](paper/main.pdf) · [LaTeX source](paper/main.tex).
- **Complete program guide:** [programs/README.md](programs/README.md), including inputs, outputs, data structures, and execution order.
- **Exact data:** [spectral_data.py](programs/spectral_data.py) and [sigma_data.py](programs/sigma_data.py).
- **Equality-case identities:** [verify_equality.py](programs/verify_equality.py) and its two [rational vectors](programs/equality_vectors.json).

All Python comments, docstrings, and error messages in the published programs are in English.

## Authors

Dongxiu Cai<sup>1</sup>, Zhengbo Chen<sup>1</sup>, and Xiao-Dong Zhang<sup>1,2</sup>.

<sup>1</sup> School of Mathematical Sciences, MOE-LSC, SHL-MAC, Shanghai Jiao Tong University, 800 Dongchuan Road, Shanghai 200240, P.R. China. Emails: [diudiutse@sjtu.edu.cn](mailto:diudiutse@sjtu.edu.cn) (Dongxiu Cai) and [czb911@sjtu.edu.cn](mailto:czb911@sjtu.edu.cn) (Zhengbo Chen).

<sup>2</sup> Corresponding author: Xiao-Dong Zhang. Email: [xiaodong@sjtu.edu.cn](mailto:xiaodong@sjtu.edu.cn).

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
python programs/verify_equality.py
```

The six generated pickle files are included, so these commands do not require regenerating graphs or matrices. `verify_39.py` checks conditions (ii) and (iii) of Section 4.2: strict diagonal dominance of `qI + X_star` and all `qI + X_sigma`. `verify_40.py` checks equation (60), the exact trace identity for each of the 410 eight-vertex triangle-free graphs. The filenames and their console messages retain the equation numbers (39) and (40) from the earlier draft. The success messages are:

```text
X_star and all X_sigma satisfy (39).
(40) passed: all 410 totals are exactly zero.
(61) passed: all 12 coordinates agree exactly.
(62) passed: all 57 coordinates agree exactly.
```

The second check prints progress every ten graphs and writes [programs/results/verify_40.json](programs/results/verify_40.json). It performs arbitrary-precision integer and rational arithmetic, so allow it to finish; the calculation is not a floating-point tolerance test. The checked matrices have 1,650 rows in total across 67 matrices.

`verify_equality.py` checks the two flag-vector identities in Lemma 5 of the paper, numbered (61) and (62). It recomputes the count vectors from the existing flag lists and compares the matrix-vector products exactly. Its only inputs are `programs/results/05/F_sigma.pkl`, the existing fixed matrix data, and `programs/equality_vectors.json`. The JSON contains only `b_one_root` and `b_two_roots`, with 10 and 54 rational entries, respectively. See the [equality verification guide](programs/README.md#equality-case-identities-lemma-5).

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
| [verify_39.py](programs/verify_39.py) | Check the strict row inequalities in conditions (ii) and (iii). | Console verdict |
| [verify_40.py](programs/verify_40.py) | Check equation (60) for all 410 graphs. | `programs/results/verify_40.json` |
| [verify_equality.py](programs/verify_equality.py) | Recompute the one-root and two-root flag counts and check the two identities in Lemma 5 of the paper. | Console verdict |
| [equality_vectors.json](programs/equality_vectors.json) | Supply only the two rational vectors used by `verify_equality.py`. | `b_one_root` (10 entries), `b_two_roots` (54 entries) |

To regenerate every output, follow the commands in the [program guide](programs/README.md#recompute-all-six-stages-and-verify). The fixed integer data modules are supplied and used directly by the verifiers. They are not regenerated by running the six numbered calculation programs.

## Repository layout

```text
README.md
requirements.txt
programs/
  README.md
  01_expand_F.py ... 06_compute_C.py
  spectral_data.py
  sigma_data.py
  verify_39.py
  verify_40.py
  verify_equality.py
  equality_vectors.json
  results/01/ ... results/06/
  results/verify_40.json
paper/
  main.tex
  main.pdf
  sections/
    auxiliary-matrices.tex
    definitions.tex
    proof.tex
    equality-cases.tex
  images/
    flag-ordering-structured-example.png
```

`programs/results/06/C.pkl` contains complete dense matrices and is approximately 91 MB. The code can regenerate it. Allow memory for these Python lists and the temporary matrix arrays. Load only trusted pickle files, such as the supplied files or outputs you generate yourself.

## Paper and verification scope

Section 1 introduces the result. Section 2 constructs the auxiliary matrices and gives a sufficient condition for the upper bound. Section 3 proves the sampling and flag-positivity identities. Section 4 constructs the fixed matrices and proves the upper bound through finite exact verification. Section 5 characterizes equality. Lemmas 1 and 4 give the reductions for the bound; Lemma 5 contains the two computational flag-vector identities used in the equality proof.

A numerical semidefinite program was used to find candidate matrices. Rational reconstruction and exact linear correction produced the supplied integer data. Running the verification programs requires no SDP solver, CVXPY, Clarabel, FLINT, or proof assistant. The checks are exact finite computations, not a completed Lean formalization.

## Statement on AI use

The authors used GPT-6.1 Sol in the preparation of this work, including the development of proof arguments, the checking of intermediate steps, the generation and execution of semidefinite programming (SDP) code and code for verifying inequalities and equalities, and the drafting and revision of the manuscript and its LaTeX source. All mathematical results, arguments, and proofs were independently reviewed and verified by the authors, who take full responsibility for the accuracy and content of this work.
