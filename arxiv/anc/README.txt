Exact certificate for the sharp triangle-free signless Laplacian bound

Run these programs in this directory with ordinary Python 3, NumPy and NetworkX:
    python verify_all.py
    python verify_collisions.py
Do not use -O or -OO; the programs explicitly reject optimized execution.

verify_all.py rebuilds spectral expansions, source lifts, flags and counts,
then checks all 410 exact coefficient identities and 1650 positive row margins.
verify_collisions.py independently constructs the complete catalog by induction
and checks exact conditional flag and spectral identities with repeated classes.

The six inputs are pinned by manifest.json. The certificate is unchanged from
the original exact recovery. Historical discovery-file hashes inside it are
provenance and do not need to be resolved to run these portable verifiers.
results.json and collision-source-results.json record actual successful runs.
Integers/Fraction are exact. Graph isomorphism uses NetworkX.
Tested environment: CPython 3.14.6, NumPy 2.5.1, NetworkX 3.5.

The programs adapt an independently written workspace review implementation;
the manuscript gives the mathematical argument supporting the finite checks.
