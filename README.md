# ML Algorithms From Scratch

Five classic machine learning algorithms implemented from first
principles with NumPy only — no scikit-learn inside the implementation
itself. Each one is verified two ways: against a known analytical
correctness case, and against scikit-learn's own implementation as an
independent check.

```
algorithms/
  linear_regression.py      batch gradient descent, standardized internally
  logistic_regression.py    batch gradient descent on cross-entropy
  decision_tree.py          CART-style, recursive Gini-impurity splits
  kmeans.py                 Lloyd's algorithm with multiple random restarts
  bayesian_regression.py    closed-form Gaussian posterior, no sampling
```

Run `python3 demo.py` to see every scratch model run side by side with
its scikit-learn equivalent on the same data.

## Why build these by hand

A library call hides the part that is actually worth understanding: what
gradient descent is doing step by step, why a greedy decision tree can
get stuck on certain patterns, why k-means needs more than one random
restart, and what a Bayesian posterior update actually computes instead
of just approximates. Building each one from scratch and then checking it
against scikit-learn is a way to confirm that understanding is actually
correct, not just plausible-sounding.

## Two real bugs found and fixed while building this

**A decision tree that refused to split on symmetric XOR-like data.**
The first version of `_best_split` only accepted a candidate split if it
strictly improved on the parent node's Gini impurity. On a perfectly
symmetric XOR pattern, every single-feature split at the root produces
exactly the same weighted Gini as the parent — a well-known, textbook
limitation of greedy top-down tree induction, since no single
axis-aligned cut can separate XOR at all. Rejecting every tied candidate
left the tree unable to split past the root at all, predicting the
majority class for every row. Comparing against scikit-learn's tree on
the same harder, asymmetric version of this pattern confirmed it was not
simply a bug in this implementation — scikit-learn's tree gets the exact
same two points wrong, a genuine shared limitation of greedy induction.
The actual bug was more specific: always take the single best split
among candidates, tied or not, and let depth and minimum-samples be the
real stopping criteria — a "wasted" first split still subdivides the
data so a genuinely separating split can be found one level down. The
minimal four-point XOR case is solved perfectly after this fix.

**K-means converging to a bad local optimum from a single random
initialization.** On a well-separated four-cluster dataset, one unlucky
random seed put two of the four initial centers inside the same true
cluster. The algorithm converged with two final centers splitting that
one cluster while a third center drifted to the midpoint between two
other true clusters it never started near — a textbook bad local
optimum for naively initialized k-means. Fixed the way real
implementations do: run several random restarts and keep the one with
the lowest final inertia, rather than trusting a single run. A direct
regression test confirms the multi-restart version reliably beats the
best of several single-restart runs.

## Running it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python3 demo.py
pytest -v
```

16 tests: correctness cases with known right answers (e.g., linear
regression recovering exact coefficients on noise-free data, Bayesian
posterior uncertainty shrinking as more data arrives), comparisons
against scikit-learn, and direct regression tests for both bugs above.
