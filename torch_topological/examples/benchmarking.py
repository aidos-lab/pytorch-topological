"""Debug script for benchmarking some calculations."""

import time
import torch
import sys

from torch_topological.nn import WassersteinDistance
from torch_topological.nn import VietorisRipsComplex


def run_test(X, vr, name, dist=False):
    W1 = WassersteinDistance()

    pre = time.perf_counter()

    X_pi = vr(X, treat_as_distances=dist)
    Y_pi = vr(X, treat_as_distances=dist)

    dists = torch.stack([W1(x_pi, y_pi) for x_pi, y_pi in zip(X_pi, Y_pi)])
    dist = dists.mean()

    cur = time.perf_counter()
    print(f"{name}: {cur - pre:.4f}s")


if __name__ == "__main__":
    X = torch.load(sys.argv[1])

    print("Calculating everything ourselves")

    run_test(X, VietorisRipsComplex(dim=0), "raw")
    run_test(X, VietorisRipsComplex(dim=0, threshold=1.0), "thresholded")

    print("\nPre-defined distances")

    D = torch.cdist(X, X)

    run_test(D, VietorisRipsComplex(dim=0), "raw", dist=True)
    run_test(
        D, VietorisRipsComplex(dim=0, threshold=1.0), "thresholded", dist=True
    )
