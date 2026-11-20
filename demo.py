"""
Clone a simulated PUF with ML.

    python demo.py                      # 64-bit arbiter PUF, LR, 2000 CRPs
    python demo.py --k 3 --crps 50000   # 3-XOR PUF, MLP
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from attacks import ProductLR, logistic_regression, mlp, perceptron
from puf import XORArbiterPUF, make_crp_dataset, parity_features, random_challenges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--crps", type=int, default=2000)
    ap.add_argument("--attack", choices=["auto", "perceptron", "lr", "mlp", "plr"], default="auto")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    attack = a.attack if a.attack != "auto" else ("lr" if a.k == 1 else "mlp")
    chip = XORArbiterPUF(a.n, a.k, noisiness=a.noise, seed=a.seed)
    name = "Arbiter PUF" if a.k == 1 else f"{a.k}-XOR Arbiter PUF"
    print(f"\nChip: {a.n}-bit {name}, reliability {chip.reliability() * 100:.1f}%")

    _, X, y = make_crp_dataset(chip, a.crps, seed=a.seed + 1, noisy=True)
    print(f"Collected {a.crps:,} CRPs")

    model = {"perceptron": lambda: perceptron(), "lr": lambda: logistic_regression(),
             "mlp": lambda: mlp(a.k, a.crps), "plr": lambda: ProductLR(a.k)}[attack]()
    t0 = time.perf_counter()
    model.fit(X, y)
    print(f"Trained {getattr(model, 'name', attack)} in {time.perf_counter() - t0:.1f}s")

    rng = np.random.default_rng(a.seed + 99)
    c = random_challenges(20000, a.n, rng)
    phi = parity_features(c)
    truth = (chip.eval_phi(phi, noisy=False) < 0).astype(int)
    pred = model.predict(phi)
    acc = np.mean(pred == truth)
    print(f"Accuracy on 20,000 new challenges: {acc * 100:.2f}%\n")

    print("challenge (first 24 bits)    chip  clone")
    for i in range(8):
        bits = "".join(map(str, c[i, :24]))
        mark = "" if pred[i] == truth[i] else "  x"
        print(f"{bits}...  {truth[i]}     {pred[i]}{mark}")
    print("\ncloned!" if acc >= 0.95 else "\nnot cloned yet, try more --crps")


if __name__ == "__main__":
    main()
