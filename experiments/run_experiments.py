"""
Runs all the experiments and saves CSVs to results/.

    python experiments/run_experiments.py --exp all --jobs 4

e0: PUF quality (uniformity, uniqueness, reliability)
e1: 64-bit arbiter PUF, accuracy vs number of CRPs
e2: CRPs needed vs challenge length
e3: XOR PUFs with k = 2..5
e4: effect of noise
e5: does LR recover the delay weights?
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from attacks import (ProductLR, linear_svm, logistic_regression, mlp,
                     perceptron, rbf_svm, timed_fit_eval)
from puf import XORArbiterPUF, make_crp_dataset, parity_features, random_challenges, to_bits

RESULTS = os.path.join(ROOT, "results")
N_TEST = 20000


def build_attack(model: str, k: int, N: int, seed: int):
    if model == "Perceptron":
        return perceptron(seed)
    if model == "LR":
        return logistic_regression()
    if model == "LinSVM":
        return linear_svm()
    if model == "RBFSVM":
        return rbf_svm()
    if model == "MLP":
        return mlp(k, N, seed)
    if model == "PLR":
        return ProductLR(k, seed=seed, restarts=6 if k <= 3 else 4,
                         max_epochs=300 if k <= 3 else 150)
    raise ValueError(model)


def run_job(exp, n, k, seed, N, model, features="phi", noisiness=0.0):
    puf = XORArbiterPUF(n, k, noisiness=noisiness, seed=seed)
    c_tr, phi_tr, y_tr = make_crp_dataset(puf, N, seed=1000 + seed, noisy=True)
    c_te, phi_te, y_te_clean = make_crp_dataset(puf, N_TEST, seed=5000 + seed, noisy=False)
    if features == "raw":
        Xtr, Xte = (1 - 2 * c_tr).astype(np.float32), (1 - 2 * c_te).astype(np.float32)
    else:
        Xtr, Xte = phi_tr, phi_te
    attack = build_attack(model, k, N, seed)
    acc, t = timed_fit_eval(attack, Xtr, y_tr, Xte, y_te_clean)
    row = dict(exp=exp, n=n, k=k, seed=seed, N=N, model=model, features=features,
               noisiness=noisiness, acc=acc, train_time_s=t)
    if noisiness > 0:
        y_te_noisy = to_bits(puf.eval_phi(phi_te, noisy=True))
        row["acc_vs_noisy"] = float(np.mean(attack.predict(Xte) == y_te_noisy))
        row["reliability"] = puf.reliability()
    return row


JOB_KEYS = ("exp", "n", "k", "seed", "N", "model", "features", "noisiness")


def _key(d):
    d = {"features": "phi", "noisiness": 0.0, **d}
    return tuple(str(d[k]) for k in JOB_KEYS)


def run_parallel(jobs, n_jobs, tag):
    # finished jobs go into a .jsonl file so a stopped run can pick up where it left off
    t0 = time.time()
    log = os.path.join(RESULTS, f"{tag}.jsonl")
    done = {}
    if os.path.exists(log):
        for line in open(log):
            r = json.loads(line)
            done[_key(r)] = r
    todo = [j for j in jobs if _key(j) not in done]
    print(f"[{tag}] {len(jobs)} jobs, {len(jobs) - len(todo)} already done, running {len(todo)}", flush=True)
    with open(log, "a") as fh:
        for i, row in enumerate(Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
                delayed(run_job)(**j) for j in todo)):
            fh.write(json.dumps(row) + "\n"); fh.flush()
            done[_key(row)] = row
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                print(f"[{tag}] {i + 1}/{len(todo)} done, {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame([done[_key(j)] for j in jobs])
    path = os.path.join(RESULTS, f"{tag}.csv")
    df.to_csv(path, index=False)
    print(f"[{tag}] {len(jobs)} jobs in {time.time() - t0:.0f} s -> {path}", flush=True)
    return df


def e0_quality():
    rows = []
    n, n_chips, n_ch = 64, 40, 20000
    rng = np.random.default_rng(7)
    phi = parity_features(random_challenges(n_ch, n, rng))
    for k in [1, 2, 3, 4, 5]:
        chips = [XORArbiterPUF(n, k, seed=s) for s in range(n_chips)]
        R = np.array([(p.eval_phi(phi, noisy=False) < 0) for p in chips], dtype=np.int8)
        uniformity = R.mean()
        hd = [np.mean(R[i] != R[j]) for i, j in itertools.combinations(range(n_chips), 2)]
        for nz in [0.0, 0.02, 0.05, 0.1, 0.2]:
            rel = np.mean([XORArbiterPUF(n, k, noisiness=nz, seed=s).reliability() for s in range(10)])
            rows.append(dict(k=k, noisiness=nz, uniformity=uniformity,
                             uniqueness=float(np.mean(hd)), uniqueness_std=float(np.std(hd)),
                             reliability=rel))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "e0_quality.csv"), index=False)
    print(df.to_string())


def e1_jobs():
    Ns = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    jobs = []
    for seed, N in itertools.product(range(10), Ns):
        for model in ["Perceptron", "LR", "LinSVM"]:
            jobs.append(dict(exp="e1", n=64, k=1, seed=seed, N=N, model=model))
        jobs.append(dict(exp="e1", n=64, k=1, seed=seed, N=N, model="LR", features="raw"))
        if N <= 10000:
            jobs.append(dict(exp="e1", n=64, k=1, seed=seed, N=N, model="RBFSVM", features="raw"))
            jobs.append(dict(exp="e1", n=64, k=1, seed=seed, N=N, model="RBFSVM"))
        if N >= 200:
            jobs.append(dict(exp="e1", n=64, k=1, seed=seed, N=N, model="MLP"))
    return jobs


def e2_jobs():
    Ns = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000]
    return [dict(exp="e2", n=n, k=1, seed=s, N=N, model="LR")
            for n in [16, 32, 64, 128, 256] for s in range(3) for N in Ns]


# CRP budgets for each k
E3_GRID = {2: [1000, 2000, 5000, 10000, 20000, 50000],
           3: [5000, 10000, 20000, 50000, 100000],
           4: [10000, 20000, 50000, 100000, 200000, 400000],
           5: [50000, 100000, 200000, 400000]}
E3_SEEDS = {2: 3, 3: 3, 4: 3, 5: 2}


def e3_jobs():
    jobs = []
    for k, Ns in E3_GRID.items():
        for seed, N in itertools.product(range(E3_SEEDS[k]), Ns):
            jobs.append(dict(exp="e3", n=64, k=k, seed=seed, N=N, model="MLP"))
            jobs.append(dict(exp="e3", n=64, k=k, seed=seed, N=N, model="PLR"))
            if N <= 100000:
                jobs.append(dict(exp="e3", n=64, k=k, seed=seed, N=N, model="LR"))
            if k <= 3 and N <= 20000:
                jobs.append(dict(exp="e3", n=64, k=k, seed=seed, N=N, model="RBFSVM"))
    # big jobs first
    jobs.sort(key=lambda j: -j["N"] * (2 ** j["k"]) * (1 if j["model"] in ("MLP", "PLR") else 0.01))
    return jobs


def e4_jobs():
    jobs = []
    for nz in [0.0, 0.02, 0.05, 0.1, 0.2]:
        for seed in range(3):
            for N in [500, 2000, 10000, 50000]:
                jobs.append(dict(exp="e4", n=64, k=1, seed=seed, N=N, model="LR", noisiness=nz))
            jobs.append(dict(exp="e4", n=64, k=2, seed=seed, N=20000, model="MLP", noisiness=nz))
    return jobs


def e5_weights():
    rows = []
    for N in [100, 1000, 10000]:
        puf = XORArbiterPUF(64, 1, seed=0)
        _, X, y = make_crp_dataset(puf, N, seed=1000)
        lr = logistic_regression().fit(X, y).model
        w_hat = -lr.coef_.ravel()
        w_hat[-1] = -lr.intercept_[0] - lr.coef_.ravel()[-1]
        w_true = puf.weights[0]
        corr = float(np.corrcoef(w_true, w_hat)[0, 1])
        for i, (wt, wh) in enumerate(zip(w_true, w_hat)):
            rows.append(dict(N=N, stage=i, w_true=wt, w_hat=wh, corr=corr))
        print(f"N={N}: corr(w_true, w_hat) = {corr:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "e5_weights.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", help="e0,e1,...,e5 or all (comma-separated)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    todo = ["e0", "e1", "e2", "e3", "e4", "e5"] if args.exp == "all" else args.exp.split(",")
    for e in todo:
        print(f"===== {e} =====", flush=True)
        if e == "e0":
            e0_quality()
        elif e == "e5":
            e5_weights()
        else:
            jobs = {"e1": e1_jobs, "e2": e2_jobs, "e3": e3_jobs, "e4": e4_jobs}[e]()
            run_parallel(jobs, args.jobs, {"e1": "e1_arbiter", "e2": "e2_length",
                                           "e3": "e3_xor", "e4": "e4_noise"}[e])
