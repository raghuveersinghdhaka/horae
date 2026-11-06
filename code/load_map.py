
import json
import os

import numpy as np

import fields
import hdata
import run_studies as RR

HERE = os.path.dirname(os.path.abspath(__file__))


def active(X, std, goal_spec):
    g = RR.mk_goal(goal_spec, X, std, X.shape[1])
    return hdata.active_fraction(X, g, tmax=len(X))*X.shape[1]


def main():
    out = {"scale": {}, "headline": {}, "raw": {}}
    bj = dict(ds="beijing", kind="alert")
    for reps in (8, 20, 42, 84, 170):
        vals = []
        for seed in range(1, 6):
            X, _, std = hdata.build_population("beijing_field.npz", reps, seed=seed, tmax=3000)
            vals.append(active(X, std, bj))
        out["scale"][str(reps*12)] = float(np.mean(vals))
    for cfg, c in RR.CONFIGS.items():
        if cfg.endswith("_w"):
            continue
        vals = []
        for seed in range(1, 6):
            X, _, std = hdata.build_population(f"{c['pop'][1]}_field.npz", c["pop"][2], seed=seed, tmax=c["pop"][3])
            vals.append(active(X, std, c["goal"]))
        out["headline"][cfg] = dict(active=float(np.mean(vals)), N=int(X.shape[1]), S=int(c["S"]))
    for name, ds, tmax in (("intel", "intel", 1500), ("beijing", "beijing", 6000)):
        X0, _ = hdata.load_base(f"{ds}_field.npz")
        X0 = X0[:tmax]
        out["raw"][name] = active(X0, float(np.std(X0)), dict(ds=ds, kind="alert"))
    with open(os.path.join(HERE, "results", "load_map.json"), "w", encoding="utf8") as f:
        json.dump(out, f, indent=1)
    print(out)


if __name__ == "__main__":
    main()
