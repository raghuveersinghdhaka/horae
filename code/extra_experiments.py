
import json, os, time
import numpy as np
import hdata
from horae_sim import Goal
from run_experiments import make_goal, build, run_seeds, POLS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "results.json")


def main():
    t0 = time.time()
    R = json.load(open(RES, encoding="utf-8"))

    
    print("E12b fixed-channel scaling at several pinned S")
    Xb, xyb, stdb = hdata.build_population("beijing_field.npz", 170, tmax=3000)
    R["E12b"] = {"S_pinned": [], "by_S": {}}
    for S_fix in [4, 11, 22]:
        rec = {"N": [], "load": [], "HORAE_vwaoi": [], "ALOHA_vwaoi": [], "ADP_vwaoi": [],
               "HORAE_over_ALOHA": [], "HORAE_over_ADP": [], "HORAE_coll": [],
               "HORAE_succ": [], "ALOHA_succ": [], "HORAE_age_active": []}
        for reps in [8, 20, 42, 84, 170]:
            Nn = 12 * reps
            Xs = Xb[:, :Nn]; xys = xyb[:Nn]
            g = make_goal("beijing", "alert", Xs, stdb, Nn, importance=False)
            f = hdata.active_fraction(Xs, g)
            r = run_seeds(Xs, xys, g, S_fix, pols=["HORAE", "ALOHA", "ALOHA-ADP"], seeds=[0, 1])
            rec["N"].append(Nn)
            rec["load"].append(float(f * Nn / S_fix))
            rec["HORAE_vwaoi"].append(r["HORAE"]["vwaoi"][0])
            rec["ALOHA_vwaoi"].append(r["ALOHA"]["vwaoi"][0])
            rec["ADP_vwaoi"].append(r["ALOHA-ADP"]["vwaoi"][0])
            rec["HORAE_over_ALOHA"].append(r["ALOHA"]["vwaoi"][0] / r["HORAE"]["vwaoi"][0])
            rec["HORAE_over_ADP"].append(r["ALOHA-ADP"]["vwaoi"][0] / r["HORAE"]["vwaoi"][0])
            rec["HORAE_coll"].append(r["HORAE"]["coll_rate"][0])
            rec["HORAE_succ"].append(r["HORAE"]["succ"][0])
            rec["ALOHA_succ"].append(r["ALOHA"]["succ"][0])
            rec["HORAE_age_active"].append(r["HORAE"]["age_active"][0])
        R["E12b"]["S_pinned"].append(S_fix)
        R["E12b"]["by_S"][str(S_fix)] = rec
        print(f"  S={S_fix:2d} load {[round(x,1) for x in rec['load']]} "
              f"HORAE/ALOHA {[round(x,2) for x in rec['HORAE_over_ALOHA']]} "
              f"HORAE/ADP {[round(x,2) for x in rec['HORAE_over_ADP']]}")

    
    print("E14 routine vs critical sources under congestion")
    X, xy, std, N = build("beijing")
    w = hdata.importance_weights(N)
    th = float(np.percentile(X, 92))
    g = Goal("alert", tol=8.0, theta=th, scale=std, act_tol=4.0, w=w)
    f = hdata.active_fraction(X, g); ac = f * N
    KEYS = ["vwaoi", "age_active_routine", "age_active_critical", "age_active",
            "age_active_p95", "jain_deliv", "alert_f1", "succ"]
    R["E14"] = {"N": N, "w_critical": 8.0, "frac_critical": 0.15,
                "loads": [], "S": [], "by_pol": {p: {k: [] for k in KEYS} for p in POLS}}
    for load in [2.0, 4.0]:
        S = max(3, int(round(ac / load)))
        res = run_seeds(X, xy, g, S, seeds=[0, 1, 2])
        R["E14"]["loads"].append(load); R["E14"]["S"].append(S)
        for p in POLS:
            for k in KEYS:
                R["E14"]["by_pol"][p][k].append(res[p][k][0])
        h = res["HORAE"]; a = res["ALOHA"]; d = res["ALOHA-ADP"]
        print(f"  load {load}x S={S}: HORAE routine {h['age_active_routine'][0]:.1f} "
              f"critical {h['age_active_critical'][0]:.1f} | ALOHA routine "
              f"{a['age_active_routine'][0]:.1f} critical {a['age_active_critical'][0]:.1f} | "
              f"ADP routine {d['age_active_routine'][0]:.1f} critical {d['age_active_critical'][0]:.1f}")

    R["meta"]["extra_runtime_s"] = time.time() - t0
    with open(RES, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
    print(f"\nDONE in {time.time()-t0:.0f}s -> results/results.json (E12b, E14 merged)")


if __name__ == "__main__":
    main()
