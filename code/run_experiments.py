
import numpy as np, json, os, time
from horae_sim import simulate, Goal
import hdata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)

POLS = ["HORAE", "AGE", "VOI", "ATA", "LTP", "ALOHA", "ALOHA-ADP", "TDMA", "MAXWEIGHT"]
DECENTRAL = ["HORAE", "AGE", "VOI", "ATA", "LTP", "ALOHA", "ALOHA-ADP"]
SEEDS = [0, 1, 2, 3, 4]


def make_goal(ds, kind, X, std, N, importance=False, pct=92):
    w = hdata.importance_weights(N) if importance else None
    th = np.percentile(X, pct)
    if ds == "intel":
        if kind == "alert":
            return Goal("alert", tol=0.5, theta=th, scale=std, act_tol=0.3, w=w)
        return Goal("track", tol=0.2*std, scale=std, act_tol=0.45*std, w=w)
    else:  # beijing
        if kind == "alert":
            return Goal("alert", tol=8.0, theta=th, scale=std, act_tol=4.0, w=w)
        return Goal("track", tol=0.2*std, scale=std, act_tol=0.45*std, w=w)

TESTBEDS = {
    "intel":   dict(name="IntelLab-temp", reps=10, tmax=1500, unit="degC"),
    "beijing": dict(name="BeijingPRSA-PM25", reps=40, tmax=6000, unit="ug/m3"),
}

def build(ds):
    cfg = TESTBEDS[ds]
    X, xy, std = hdata.build_population(f"{ds}_field.npz", cfg["reps"], tmax=cfg["tmax"])
    return X, xy, std, X.shape[1]


def run_seeds(X, xy, goal, S, pols=POLS, seeds=SEEDS, **kw):
    
    acc = {p: [] for p in pols}
    for sd in seeds:
        for p in pols:
            acc[p].append(simulate(X, xy, p, goal, S=S, seed=sd, **kw))
    out = {}
    for p in pols:
        keys = [k for k in acc[p][0] if isinstance(acc[p][0][k], (int, float))]
        out[p] = {}
        for k in keys:
            a = np.array([r[k] for r in acc[p]], float)
            out[p][k] = [float(a.mean()), float(1.96*a.std(ddof=1)/np.sqrt(len(a)))]
    return out

def summarise(tag, res, key):
    h = res["HORAE"]["vwaoi"][0]
    o = res["MAXWEIGHT"]["vwaoi"][0]; al = res["ALOHA"]["vwaoi"][0]
    closes = 100*(1-(h-o)/max(al-o, 1e-9))
    print(f"  [{tag}] VWAoI HORAE={h:.1f} AGE={res['AGE']['vwaoi'][0]:.1f} "
          f"VOI={res['VOI']['vwaoi'][0]:.1f} ALOHA={al:.1f} ADP={res['ALOHA-ADP']['vwaoi'][0]:.1f} "
          f"TDMA={res['TDMA']['vwaoi'][0]:.1f} ORACLE={o:.1f} | vs ALOHA {al/h:.2f}x "
          f"vs AGE {res['AGE']['vwaoi'][0]/h:.2f}x closes {closes:.0f}% | "
          f"{key} HORAE={res['HORAE'][key][0]:.3f} ALOHA={res['ALOHA'][key][0]:.3f}")


def main():
    t0 = time.time()
    R = {"meta": {"seeds": SEEDS, "policies": POLS, "testbeds": TESTBEDS}}

    pops = {ds: build(ds) for ds in ("intel", "beijing")}

    
    print("E0 raw-network anchor (real, unmodified)")
    R["E0"] = {}
    for ds, nat in (("intel", "alert"), ("beijing", "alert")):
        X0, xy0 = hdata.load_base(f"{ds}_field.npz")
        if ds == "beijing":
            X0 = X0[:6000]
        std0 = float(np.std(X0)); N0 = X0.shape[1]
        g = make_goal(ds, nat, X0, std0, N0, importance=False)
        f = hdata.active_fraction(X0, g); S = max(1, int(round(max(1, f*N0))))  # load ~1
        res = run_seeds(X0, xy0, g, S)
        R["E0"][ds] = dict(N=N0, S=S, active_frac=f, results=res, metric="alert_f1")
        summarise(f"RAW {ds}/{nat} (N={N0})", res, "alert_f1")

    
    print("E1 main comparison")
    R["E1"] = {}
    CONFIGS = [("intel", "alert", 1.0), ("beijing", "alert", 1.0), ("beijing", "track", 2.0)]
    for ds, kind, load in CONFIGS:
        X, xy, std, N = pops[ds]
        R["E1"].setdefault(ds, {})
        g = make_goal(ds, kind, X, std, N, importance=False)
        f = hdata.active_fraction(X, g); S = max(4, int(round(f*N/load)))
        res = run_seeds(X, xy, g, S)
        key = "alert_f1" if kind == "alert" else "track_nrmse"
        R["E1"][ds][kind] = dict(S=S, N=N, active_frac=f, load=f*N/S, results=res, metric=key)
        summarise(f"{ds}/{kind}", res, key)

    
    print("E2 load sweep")
    R["E2"] = {}
    for ds in ("intel", "beijing"):
        X, xy, std, N = pops[ds]
        g = make_goal(ds, "alert", X, std, N, importance=False)
        f = hdata.active_fraction(X, g); ac = f*N
        R["E2"][ds] = {"active_cnt": ac, "N": N, "loads": [], "S": [], "by_pol": {p: [] for p in POLS}}
        for load in [4, 3, 2, 1.5, 1, 0.75, 0.5]:
            S = max(3, int(round(ac/load)))
            res = run_seeds(X, xy, g, S, seeds=[0, 1, 2])
            R["E2"][ds]["loads"].append(load); R["E2"][ds]["S"].append(S)
            for p in POLS:
                R["E2"][ds]["by_pol"][p].append(res[p]["vwaoi"][0])
        print(f"  {ds}: loads {R['E2'][ds]['loads']}")

    
    print("E4 ablation + importance")
    ds = "beijing"; X, xy, std, N = pops[ds]
    R["E4"] = {"ds": ds, "N": N, "spread": [], "HORAE_over_AGE": [], "HORAE_over_VOI": [],
               "HORAE_over_ALOHA": [], "abl": {}}
    
    g0 = make_goal(ds, "track", X, std, N, importance=False)
    f = hdata.active_fraction(X, g0); S = max(4, int(round(f*N/2)))
    res0 = run_seeds(X, xy, g0, S, seeds=[0, 1, 2])
    R["E4"]["abl"] = {p: res0[p]["vwaoi"][0] for p in POLS}; R["E4"]["S"] = S
    for wc in [1.0, 2.0, 4.0, 8.0, 16.0]:
        w = hdata.importance_weights(N, w_critical=wc)
        g = Goal("track", tol=0.2*std, scale=std, act_tol=0.45*std, w=w)
        res = run_seeds(X, xy, g, S, pols=["HORAE", "AGE", "VOI", "ALOHA"], seeds=[0, 1, 2])
        h = res["HORAE"]["vwaoi"][0]
        R["E4"]["spread"].append(wc)
        R["E4"]["HORAE_over_AGE"].append(res["AGE"]["vwaoi"][0]/h)
        R["E4"]["HORAE_over_VOI"].append(res["VOI"]["vwaoi"][0]/h)
        R["E4"]["HORAE_over_ALOHA"].append(res["ALOHA"]["vwaoi"][0]/h)
    print(f"  importance spread {R['E4']['spread']} -> HORAE/AGE {[round(x,2) for x in R['E4']['HORAE_over_AGE']]}")

    
    print("E5 scale")
    
    R["E5"] = {"N": [], "S": [], "load": 2.0, "HORAE_vwaoi": [], "ALOHA_vwaoi": [],
               "HORAE_coll": [], "G_over_S": []}
    Xb, xyb, stdb = hdata.build_population("beijing_field.npz", 170, tmax=3000)  # up to ~2000 nodes
    for reps in [8, 20, 42, 84, 170]:
        Nn = 12*reps
        Xs = Xb[:, :Nn]; xys = xyb[:Nn]
        g = make_goal("beijing", "alert", Xs, stdb, Nn, importance=False)
        f = hdata.active_fraction(Xs, g); S = max(4, int(round(f*Nn/2)))
        rh = run_seeds(Xs, xys, g, S, pols=["HORAE", "ALOHA"], seeds=[0, 1])
        R["E5"]["N"].append(Nn); R["E5"]["S"].append(S)
        R["E5"]["HORAE_vwaoi"].append(rh["HORAE"]["vwaoi"][0])
        R["E5"]["ALOHA_vwaoi"].append(rh["ALOHA"]["vwaoi"][0])
        R["E5"]["HORAE_coll"].append(rh["HORAE"]["coll_rate"][0])
    print(f"  N {R['E5']['N']} S {R['E5']['S']} HORAE VWAoI {[round(x,1) for x in R['E5']['HORAE_vwaoi']]} "
          f"coll {[round(x,2) for x in R['E5']['HORAE_coll']]}")

    
    print("E12 scale at fixed channel (S pinned)")
    S_fixed = R["E5"]["S"][0]
    R["E12"] = {"N": [], "S": S_fixed, "HORAE_vwaoi": [], "ALOHA_vwaoi": [],
                "HORAE_coll": [], "HORAE_age_active": [], "HORAE_over_ALOHA": []}
    for reps in [8, 20, 42, 84, 170]:
        Nn = 12*reps
        Xs = Xb[:, :Nn]; xys = xyb[:Nn]
        g = make_goal("beijing", "alert", Xs, stdb, Nn, importance=False)
        rh = run_seeds(Xs, xys, g, S_fixed, pols=["HORAE", "ALOHA"], seeds=[0, 1])
        R["E12"]["N"].append(Nn)
        R["E12"]["HORAE_vwaoi"].append(rh["HORAE"]["vwaoi"][0])
        R["E12"]["ALOHA_vwaoi"].append(rh["ALOHA"]["vwaoi"][0])
        R["E12"]["HORAE_coll"].append(rh["HORAE"]["coll_rate"][0])
        R["E12"]["HORAE_age_active"].append(rh["HORAE"]["age_active"][0])
        R["E12"]["HORAE_over_ALOHA"].append(rh["ALOHA"]["vwaoi"][0]/rh["HORAE"]["vwaoi"][0])
    print(f"  S fixed at {S_fixed}: N {R['E12']['N']} HORAE VWAoI "
          f"{[round(x,1) for x in R['E12']['HORAE_vwaoi']]} coll "
          f"{[round(x,2) for x in R['E12']['HORAE_coll']]}")

    
    print("E6 robustness")
    R["E6"] = {}
    
    ds = "beijing"; X0, xy0 = hdata.load_base("beijing_field.npz"); X0 = X0[:4000]; std0 = float(np.std(X0))
    T0, Nb = X0.shape; reps = 40; rng = np.random.default_rng(7)
    syncs = [0.0, 0.25, 0.5, 0.75, 1.0]; R["E6"]["sync"] = {"level": syncs, "HORAE_over_ALOHA": [], "HORAE_vwaoi": [], "ALOHA_vwaoi": []}
    for s in syncs:
        Xs = [X0]; xys = [xy0]
        for r in range(1, reps):
            sh = 0 if rng.random() < s else int(rng.integers(1, T0))  
            Xs.append(np.roll(X0, sh, axis=0)+rng.normal(0, 0.03*std0, X0.shape))
            xys.append(xy0+np.array([(r % 6)*40.0, (r//6)*40.0])+rng.normal(0, 2, xy0.shape))
        Xp = np.hstack(Xs); xyp = np.vstack(xys); Np = Xp.shape[1]
        g = make_goal("beijing", "alert", Xp, std0, Np, importance=False)
        f = hdata.active_fraction(Xp, g); S = max(4, int(round(f*Np/2)))
        res = run_seeds(Xp, xyp, g, S, pols=["HORAE", "ALOHA"], seeds=[0, 1])
        h = res["HORAE"]["vwaoi"][0]; a = res["ALOHA"]["vwaoi"][0]
        R["E6"]["sync"]["HORAE_over_ALOHA"].append(a/h)
        R["E6"]["sync"]["HORAE_vwaoi"].append(h); R["E6"]["sync"]["ALOHA_vwaoi"].append(a)
    print(f"  synchrony {syncs} -> HORAE/ALOHA {[round(x,2) for x in R['E6']['sync']['HORAE_over_ALOHA']]}")
    
    ds = "intel"; X, xy, std, N = pops[ds]
    g = make_goal("intel", "alert", X, std, N, importance=False)
    f = hdata.active_fraction(X, g); S = max(4, int(round(f*N/2)))
    R["E6"]["ploss"] = {"p": [], "HORAE_vwaoi": [], "HORAE_f1": []}
    for pl in [0.0, 0.05, 0.1, 0.2, 0.3]:
        res = run_seeds(X, xy, g, S, pols=["HORAE"], seeds=[0, 1, 2], p_loss=pl)
        R["E6"]["ploss"]["p"].append(pl)
        R["E6"]["ploss"]["HORAE_vwaoi"].append(res["HORAE"]["vwaoi"][0])
        R["E6"]["ploss"]["HORAE_f1"].append(res["HORAE"]["alert_f1"][0])
    print(f"  ACK loss {R['E6']['ploss']['p']} -> VWAoI {[round(x,1) for x in R['E6']['ploss']['HORAE_vwaoi']]}")

    
    print("E7 answer-per-joule")
    R["E7"] = {}
    for ds in ("intel", "beijing"):
        e1 = R["E1"][ds]["alert"]["results"]
        R["E7"][ds] = {p: dict(energy=e1[p]["energy_J"][0], f1=e1[p]["alert_f1"][0],
                               epv=e1[p]["energy_per_valbit"][0], vwaoi=e1[p]["vwaoi"][0]) for p in POLS}

    
    print("E8 controller trace")
    ds = "beijing"; X, xy, std, N = pops[ds]
    g = make_goal("beijing", "alert", X, std, N, importance=False)
    f = hdata.active_fraction(X, g); S = max(4, int(round(f*N/2)))
    tr = simulate(X, xy, "HORAE", g, S=S, seed=0, trace=True)
    R["E8"] = dict(S=S, G=tr["trace_G"], coll=tr["trace_coll"], succ=tr["trace_succ"])
    print(f"  S={S} mean G/S last-half={np.mean(np.array(tr['trace_G'][len(tr['trace_G'])//2:]))/S:.2f}")

    
    print("E9 capture/fading robustness")
    ds = "beijing"; X, xy, std, N = pops[ds]
    g = make_goal("beijing", "alert", X, std, N, importance=False)
    f = hdata.active_fraction(X, g); S = max(4, int(round(f*N/2)))
    R["E9"] = {"capture_db": [], "by_pol": {p: {"vwaoi": [], "f1": []} for p in ["HORAE", "AGE", "ATA", "ALOHA"]}}
    for cdb in [None, 9.0, 6.0, 3.0, 0.0]:
        R["E9"]["capture_db"].append(-1 if cdb is None else cdb)
        for p in ["HORAE", "AGE", "ATA", "ALOHA"]:
            res = run_seeds(X, xy, g, S, pols=[p], seeds=[0, 1, 2],
                            capture_db=cdb, fading=(cdb is not None))
            R["E9"]["by_pol"][p]["vwaoi"].append(res[p]["vwaoi"][0])
            R["E9"]["by_pol"][p]["f1"].append(res[p]["alert_f1"][0])
    print(f"  capture_db {R['E9']['capture_db']} -> HORAE VWAoI {[round(x,1) for x in R['E9']['by_pol']['HORAE']['vwaoi']]}")

    
    print("E10 real-segment population (no synthesis)")
    R["E10"] = {}
    Xr, xyr, stdr = hdata.build_real_segment_population("beijing_field.npz", L=1200)
    Nr = Xr.shape[1]; R["E10"]["N"] = Nr
    for kind, load, imp in [("alert", 2.0, False), ("track", 2.0, False), ("alert", 2.0, True)]:
        w = hdata.importance_weights(Nr) if imp else None
        th = np.percentile(Xr, 92)
        if kind == "alert":
            g = Goal("alert", tol=8.0, theta=th, scale=stdr, act_tol=4.0, w=w)
        else:
            g = Goal("track", tol=0.2*stdr, scale=stdr, act_tol=0.45*stdr, w=w)
        fr = hdata.active_fraction(Xr, g); Sr = max(4, int(round(fr*Nr/load)))
        res = run_seeds(Xr, xyr, g, Sr, seeds=[0, 1, 2])
        tag = f"{kind}{'+imp' if imp else ''}"
        key = "alert_f1" if kind == "alert" else "track_nrmse"
        R["E10"][tag] = dict(S=Sr, results=res, metric=key)
        summarise(f"REAL-SEG {tag} (N={Nr})", res, key)

    
    print("E11 theory validation")
    ds = "beijing"; X, xy, std, N = pops[ds]
    g = make_goal("beijing", "alert", X, std, N, importance=False)
    f = hdata.active_fraction(X, g)
    R["E11"] = {"e": float(np.e), "load": [], "S": [], "succ_per_slot": [], "S_over_e": [], "meanGoverS": []}
    for load in [1.0, 2.0, 4.0]:
        S = max(4, int(round(f*N/load)))
        tr = simulate(X, xy, "HORAE", g, S=S, seed=0, trace=True)
        succ = np.array(tr["trace_succ"], float); Gtr = np.array(tr["trace_G"], float)
        R["E11"]["load"].append(load); R["E11"]["S"].append(S)
        R["E11"]["succ_per_slot"].append(float(succ.mean()))
        R["E11"]["S_over_e"].append(S/np.e)
        R["E11"]["meanGoverS"].append(float(Gtr.mean()/S))
    
    rng = np.random.default_rng(0); R["E11"]["sat_age"] = []
    for S in [10, 30, 60]:
        ages = []; a = 1
        for _ in range(120000):
            others = rng.integers(0, S, size=S-1)   # G-1 other attempters at G=S
            if not np.any(others == 0):
                ages.append(a); a = 1
            else:
                a += 1
        R["E11"]["sat_age"].append([S, float(np.mean(ages))])
    print(f"  succ/slot vs S/e: {[round(a/b,2) for a,b in zip(R['E11']['succ_per_slot'],R['E11']['S_over_e'])]} "
          f"| sat age {[round(x[1],2) for x in R['E11']['sat_age']]} vs e={np.e:.2f}")

    
    print("E13 distributional cost under congestion")
    ds = "beijing"; X, xy, std, N = pops[ds]
    g = make_goal("beijing", "alert", X, std, N, importance=False)
    f = hdata.active_fraction(X, g); ac = f*N
    R["E13"] = {"N": N, "loads": [], "S": [], "by_pol": {}}
    KEYS = ["vwaoi", "age_active", "age_active_lowval", "age_active_highval",
            "age_active_p95", "jain_deliv", "alert_f1", "succ"]
    for p in POLS:
        R["E13"]["by_pol"][p] = {k: [] for k in KEYS}
    for load in [2.0, 4.0]:
        S = max(3, int(round(ac/load)))
        res = run_seeds(X, xy, g, S, seeds=[0, 1, 2])
        R["E13"]["loads"].append(load); R["E13"]["S"].append(S)
        for p in POLS:
            for k in KEYS:
                R["E13"]["by_pol"][p][k].append(res[p][k][0])
        h = res["HORAE"]; a = res["ALOHA"]
        print(f"  load {load}x S={S}: HORAE low-value age {h['age_active_lowval'][0]:.1f} "
              f"(ALOHA {a['age_active_lowval'][0]:.1f}), high-value {h['age_active_highval'][0]:.1f} "
              f"(ALOHA {a['age_active_highval'][0]:.1f}), Jain {h['jain_deliv'][0]:.3f} "
              f"(ALOHA {a['jain_deliv'][0]:.3f})")

    R["meta"]["runtime_s"] = time.time()-t0
    with open(os.path.join(OUT, "results.json"), "w") as fjson:
        json.dump(R, fjson, indent=1)
    print(f"\nDONE in {R['meta']['runtime_s']:.0f}s -> results/results.json")

if __name__ == "__main__":
    main()
