
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
import hdata, fields
from horae_sim import Goal, StaticGoal, simulate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
WORKERS = 7

ALIAS = {"HORAE-G": ("HORAE", dict(beta_ceil=1.0, freeze_idle=True, fast_down=3.0)),
         "ADP-G": ("ALOHA-ADP", dict(freeze_idle=True, fast_down=3.0)),
         "AGE-G": ("AGE", dict(beta_ceil=1.0, freeze_idle=True, fast_down=3.0)),
         
         "HORAE-R": ("HORAE", dict(freeze_idle=True, fast_down=3.0, hz_thresh=2.0, ceil_rel=0.25, hz_exp=1.0, hz_exempt=True)),
         "AGE-R": ("AGE", dict(freeze_idle=True, fast_down=3.0, hz_thresh=0.5, ceil_rel=1.0)),
         "AGE-RE": ("AGE", dict(freeze_idle=True, fast_down=3.0, hz_thresh=0.5, ceil_rel=1.0, hz_exempt=True))}
KEEP = ["vwaoi", "vwaoi_active", "alert_f1", "track_nrmse", "energy_J", "tx", "succ", "coll", "coll_rate", "G_over_S",
        "attempts_per_slot", "active_per_slot", "jain_deliv", "age_active", "age_active_max", "age_active_p95",
        "age_active_routine", "age_active_critical", "age_active_lowval", "age_active_highval", "mean_age",
        "tx_J", "rx_J", "listen_J", "comp_J", "n_listen", "beta_at_floor", "beta_at_ceil", "beta_log_sd",
        "val_delivered", "energy_per_succ", "energy_per_valbit", "energy_per_dev_slot_uJ", "slots", "N", "S", "T",
        "n_ever_active", "deliv_min", "idle_per_slot", "frac_unheard", "succ_after_warmup", "frac_wrong_end"]

CONFIGS = {  
    "intel_alert": dict(pop=("rep", "intel", 10, 1500), goal=dict(ds="intel", kind="alert"), S=62),
    "beijing_alert": dict(pop=("rep", "beijing", 40, 6000), goal=dict(ds="beijing", kind="alert"), S=21),
    "beijing_track": dict(pop=("rep", "beijing", 40, 6000), goal=dict(ds="beijing", kind="track"), S=32),
    "beijing_alert_w": dict(pop=("rep", "beijing", 40, 6000), goal=dict(ds="beijing", kind="alert", wc=8.0), S=21),
    "beijing_track_w": dict(pop=("rep", "beijing", 40, 6000), goal=dict(ds="beijing", kind="track", wc=8.0), S=32),
}
HEAD = ["intel_alert", "beijing_alert", "beijing_track"]



_cache = {}


def get_pop(spec, popseed):
    key = (tuple(spec), popseed)
    if key in _cache:
        return _cache[key]
    t = spec[0]
    if t == "rep":
        out = hdata.build_population(f"{spec[1]}_field.npz", spec[2], seed=popseed, tmax=spec[3])
    elif t == "raw":
        X0, xy0 = hdata.load_base(f"{spec[1]}_field.npz")
        if spec[2]:
            X0 = X0[:spec[2]]
        out = (X0, xy0, float(np.std(X0)))
    elif t == "seg":
        out = hdata.build_real_segment_population(f"{spec[1]}_field.npz", L=spec[2])
    elif t == "sync":
        out = fields.sync_population(spec[1], popseed=6 + popseed)
    elif t == "corr":
        out = fields.corr_field(spec[1], spec[2], spec[3], popseed=popseed)
    elif t == "burst":
        out = fields.burst_field(spec[1], popseed=popseed)
    elif t == "field":
        out = fields.load_field(spec[1], tmax=spec[2])
    elif t == "shifted":
        out = fields.shifted_subset(spec[1], n_base=spec[2], popseed=popseed, tmax=spec[3])
    elif t == "part":           
        out = hdata.build_population(f"{spec[1]}_field.npz", spec[2], seed=popseed, tmax=spec[3], tstart=spec[3]//2)
    elif t == "static":
        rng = np.random.default_rng(5)
        N = spec[1]; c = np.exp(spec[2]*rng.normal(size=N)); c = c/np.mean(np.sqrt(c))**2
        out = (np.broadcast_to(np.zeros((1, N)), (spec[3], N)), np.zeros((N, 2)), 1.0, c)
    else:
        raise ValueError(spec)
    _cache.clear()
    _cache[key] = out
    return out


def mk_goal(gs, X, std, N):
    kind = gs["kind"]; ds = gs.get("ds", "field")
    if kind == "static":
        return None
    pct = gs.get("pct", 92)
    w = None
    if gs.get("wc") is not None:
        w = hdata.importance_weights(N, frac_critical=gs.get("wf", 0.15), w_critical=gs["wc"])
    th = gs.get("theta")
    if th is None:
        th = float(np.percentile(gs["pct_on"](X) if "pct_on" in gs else X, pct))
    rmul = gs.get("rho_mul", 1.0); dmul = gs.get("delta_mul", 1.0)
    extra = {k: gs[k] for k in ("kappa", "boost", "v_floor") if k in gs}
    if kind == "alert":
        if ds == "intel":
            return Goal("alert", tol=0.5*rmul, theta=th, scale=std, act_tol=0.3*dmul, w=w, **extra)
        if ds == "beijing":
            return Goal("alert", tol=8.0*rmul, theta=th, scale=std, act_tol=4.0*dmul, w=w, **extra)
        return Goal("alert", tol=0.1*std*rmul, theta=th, scale=std, act_tol=0.05*std*dmul, w=w, **extra)
    return Goal("track", tol=0.2*std, scale=std, act_tol=0.45*std*dmul, w=w, **extra)


def run_job(job):
    spec = job["pop"]
    popseed = job.get("popseed", 1)
    P = get_pop(spec, popseed)
    X, xy, std = P[0], P[1], P[2]
    N = X.shape[1]
    if job["goal"]["kind"] == "static":
        goal = StaticGoal(P[3])
    else:
        goal = mk_goal(job["goal"], X, std, N)
    S = job.get("S")
    
    span = len(X) if job["pop"][0] in ("sync", "field", "shifted", "corr", "burst") else 500
    if S is None:
        f = hdata.active_fraction(X, goal, tmax=span)
        S = max(4, int(round(f*N/job["load"])))
    kw = dict(job.get("kw", {}))
    pol = job["pol"]
    if pol in ALIAS:
        pol, extra = ALIAS[pol]
        kw = {**extra, **kw}
    if "aloha_mult" in kw:
        f = hdata.active_fraction(X, goal, tmax=span)
        kw["aloha_p"] = kw.pop("aloha_mult")*S/max(f*N, 1e-9)
    if "ata_mult" in kw:
        kw["ata_gamma"] = max(2.0, kw.pop("ata_mult")*N/max(S, 1))
    kw.setdefault("detail", job.get("detail", False))
    r = simulate(X, xy, pol, goal, S=S, seed=job["seed"], **kw)
    if job.get("want_sync"):
        r["sync_index"] = fields.synchrony_index(X, goal, tmax=max(span, 3000))
        r["active_frac"] = hdata.active_fraction(X, goal, tmax=span)
    keep = {k: r[k] for k in KEEP + ["sync_index", "active_frac"] if k in r}
    return dict(key=job["key"], **{k: v for k, v in job.items() if k in ("tag", "pol", "rep")}, res=keep)


def execute(name, jobs):
    path = os.path.join(OUT, f"ext_{name}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf8") as f:
            for line in f:
                done.add(json.loads(line)["key"])
    pending = [j for j in jobs if j["key"] not in done]
    print(f"[{name}] {len(jobs)} runs, {len(pending)} to do", flush=True)
    t0 = time.time()
    if pending:
        with Pool(WORKERS, maxtasksperchild=40) as P, open(path, "a", encoding="utf8") as f:
            for i, r in enumerate(P.imap_unordered(run_job, pending, chunksize=1)):
                f.write(json.dumps(r) + "\n"); f.flush()
                if (i + 1) % 200 == 0:
                    print(f"  {name}: {i+1}/{len(pending)} {time.time()-t0:.0f}s", flush=True)
    print(f"[{name}] finished {time.time()-t0:.0f}s", flush=True)


def J(name, tag, pol, rep, cfg=None, **over):
    
    base = dict(CONFIGS[cfg]) if cfg else {}
    base.update(over)
    kw = base.get("kw", {})
    key = f"{tag}|{pol}|{rep}|{json.dumps(kw, sort_keys=True)}"
    return dict(key=key, tag=tag, pol=pol, rep=rep, seed=rep, popseed=rep + 1, **base)



def s_tune():
    
    jobs = []
    for cfg in CONFIGS:
        c = CONFIGS[cfg]
        spec = c["pop"]
        half = ("rep", spec[1], spec[2], spec[3]//2)
        for m in (0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0):
            jobs.append(J("tune", f"{cfg}:ata:{m}", "ATA", 0, pop=half, goal=c["goal"], S=c["S"], kw=dict(ata_mult=m)))
        for eta in (0.01, 0.02, 0.05, 0.1, 0.2, 0.4):
            for pen in (0.125, 0.25, 0.5, 1.0):
                jobs.append(J("tune", f"{cfg}:ltp:{eta}:{pen}", "LTP", 0, pop=half, goal=c["goal"], S=c["S"],
                              kw=dict(ltp_eta=eta, ltp_pen=pen)))
        for m in (0.0625, 0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
            jobs.append(J("tune", f"{cfg}:opt:{m}", "ALOHA-OPT", 0, pop=half, goal=c["goal"], S=c["S"], kw=dict(aloha_mult=m)))
    return jobs



GRID_R = dict(rho=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), c=(0.125, 0.25, 0.5, 1.0), eps=(0.5, 1.0))
GRID_AR = dict(rho=(0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0), c=(0.125, 0.25, 0.5, 1.0))
GUARD_KW = dict(freeze_idle=True, fast_down=3.0)


def s_calib():
    jobs = []
    for cfg in CONFIGS:
        c = CONFIGS[cfg]
        spec = c["pop"]
        half = ("rep", spec[1], spec[2], spec[3]//2)
        for rho in GRID_R["rho"]:
            for ce in GRID_R["c"]:
                for eps in GRID_R["eps"]:
                    jobs.append(J("calib", f"{cfg}:hr:{rho}:{ce}:{eps}", "HORAE", 0, pop=half, goal=c["goal"], S=c["S"],
                                  kw=dict(hz_thresh=rho, ceil_rel=ce, hz_exp=eps, hz_exempt=True, **GUARD_KW)))
        for rho in GRID_AR["rho"]:
            for ce in GRID_AR["c"]:
                jobs.append(J("calib", f"{cfg}:ar:{rho}:{ce}", "AGE", 0, pop=half, goal=c["goal"], S=c["S"],
                              kw=dict(hz_thresh=rho, ceil_rel=ce, **GUARD_KW)))
    return jobs


def select_r(what="hr"):
    
    vals = {}
    for line in open(os.path.join(OUT, "ext_calib.jsonl"), encoding="utf8"):
        r = json.loads(line)
        t = r["tag"].split(":")
        if t[1] == what:
            vals.setdefault(tuple(float(x) for x in t[2:]), []).append(np.log(r["res"]["vwaoi"]))
    ranked = sorted(vals, key=lambda k: np.mean(vals[k]))
    return ranked, {k: float(np.exp(np.mean(v))) for k, v in vals.items()}


def selected_r(width=0.01):
    
    ranked, gm = select_r("hr")
    tied = [k for k in ranked if gm[k]/gm[ranked[0]] <= 1 + width]
    dq = {}
    for line in open(os.path.join(OUT, "ext_calib.jsonl"), encoding="utf8"):
        r = json.loads(line)
        t = r["tag"].split(":")
        if t[1] == "hr":
            q = r["res"].get("alert_f1")
            q = q if q is not None else 1.0/r["res"]["track_nrmse"]
            dq.setdefault(tuple(float(x) for x in t[2:]), []).append(np.log(q))
    return max(tied, key=lambda k: float(np.mean(dq[k]))), tied


def tuned_settings():
    path = os.path.join(OUT, "ext_tune.jsonl")
    best = {}
    if not os.path.exists(path):
        return best
    for line in open(path, encoding="utf8"):
        r = json.loads(line)
        cfg, what = r["tag"].split(":")[0], r["tag"].split(":")[1]
        v = r["res"]["vwaoi"]
        if (cfg, what) not in best or v < best[(cfg, what)][0]:
            best[(cfg, what)] = (v, r["tag"].split(":")[2:])
    return best


POL_MAIN = ["HORAE", "HORAE-G", "HORAE-R", "AGE", "VOI", "ATA", "LTP", "ALOHA", "ALOHA-ADP", "ADP-G", "AOII", "GOAL",
            "TDMA", "ACS", "RG", "MAXWEIGHT", "MWQ"]


def s_main():
    T = tuned_settings()
    jobs = []
    for cfg in CONFIGS:
        pols = POL_MAIN + ["AGE-R", "AGE-RE"] + (["IMP"] if CONFIGS[cfg]["goal"].get("wc") else [])
        for rep in range(20):
            for p in pols:
                jobs.append(J("main", cfg, p, rep, cfg, detail=True))
            if T:
                g = float(T[(cfg, "ata")][1][0])
                jobs.append(J("main", cfg, "ATA", rep, cfg, detail=True, kw=dict(ata_mult=g)))
                eta, pen = map(float, T[(cfg, "ltp")][1])
                jobs.append(J("main", cfg, "LTP", rep, cfg, detail=True, kw=dict(ltp_eta=eta, ltp_pen=pen)))
                m = float(T[(cfg, "opt")][1][0])
                jobs.append(J("main", cfg, "ALOHA-OPT", rep, cfg, detail=True, kw=dict(aloha_mult=m)))
    return jobs


def s_variance():
    
    jobs = []
    for cfg in HEAD:
        for p in ("HORAE", "ALOHA-ADP", "ALOHA"):
            for r in range(10):
                j = J("variance", f"{cfg}:pop", p, r, cfg); j["seed"] = 0; jobs.append(j)
                j = J("variance", f"{cfg}:chan", p, r, cfg); j["popseed"] = 1; jobs.append(j)
    return jobs


def s_raw():
    jobs = []
    for r in range(10):
        for p in POL_MAIN:
            jobs.append(J("raw", "intel", p, r, pop=("raw", "intel", None), goal=dict(ds="intel", kind="alert"), S=10))
            for S in (1, 2, 3, 4):
                jobs.append(J("raw", f"beijing:S{S}", p, r, pop=("raw", "beijing", 6000), goal=dict(ds="beijing", kind="alert"), S=S))
                jobs.append(J("raw", f"intel:S{S}", p, r, pop=("raw", "intel", None), goal=dict(ds="intel", kind="alert"), S=S))
    return jobs


def s_segments():
    jobs = []
    spec = ("seg", "beijing", 1200)
    for r in range(10):
        for p in POL_MAIN:
            jobs.append(J("segments", "alert", p, r, pop=spec, goal=dict(ds="beijing", kind="alert"), S=5))
            jobs.append(J("segments", "track", p, r, pop=spec, goal=dict(ds="beijing", kind="track"), S=16))
            jobs.append(J("segments", "alert_w", p, r, pop=spec, goal=dict(ds="beijing", kind="alert", wc=8.0), S=5))
    return jobs


def s_operating():
    jobs = []
    for cfg in HEAD:
        for tau in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
            for p in ("HORAE", "HORAE-R", "AGE", "ALOHA-ADP"):
                for r in range(10):
                    jobs.append(J("operating", f"{cfg}:{tau}", p, r, cfg, kw=dict(target=tau)))
    
    for S in (10, 5, 42):
        for tau in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            for p in ("HORAE", "HORAE-R", "ALOHA-ADP"):
                for r in range(5):
                    jobs.append(J("operating", f"beijing_alert_S{S}:{tau}", p, r, pop=CONFIGS["beijing_alert"]["pop"],
                                  goal=CONFIGS["beijing_alert"]["goal"], S=S, kw=dict(target=tau)))
    return jobs


def s_meanfield():
    jobs = []
    for N, S in ((500, 10), (2000, 20), (2000, 50), (5000, 20)):
        T = 8000 if N <= 2000 else 12000
        for sig in (0.0, 0.5, 1.0, 1.5):
            pop = ("static", N, sig, T)
            g = dict(kind="static")
            for r in range(5):
                for p, ex in (("HORAE", (1., 1.)), ("AGE", (1., 1.)), ("VOI", (1., 1.)), ("ALOHA-ADP", (1., 1.)),
                              ("MAXWEIGHT", (1., 1.)), ("MWQ", (1., 1.)), ("IDX", (1.5, 2.)), ("IDX", (2., 3.)), ("IDX", (1., 2.)), ("IDX", (.5, 1.))):
                    j = J("meanfield", f"{N}:{S}:{sig}:{ex[0]}:{ex[1]}", p, r, pop=pop, goal=g, S=S,
                          kw=dict(warmup=T//2, exps=list(ex)))
                    j["popseed"] = 1
                    jobs.append(j)
    for N, S in ((2000, 20), (500, 10), (5000, 20)):
        T = 8000 if N <= 2000 else 12000
        for sig in (0.0, 1.0, 1.5):
            pop = ("static", N, sig, T)
            c = get_pop(pop, 1)[3]
            for eps in (0.5, 1.0):
                for rf in (0.7, 0.85, 0.95, 1.0, 1.2):
                    rho = rf*np.e*float(np.mean(c**eps))
                    for r in range(5):
                        j = J("meanfield", f"thr:{N}:{S}:{sig}:{eps}:{rf}", "HORAE", r, pop=pop, goal=dict(kind="static"), S=S,
                              kw=dict(warmup=T//2, hz_thresh=rho, hz_exp=eps))
                        j["popseed"] = 1
                        jobs.append(j)
    pop = ("static", 2000, 1.0, 8000)
    for tau in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        for r in range(5):
            j = J("meanfield", f"tau:{tau}", "HORAE", r, pop=pop, goal=dict(kind="static"), S=20, kw=dict(warmup=4000, target=tau))
            j["popseed"] = 1
            jobs.append(j)
    return jobs


def s_exponents():
    jobs = []
    for cfg in HEAD:
        for a in (0.0, 0.5, 1.0, 1.5, 2.0):
            for b in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
                for r in range(5):
                    jobs.append(J("exponents", f"{cfg}:{a}:{b}", "IDX", r, cfg, kw=dict(exps=[a, b])))
    return jobs


def s_sensitivity():
    jobs = []
    grids = {"v_floor": (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5), "kappa": (1.0, 2.0, 3.0, 5.0, 8.0),
             "boost": (0.0, 1.0, 2.0, 4.0, 8.0, 16.0), "rho_mul": (0.25, 0.5, 1.0, 2.0, 4.0),
             "delta_mul": (0.5, 1.0, 2.0), "pct": (80, 85, 90, 92, 95, 98)}
    for cfg in HEAD:
        c = CONFIGS[cfg]
        for name, vals in grids.items():
            if c["goal"]["kind"] == "track" and name in ("kappa", "boost", "rho_mul", "pct"):
                continue
            for v in vals:
                g = dict(c["goal"]); g[name] = v
                for p in ("HORAE", "HORAE-R", "ALOHA-ADP", "AGE"):
                    for r in range(5):
                        jobs.append(J("sensitivity", f"{cfg}:{name}:{v}", p, r, pop=c["pop"], goal=g, S=c["S"]))
        for eta in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
            for p in ("HORAE", "HORAE-R", "ALOHA-ADP", "AGE"):
                for r in range(5):
                    jobs.append(J("sensitivity", f"{cfg}:eta:{eta}", p, r, cfg, kw=dict(gain_up=eta)))
        for b0 in (1e-4, 1e-3, 0.02, 0.1, 1.0):
            for r in range(5):
                jobs.append(J("sensitivity", f"{cfg}:beta0:{b0}", "HORAE", r, cfg, kw=dict(beta0=b0)))
        for bc in (0.1, 1.0, 10.0, 50.0):
            for r in range(5):
                jobs.append(J("sensitivity", f"{cfg}:beta_ceil:{bc}", "HORAE", r, cfg, kw=dict(beta_ceil=bc)))
    
    for cfg, S in (("beijing_track", 32), ("beijing_alert", 21)):
        c = CONFIGS[cfg]
        for wc in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
            for wf in (0.05, 0.15, 0.3):
                g = dict(c["goal"]); g["wc"] = wc; g["wf"] = wf
                for p in ("HORAE", "HORAE-R", "ALOHA-ADP", "AGE", "VOI"):
                    for r in range(5):
                        jobs.append(J("sensitivity", f"{cfg}:w:{wc}:{wf}", p, r, pop=c["pop"], goal=g, S=S))
    return jobs


def s_value():
    jobs = []
    cases = [("true", {})] + [(f"noise:{s}", dict(v_noise=s)) for s in (0.25, 0.5, 1.0, 2.0)] + \
            [(f"bias:{s}", dict(v_bias=s)) for s in (0.25, 0.5, 1.0, 2.0)] + \
            [(f"bits:{b}", dict(v_bits=b)) for b in (1, 2, 3, 4)] + \
            [("invert", dict(v_mode="invert")), ("shuffle", dict(v_mode="shuffle"))]
    for cfg in HEAD + ["beijing_track_w", "beijing_alert_w"]:
        for tag, kw in cases:
            for r in range(5):
                jobs.append(J("value", f"{cfg}:{tag}", "HORAE", r, cfg, kw=kw))
                jobs.append(J("value", f"{cfg}:{tag}", "HORAE-R", r, cfg, kw=kw))
        for r in range(5):
            for p in ("ALOHA-ADP", "AGE", "ALOHA"):
                jobs.append(J("value", f"{cfg}:base", p, r, cfg))
    
    for cfg, unit in (("intel_alert", 1.0), ("beijing_alert", 10.0), ("beijing_track", 10.0)):
        for m in (0.1, 0.25, 0.5, 1.0, 2.0):
            for p in ("HORAE", "HORAE-R", "ALOHA-ADP", "AGE"):
                for r in range(5):
                    jobs.append(J("value", f"{cfg}:xnoise:{m}", p, r, cfg, kw=dict(x_noise=m*unit)))
        if CONFIGS[cfg]["goal"]["kind"] == "alert":
            for m in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
                for p in ("HORAE", "HORAE-R", "ALOHA-ADP", "AGE"):
                    for r in range(5):
                        jobs.append(J("value", f"{cfg}:thetaerr:{m}", p, r, cfg, kw=dict(theta_err=m*unit)))
    return jobs


def s_feedback():
    jobs = []
    cases = [("ideal", {})] + [(f"loss:{x}", dict(beacon_loss=x)) for x in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)] + \
            [(f"common:{x}", dict(common_loss=x)) for x in (0.1, 0.3, 0.5)] + \
            [(f"delay:{x}", dict(delay=x)) for x in (1, 2, 5, 10, 20)] + \
            [(f"incons:{x}", dict(idle_noise=x)) for x in (0.05, 0.1, 0.2, 0.3)] + \
            [(f"shared:{x}", dict(p_loss=x)) for x in (0.05, 0.1, 0.2, 0.3)] + \
            [(f"duty:{x}", dict(duty=x)) for x in (2, 5, 10, 20, 50, 100)] + \
            [(f"misalign:{x}", dict(misalign=x)) for x in (0.1, 0.25, 0.5, 1.0)] + \
            [("noack", dict(no_ack=True))]
    for cfg in HEAD:
        for tag, kw in cases:
            for p in ("HORAE", "HORAE-R", "ALOHA-ADP", "AGE"):
                for r in range(5):
                    jobs.append(J("feedback", f"{cfg}:{tag}", p, r, cfg, kw=kw))
            if tag.startswith(("loss", "duty", "delay", "ideal")):
                for r in range(5):
                    jobs.append(J("feedback", f"{cfg}:{tag}", "HORAE-G", r, cfg, kw=kw))
    return jobs


def s_feedback2():
    
    jobs = []
    bc = [("bc:ideal", dict(beta_bcast=True))] + [(f"bc:loss:{x}", dict(beta_bcast=True, beacon_loss=x)) for x in (0.05, 0.1, 0.3, 0.5)] + \
         [("bc:common:0.3", dict(beta_bcast=True, common_loss=0.3))] + [(f"bc:duty:{x}", dict(beta_bcast=True, duty=x)) for x in (2, 5, 20, 100)] + \
         [(f"bc:delay:{x}", dict(beta_bcast=True, delay=x)) for x in (1, 5)]
    lowgain = [(f"gain:{d}:{g}", dict(delay=d, gain_up=g)) for d in (5, 10, 20) for g in (0.1, 0.05, 0.02)]
    mis = [(f"guard:{x}", dict(misalign=x, guard_t=0.1)) for x in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0)]
    inc = [(f"incs:{x}", dict(idle_noise=x, gain_up=0.25)) for x in (0.05, 0.1, 0.2, 0.3)]
    for cfg in HEAD:
        for tag, kw in bc + lowgain + mis + inc:
            for p in ("HORAE", "HORAE-R", "ALOHA-ADP"):
                for r in range(5):
                    jobs.append(J("feedback2", f"{cfg}:{tag}", p, r, cfg, kw=kw))
    return jobs


GUARD = {"base": {}, "ceil0.1": dict(beta_ceil=0.1), "ceil1": dict(beta_ceil=1.0), "ceil10": dict(beta_ceil=10.0),
         "freeze": dict(freeze_idle=True),
         "fast1.5": dict(fast_down=1.5), "fast2": dict(fast_down=2.0), "fast3": dict(fast_down=3.0),
         "fast5": dict(fast_down=5.0), "fast10": dict(fast_down=10.0),
         "G": dict(beta_ceil=1.0, freeze_idle=True, fast_down=3.0),
         "pmin1e-3": dict(p_min=1e-3), "pmin1e-2": dict(p_min=1e-2), "pmin5e-2": dict(p_min=5e-2),
         "expl20": dict(explore_every=20), "expl10": dict(explore_every=10), "expl5": dict(explore_every=5)}


def regimes():
    bj = dict(ds="beijing", kind="alert")
    R = [(c, dict(pop=CONFIGS[c]["pop"], goal=CONFIGS[c]["goal"], S=CONFIGS[c]["S"])) for c in HEAD]
    for s in (0.25, 0.5, 0.75, 1.0):
        R.append((f"sync{s}", dict(pop=("sync", s), goal=bj, load=2.0)))
    for S, reps in ((4, 8), (4, 42), (4, 84), (4, 170), (11, 170), (22, 170)):
        R.append((f"over_S{S}_N{reps*12}", dict(pop=("rep", "beijing", reps, 3000), goal=bj, S=S)))
    for S in (1, 2, 4):
        R.append((f"raw_S{S}", dict(pop=("raw", "beijing", 6000), goal=bj, S=S)))
    return R


def s_guards():
    jobs = []
    for name, base in regimes():
        for gname, kw in GUARD.items():
            for r in range(5):
                jobs.append(J("guards", f"{name}:{gname}", "HORAE", r, kw=kw, **base))
        for p, kw in (("ALOHA-ADP", {}), ("ADP-G", {}), ("ALOHA", {}), ("AGE", {}), ("AGE-G", {}), ("HORAE-R", {}), ("AGE-R", {})):
            for r in range(5):
                jobs.append(J("guards", f"{name}:{p}", p, r, kw=kw, **base))
    return jobs


POL_OVER = ["HORAE", "HORAE-G", "HORAE-R", "AGE", "ALOHA", "ALOHA-ADP", "ADP-G", "ATA"]


def s_overload():
    jobs = []
    bj = dict(ds="beijing", kind="alert")
    for S in (4, 11, 22, 44):
        for reps in (8, 20, 42, 84, 170):
            base = dict(pop=("rep", "beijing", reps, 3000), goal=bj, S=S, want_sync=False)
            for r in range(10):
                for p in POL_OVER:
                    jobs.append(J("overload", f"S{S}:N{reps*12}", p, r, **base))
            for r in range(3):
                jobs.append(J("overload", f"S{S}:N{reps*12}", "HORAE", r, kw=dict(beta_floor=1e-4, idle_floor=0.5), **base))
    
    for load, S in ((2.0, 11), (4.0, 5), (8.0, 4)):
        for r in range(5):
            for p in ("HORAE", "HORAE-G", "HORAE-R", "AGE", "VOI", "ATA", "ALOHA", "ALOHA-ADP", "ADP-G"):
                jobs.append(J("overload", f"imp:{load}", p, r, pop=CONFIGS["beijing_alert"]["pop"],
                              goal=dict(ds="beijing", kind="alert", wc=8.0), S=S, detail=True))
    
    for reps in (84, 170):
        for r in range(5):
            for p in ("HORAE", "HORAE-G", "HORAE-R", "AGE", "VOI", "ATA", "ALOHA", "ALOHA-ADP", "ADP-G"):
                jobs.append(J("overload", f"impov:N{reps*12}", p, r, pop=("rep", "beijing", reps, 3000),
                              goal=dict(ds="beijing", kind="alert", wc=8.0), S=4, want_sync=False, detail=True))
    return jobs


def s_synchrony():
    jobs = []
    bj = dict(ds="beijing", kind="alert")
    for s in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
        for r in range(10):
            for p in POL_OVER + ["ALOHA-OPT"]:
                kw = dict(aloha_mult=2.0) if p == "ALOHA-OPT" else {}
                j = J("synchrony", f"{s}", p, r, pop=("sync", s), goal=bj, load=2.0, kw=kw)
                j["want_sync"] = (r == 0 and p == "HORAE")
                jobs.append(j)
    return jobs


def s_correlation():
    jobs = []
    for phi in (0.9, 0.99, 0.999):
        for ell in (None, 0.05, 0.3):
            for c in (0.0, 0.25, 0.5, 0.75):
                for r in range(5):
                    for p in ("HORAE", "HORAE-G", "HORAE-R", "ALOHA-ADP", "ADP-G", "AGE", "ALOHA"):
                        j = J("correlation", f"{phi}:{ell}:{c}", p, r, pop=("corr", phi, ell, c),
                              goal=dict(ds="field", kind="alert"), load=2.0)
                        j["want_sync"] = (p == "HORAE")
                        jobs.append(j)
    for m in (1, 10, 100, 1000):
        for r in range(5):
            for p in ("HORAE", "HORAE-G", "HORAE-R", "ALOHA-ADP", "ADP-G", "AGE", "ALOHA"):
                j = J("correlation", f"burst:{m}", p, r, pop=("burst", m), goal=dict(ds="field", kind="alert"), load=2.0)
                j["want_sync"] = (p == "HORAE")
                jobs.append(j)
    return jobs


POL_REAL = ["HORAE", "HORAE-G", "HORAE-R", "AGE", "VOI", "ALOHA", "ALOHA-ADP", "ADP-G", "ATA", "GOAL", "TDMA", "RG", "MAXWEIGHT"]


def s_fields():
    jobs = []
    for name in ("aqs", "o3", "pems"):
        if not os.path.exists(os.path.join(HERE, "data", f"{name}_field.npz")):
            continue
        for kind in ("alert", "track"):
            for load in (0.5, 1.0, 2.0, 4.0):
                for r in range(5):
                    for p in POL_REAL:
                        j = J("fields", f"{name}:{kind}:{load}", p, r, pop=("field", name, None),
                              goal=dict(ds="field", kind=kind), load=load)
                        j["popseed"] = 1
                        j["want_sync"] = (r == 0 and p == "HORAE")
                        jobs.append(j)
            
            for load in (1.0, 2.0):
                for r in range(5):
                    for p in ("HORAE", "HORAE-R", "ALOHA-ADP", "AGE", "ALOHA"):
                        j = J("fields", f"{name}:{kind}:{load}:shifted", p, r, pop=("shifted", name, 12, None),
                              goal=dict(ds="field", kind=kind), load=load)
                        j["want_sync"] = (r == 0 and p == "HORAE")
                        jobs.append(j)
    return jobs


def s_heldout():
    jobs = []
    
    for cfg, ds, reps, tmax, S in (("intel", "intel", 10, 1500, 62), ("beijing", "beijing", 40, 6000, 21)):
        X0 = hdata.load_base(f"{ds}_field.npz")[0][:tmax]
        h = X0.shape[0]//2
        th_cal = float(np.percentile(X0[:h], 92)); th_all = float(np.percentile(X0, 92)); th_test = float(np.percentile(X0[h:], 92))
        for r in range(10):
            for lab, th in (("calib", th_cal), ("whole", th_all), ("test", th_test)):
                for p in ("HORAE", "HORAE-R", "ALOHA-ADP", "AGE", "ALOHA"):
                    jobs.append(J("heldout", f"{cfg}:{lab}", p, r, pop=("part", ds, reps, tmax),
                                  goal=dict(ds=ds, kind="alert", theta=th), S=S))
    
    for r in range(10):
        for p in ("HORAE", "HORAE-R", "ALOHA-ADP", "AGE", "ALOHA"):
            jobs.append(J("heldout", "beijing_track:second", p, r, pop=("part", "beijing", 40, 6000), goal=dict(ds="beijing", kind="track"), S=32))
    for cfg in HEAD:
        c = CONFIGS[cfg]
        T = c["pop"][3]
        for w in (5, T//10, T//4, T//2):
            for r in range(5):
                for p in ("HORAE", "HORAE-R", "ALOHA-ADP"):
                    jobs.append(J("heldout", f"{cfg}:warm:{w}", p, r, cfg, kw=dict(warmup=w)))
        for frac in (4, 2, 1):
            spec = ("rep", c["pop"][1], c["pop"][2], T//frac)
            for r in range(5):
                for p in ("HORAE", "HORAE-R", "ALOHA-ADP"):
                    jobs.append(J("heldout", f"{cfg}:len:{T//frac}", p, r, pop=spec, goal=c["goal"], S=c["S"]))
    return jobs


def s_capture():
    jobs = []
    bj = dict(ds="beijing", kind="alert")
    for cdb in (None, 9.0, 6.0, 3.0, 0.0):
        kw = {} if cdb is None else dict(capture_db=cdb, fading=True)
        for r in range(10):
            for p in ("HORAE", "HORAE-G", "HORAE-R", "AGE", "ATA", "ALOHA", "ALOHA-ADP", "ADP-G"):
                jobs.append(J("capture", f"{cdb}", p, r, pop=CONFIGS["beijing_alert"]["pop"], goal=bj, S=10, kw=kw))
    return jobs


def s_unheard():
    
    jobs = []
    bj = dict(ds="beijing", kind="alert")
    for reps in (42, 84, 170):
        for r in range(10):
            for p in POL_OVER + ["VOI", "RG"]:
                jobs.append(J("unheard", f"S4:N{reps*12}", p, r, pop=("rep", "beijing", reps, 3000), goal=bj, S=4))
    return jobs


def s_thresh():
    
    jobs = []
    for cfg in CONFIGS:
        for rho in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
            for ce in (0.125, 0.25, 0.5, 1.0):
                for eps in (0.5, 1.0):
                    for r in range(5):
                        jobs.append(J("thresh", f"{cfg}:{rho}:{ce}:{eps}", "HORAE", r, cfg,
                                      kw=dict(hz_thresh=rho, ceil_rel=ce, hz_exp=eps, hz_exempt=True, freeze_idle=True, fast_down=3.0)))
    bj = dict(ds="beijing", kind="alert")
    regs = [(c, dict(pop=CONFIGS[c]["pop"], goal=CONFIGS[c]["goal"], S=CONFIGS[c]["S"])) for c in CONFIGS]
    regs += [("sync0.5", dict(pop=("sync", 0.5), goal=bj, load=2.0)), ("sync1.0", dict(pop=("sync", 1.0), goal=bj, load=2.0)),
             ("over4_96", dict(pop=("rep", "beijing", 8, 3000), goal=bj, S=4))]
    for name in ("aqs", "o3", "pems"):
        regs.append((f"{name}_a1", dict(pop=("field", name, None), goal=dict(ds="field", kind="alert"), load=1.0)))
    full = dict(hz_thresh=2.0, ceil_rel=0.25, hz_exp=1.0, hz_exempt=True, freeze_idle=True, fast_down=3.0)
    parts = {"full": full, "noexempt": {**full, "hz_exempt": False}, "nothresh": {k: v for k, v in full.items() if k not in ("hz_thresh", "hz_exp", "hz_exempt")},
             "noguard": {k: v for k, v in full.items() if k not in ("freeze_idle", "fast_down")},
             "noceil": {**{k: v for k, v in full.items() if k != "ceil_rel"}, "beta_ceil": 1.0}}
    for reg, base in regs:
        for pn, kw in parts.items():
            for r in range(5):
                j = J("thresh", f"part:{reg}:{pn}", "HORAE", r, kw=kw, **base)
                if base["pop"][0] == "field":
                    j["popseed"] = 1
                jobs.append(j)
    
    regs2 = [(f"{name}_t05", dict(pop=("field", name, None), goal=dict(ds="field", kind="track"), load=0.5), {}) for name in ("aqs", "o3")]
    for cfg in ("beijing_alert", "beijing_track"):
        for s_ in (1.0, 2.0):
            regs2.append((f"{cfg}:bias{s_}", dict(pop=CONFIGS[cfg]["pop"], goal=CONFIGS[cfg]["goal"], S=CONFIGS[cfg]["S"]), dict(v_bias=s_)))
    for te in (0.5, -0.5):
        c = CONFIGS["intel_alert"]
        regs2.append((f"intel_alert:th{te}", dict(pop=c["pop"], goal=c["goal"], S=c["S"]), dict(theta_err=te)))
    for reg, base, ex in regs2:
        for pn in ("full", "nothresh", "noceil"):
            for r in range(5):
                j = J("thresh", f"part:{reg}:{pn}", "HORAE", r, kw={**parts[pn], **ex}, **base)
                if base["pop"][0] == "field":
                    j["popseed"] = 1
                jobs.append(j)
        for r in range(5):
            if ex:
                jobs.append(J("thresh", f"part:{reg}:rt", "HORAE", r, kw={**full, "hz_exp": 0.5, **ex}, **base))
                jobs.append(J("thresh", f"part:{reg}:adp", "ALOHA-ADP", r, kw=dict(ex), **base))
    
    for rho in (0.5, 1.0, 2.0, 4.0, 8.0):
        for ex in (True, False):
            for r in range(5):
                jobs.append(J("thresh", f"exempt:beijing_alert:{rho}:{ex}", "HORAE", r, "beijing_alert", kw={**full, "hz_thresh": rho, "hz_exempt": ex}))
    
    for cfg in ("intel_alert", "beijing_alert", "beijing_track"):
        for tau in (0.25, 0.5, 0.75, 1.0):
            for M in (1, 5, 20, 100):
                for r in range(5):
                    jobs.append(J("thresh", f"energy:{cfg}:{tau}:{M}", "HORAE-R", r, cfg, kw=dict(target=tau, duty=M)))
    return jobs


def s_loadsweep():
    
    R = json.load(open(os.path.join(OUT, "results.json"), encoding="utf8"))["E2"]
    jobs = []
    for ds, reps, T in (("intel", 10, 1500), ("beijing", 40, 6000)):
        for S in R[ds]["S"]:
            for p in ("HORAE", "HORAE-R"):
                for r in range(3):
                    j = J("loadsweep", f"{ds}:{S}", p, r, pop=("rep", ds, reps, T), goal=dict(ds=ds, kind="alert"), S=S)
                    j["popseed"] = 1
                    jobs.append(j)
    return jobs


STUDIES = dict(calib=s_calib, unheard=s_unheard, thresh=s_thresh, loadsweep=s_loadsweep, feedback2=s_feedback2, tune=s_tune, main=s_main, variance=s_variance, raw=s_raw, segments=s_segments, operating=s_operating,
               meanfield=s_meanfield, exponents=s_exponents, sensitivity=s_sensitivity, value=s_value, feedback=s_feedback,
               guards=s_guards, overload=s_overload, synchrony=s_synchrony, correlation=s_correlation, fields=s_fields,
               heldout=s_heldout, capture=s_capture)

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--list":
        for k, f in STUDIES.items():
            print(k, len(f()))
        sys.exit(0)
    for name in args:
        execute(name, STUDIES[name]())
    print("ALL DONE", flush=True)
