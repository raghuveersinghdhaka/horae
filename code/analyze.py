
import json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def load(study):
    path = os.path.join(OUT, f"ext_{study}.jsonl")
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            r["kw"] = json.loads(r["key"].rsplit("|", 1)[1])
            recs.append(r)
    return recs


def table(recs, metric="vwaoi", variant=None):
    
    out = {}
    for r in recs:
        p = variant(r) if variant else r["pol"]
        if p is None or metric not in r["res"]:
            continue
        out.setdefault((r["tag"], p), {})[r["rep"]] = r["res"][metric]
    return out


def mean_ci(vals):
    a = np.asarray(list(vals), float)
    a = a[~np.isnan(a)]
    if len(a) < 2:
        return float(a.mean()) if len(a) else float("nan"), 0.0
    return float(a.mean()), float(stats.t.ppf(0.975, len(a) - 1)*a.std(ddof=1)/np.sqrt(len(a)))


def paired(a, b, B=10000, seed=1):
    """a, b: {rep: value}. Ratio is b/a (for AoI: how many times lower a is than b)."""
    reps = sorted(set(a) & set(b))
    x = np.array([a[r] for r in reps], float); y = np.array([b[r] for r in reps], float)
    lr = np.log(y/x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(lr), size=(B, len(lr)))
    boot = lr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    t = stats.ttest_1samp(lr, 0.0)
    try:
        w = stats.wilcoxon(lr) if np.any(lr != 0) else None
    except ValueError:
        w = None
    return dict(n=len(reps), a=float(x.mean()), b=float(y.mean()), ratio=float(np.exp(lr.mean())),
                lo=float(np.exp(lo)), hi=float(np.exp(hi)), t=float(t.statistic), p_t=float(t.pvalue),
                p_w=float(w.pvalue) if w is not None else 1.0, sd_log=float(lr.std(ddof=1)))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty(len(p)); run = 0.0
    for i, k in enumerate(order):
        run = max(run, (len(p) - i)*p[k])
        adj[k] = min(1.0, run)
    return adj


def main_variant(r):
    if r["kw"]:
        return r["pol"] + "-T" if r["pol"] != "ALOHA-OPT" else "ALOHA-OPT"
    return r["pol"]


def show_main():
    recs = load("main")
    if not recs:
        return
    for metric in ("vwaoi", "alert_f1", "track_nrmse", "energy_J", "tx", "succ", "G_over_S"):
        tb = table(recs, metric, main_variant)
        tags = sorted({k[0] for k in tb})
        for tag in tags:
            pols = sorted({k[1] for k in tb if k[0] == tag})
            print(f"--- {tag} {metric} n={len(tb[(tag, pols[0])])}")
            base = tb.get((tag, "HORAE"))
            rows = []
            for p in pols:
                m, c = mean_ci(tb[(tag, p)].values())
                if base and p != "HORAE":
                    s = paired(base, tb[(tag, p)])
                    rows.append((p, m, c, s))
                else:
                    rows.append((p, m, c, None))
            ps = [r[3]["p_w"] for r in rows if r[3]]
            ad = holm(ps) if ps else []
            k = 0
            for p, m, c, s in sorted(rows, key=lambda r: r[1]):
                if s:
                    print("  %-11s %10.3f +- %7.3f  other/HORAE %6.3f [%6.3f,%6.3f] t %7.2f p_t %.1e p_w %.1e" % (p, m, c, s["ratio"], s["lo"], s["hi"], s["t"], s["p_t"], s["p_w"]))
                else:
                    print("  %-11s %10.3f +- %7.3f" % (p, m, c))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "main"
    if what == "main":
        show_main()
    else:
        recs = load(what)
        metrics = sys.argv[2].split(",") if len(sys.argv) > 2 else ["vwaoi"]
        for metric in metrics:
            tb = table(recs, metric, lambda r: r["pol"] + ("" if not r["kw"] or what in ("sensitivity", "value", "feedback", "guards", "operating", "exponents", "meanfield", "capture", "heldout", "overload") and False else ""))
            tags = []
            for k in tb:
                if k[0] not in tags:
                    tags.append(k[0])
            for tag in tags:
                vals = {p: mean_ci(tb[(tag, p)].values()) for (t, p) in tb if t == tag}
                print(tag, metric, " ".join("%s=%.3f" % (p, v[0]) for p, v in vals.items()))
