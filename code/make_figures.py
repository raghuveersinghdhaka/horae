
import json, os, sys
os.environ.setdefault("SOURCE_DATE_EPOCH", "1704067200")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import analyze as A
import energy_profiles as EP
from run_studies import tuned_settings

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures"); os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results", "results.json")))
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
                     "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
                     "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7.2, "lines.linewidth": 1.25, "lines.markersize": 4, "mathtext.fontset": "dejavuserif"})
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", purple="#CC79A7", sky="#56B4E9",
         yellow="#F0E442", grey="#999999", black="#111111", brown="#7B3F00")
PC = {"HORAE": C["blue"], "HORAE-G": C["sky"], "HORAE-R": "#14213D", "AGE-R": "#7FC97F", "AGE": C["green"], "VOI": C["purple"], "ATA": C["brown"], "ATA-T": C["brown"], "AOII": "#6A3D9A",
      "GOAL": "#B15928", "LTP": C["yellow"], "LTP-T": C["yellow"], "ALOHA": C["red"], "ALOHA-OPT": "#FB9A99", "ALOHA-ADP": C["orange"],
      "ADP-G": "#FDBF6F", "TDMA": "#7F7F7F", "ACS": "#A6A6A6", "RG": "#4D4D4D", "MAXWEIGHT": "#BDBDBD", "MWQ": "#D9D9D9"}
PN = {"HORAE": "HORAE", "HORAE-G": "HORAE-G", "HORAE-R": "HORAE-R", "AGE-R": "Age-R", "AGE": "Age-only", "VOI": "Value-only", "ATA": "ATA", "ATA-T": "ATA-T", "AOII": "AoII", "GOAL": "GOAL",
      "LTP": "LTP", "LTP-T": "LTP-T", "ALOHA": "S-ALOHA", "ALOHA-OPT": "T-ALOHA", "ALOHA-ADP": "Ad-ALOHA", "ADP-G": "ADP-G", "TDMA": "TDMA", "ACS": "ACS",
      "RG": "RG", "MAXWEIGHT": "Max-weight", "MWQ": "MW-Q"}
CFG = [("intel_alert", "Intel-Lab, alert", "alert_f1", "alert $F_1$"), ("beijing_alert", "Beijing, alert", "alert_f1", "alert $F_1$"),
       ("beijing_track", "Beijing, tracking", "track_nrmse", "tracking NRMSE")]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), metadata={"Creator": "", "Producer": ""})
    plt.close(fig); print("wrote", name)


def gm(vals):
    return float(np.exp(np.mean(np.log(list(vals)))))


def plain_log(ax, axis="both", xt=None, yt=None):
    
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    fmt = FuncFormatter(lambda v, _: ("%g" % v))
    if axis in ("x", "both") and xt is not None:
        ax.xaxis.set_major_locator(FixedLocator(xt)); ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_minor_formatter(NullFormatter())
    if axis in ("y", "both") and yt is not None:
        ax.yaxis.set_major_locator(FixedLocator(yt)); ax.yaxis.set_major_formatter(fmt); ax.yaxis.set_minor_formatter(NullFormatter())


TUNED = tuned_settings()


def variant(r):
    
    if not r["kw"]:
        return r["pol"]
    cfg = r["tag"]
    if r["pol"] == "ALOHA-OPT":
        return "ALOHA-OPT" if r["kw"].get("aloha_mult") == float(TUNED[(cfg, "opt")][1][0]) else None
    if r["pol"] == "ATA":
        return "ATA-T" if r["kw"].get("ata_mult") == float(TUNED[(cfg, "ata")][1][0]) else None
    if r["pol"] == "LTP":
        eta, pen = map(float, TUNED[(cfg, "ltp")][1])
        return "LTP-T" if (r["kw"].get("ltp_eta"), r["kw"].get("ltp_pen")) == (eta, pen) else None
    return None


TICK = {"intel_alert": [25, 30, 40, 50], "beijing_alert": [10, 20, 50, 100], "beijing_track": [3, 5, 10, 20, 30]}


def fig_main():
    recs = A.load("main")
    vw = A.table(recs, "vwaoi", variant)
    order = ["HORAE-R", "HORAE", "HORAE-G", "AGE-R", "AGE", "VOI", "ATA-T", "AOII", "GOAL", "LTP-T", "ALOHA", "ALOHA-OPT", "ALOHA-ADP", "TDMA", "ACS", "RG", "MAXWEIGHT"]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.1))
    fig.subplots_adjust(wspace=0.75, bottom=0.27, top=0.91, left=0.06, right=0.95)
    for ax, (cfg, title, mk, ml) in zip(axes, CFG):
        dq = A.table(recs, mk, variant)
        x = np.arange(len(order))
        m = [A.mean_ci(vw[(cfg, p)].values()) for p in order]
        for x_, a_, p_ in zip(x, m, order):
            ax.errorbar(x_, a_[0], yerr=a_[1], fmt="o", color=PC[p_], mec="#222", mew=0.3, ms=5, elinewidth=0.7, ecolor="#333", capsize=0, zorder=3)
        ax.axvline(12.5, color="#555", lw=0.6, ls=":")
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([PN[p] for p in order], rotation=90, fontsize=7.2)
        ax.set_title(title, pad=4)
        ax.set_ylabel("value-weighted AoI")
        ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True)
        ax2.plot(x, [A.mean_ci(dq[(cfg, p)].values())[0] for p in order], "D", color=C["black"], mec="white", mew=0.3, ms=3.8, ls="none", zorder=5)
        ax2.set_ylabel(ml); ax2.set_ylim(0, 1.05)
        plain_log(ax, "y", yt=TICK[cfg])
    fig.legend(handles=[Line2D([], [], marker="D", color=C["black"], ls="none", ms=3, label="decision quality (right axis)"),
                        Line2D([], [], marker="o", color="#777", mec="#222", mew=0.3, ls="none", ms=4.5, label="value-weighted AoI (left axis, log scale; color by policy)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    save(fig, "fig_main")


def fig_frontier():
    recs = A.load("main")
    vw = A.table(recs, "vwaoi", variant)
    pols = ["HORAE-R", "HORAE", "HORAE-G", "AGE-R", "AGE", "VOI", "ATA-T", "AOII", "GOAL", "LTP-T", "ALOHA", "ALOHA-ADP", "TDMA", "RG", "MAXWEIGHT"]
    MK = ["H", "o", "s", "8", "^", "v", "D", "P", "X", "<", ">", "p", "h", "*", "d"]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.75))
    fig.subplots_adjust(wspace=0.32, bottom=0.34, top=0.9, left=0.07, right=0.98)
    for ax, (cfg, title, mk, ml) in zip(axes, CFG):
        dq = A.table(recs, mk, variant)
        for p, m in reversed(list(zip(pols, MK))):
            ax.plot(A.mean_ci(vw[(cfg, p)].values())[0], A.mean_ci(dq[(cfg, p)].values())[0], m, color=PC[p], ms=5.2, mec="#222", mew=0.3, ls="none", zorder=3)
        if cfg == "intel_alert":
            ins = ax.inset_axes([0.46, 0.38, 0.48, 0.46])
            for p, m in reversed(list(zip(pols, MK))):
                ins.plot(A.mean_ci(vw[(cfg, p)].values())[0], A.mean_ci(dq[(cfg, p)].values())[0], m, color=PC[p], ms=4.5, mec="#222", mew=0.3, ls="none", zorder=3)
            ins.set_xlim(22.9, 24.05); ins.set_ylim(0.972, 0.991); ins.tick_params(labelsize=6, length=2)
            ins.set_xticks([23, 23.5, 24]); ins.set_yticks([0.98, 0.99]); ins.grid(True, alpha=0.3)
            ax.indicate_inset_zoom(ins, edgecolor="#555", alpha=0.8, lw=0.6)
            for sp in ("top", "right"):
                ins.spines[sp].set_visible(True)
        ax.set_xscale("log"); ax.set_xlabel("value-weighted AoI"); ax.set_ylabel(ml); ax.set_title(title, pad=4)
        plain_log(ax, "x", xt=TICK[cfg])
    handles = [Line2D([], [], marker=m, color=PC[p], mec="#222", mew=0.3, ls="none", ms=5, label=PN[p]) for p, m in zip(pols, MK)]
    fig.legend(handles=handles, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.02), columnspacing=0.9, handletextpad=0.2)
    save(fig, "fig_frontier")


def fig_load():
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.05))
    fig.subplots_adjust(bottom=0.36, wspace=0.48, top=0.92, left=0.07, right=0.98)
    DISP = ["HORAE", "AGE", "ATA", "ALOHA", "ALOHA-ADP", "TDMA", "MAXWEIGHT"]
    MK = dict(zip(DISP, ["o", "^", "D", ">", "p", "h", "d"]))
    LS = A.table(A.load("loadsweep"), "vwaoi")
    head = json.load(open(os.path.join(HERE, "results", "load_map.json"), encoding="utf8"))["headline"]
    for ax, ds, title in ((axes[0], "intel", "Intel-Lab, alert"), (axes[1], "beijing", "Beijing, alert")):
        E = R["E2"][ds]
        loads = [head[ds + "_alert"]["active"]/s_ for s_ in E["S"]]
        for p in DISP:
            st = dict(ls="--", mfc="none") if p == "AGE" else {}
            ax.plot(loads, E["by_pol"][p], marker=MK[p], color=PC[p], label=PN[p], ms=3.2, **st)
        ax.plot(loads, [np.mean(list(LS[(f"{ds}:{s_}", "HORAE-R")].values())) for s_ in E["S"]], marker="H", color=PC["HORAE-R"], label=PN["HORAE-R"], ms=3.4, lw=1.5, zorder=5)
        ax.axvline(1.0, color="#555", lw=0.6, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("offered load (active sources / $S$)"); ax.set_title(title, pad=4)
        plain_log(ax, "x", xt=[0.5, 1, 2, 4])
    plain_log(axes[0], "y", yt=[20, 30, 40, 60, 80]); plain_log(axes[1], "y", yt=[10, 20, 50, 100, 200, 400])
    axes[0].set_ylabel("value-weighted AoI"); axes[1].set_ylabel("value-weighted AoI")
    h, l = axes[0].get_legend_handles_labels()
    ax = axes[2]
    vw = A.table(A.load("operating"), "vwaoi")
    for cfg, lab, c in (("intel_alert", "Intel-Lab, alert", C["blue"]), ("beijing_alert", "Beijing, alert", C["red"]), ("beijing_track", "Beijing, tracking", C["green"])):
        taus = sorted({float(k[0].split(":")[1]) for k in vw if k[0].startswith(cfg + ":") and k[1] == "HORAE"})
        y = np.array([np.mean(list(vw[(f"{cfg}:{t}", "HORAE")].values())) for t in taus])
        ax.plot(taus, y/y[taus.index(1.0)], "o-", color=c, ms=3, label=lab)
        y = np.array([np.mean(list(vw[(f"{cfg}:{t}", "HORAE-R")].values())) for t in taus])
        ax.plot(taus, y/y[taus.index(1.0)], "H--", color=c, ms=3, lw=0.9)
    mf = A.table(A.load("meanfield"), "vwaoi")
    taus = sorted(float(k[0].split(":")[1]) for k in mf if k[0].startswith("tau:"))
    y = np.array([np.mean(list(mf[(f"tau:{t}", "HORAE")].values())) for t in taus])
    ax.plot(taus, y/y[taus.index(1.0)], "s", color=C["black"], ms=3.5, mfc="none", label="static values, simulated")
    tt = np.linspace(0.2, 3.1, 100)
    ax.plot(tt, np.exp(-1)/(tt*np.exp(-tt)), "-", color=C["grey"], lw=1, label=r"mean field, $e^{-1}/(\tau e^{-\tau})$")
    ax.plot([], [], "H--", color="#333", ms=3, lw=0.9, label="dashed: HORAE-R")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"controller target $G^\star/S=\tau$"); ax.set_ylabel(r"VWAoI relative to $\tau=1$"); ax.set_title("Other operating points", pad=4)
    plain_log(ax, "both", xt=[0.25, 0.5, 1, 2, 3], yt=[0.7, 1, 1.5, 2, 3])
    fig.legend(h, l, loc="lower left", ncol=4, bbox_to_anchor=(0.03, 0.0), columnspacing=1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=1, handlelength=1.6, labelspacing=0.3)
    save(fig, "fig_load")


def fig_thresh():
    t = A.load("thresh")
    vw = A.table(t, "vwaoi"); f1 = A.table(t, "alert_f1"); nr = A.table(t, "track_nrmse")
    lams = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    fig.subplots_adjust(wspace=0.3, bottom=0.34, top=0.9, left=0.09, right=0.98)
    COL = (("intel_alert", "Intel-Lab, alert", C["blue"]), ("beijing_alert", "Beijing, alert", C["red"]), ("beijing_track", "Beijing, tracking", C["green"]), ("beijing_track_w", "Beijing, tracking, $w_i$ mixed", C["purple"]))
    for cfg, lab, c in COL:
        for eps, ls, mk in ((1.0, "-", "o"), (0.5, "--", "s")):
            y = [np.mean(list(vw[(f"{cfg}:{l}:0.25:{eps}", "HORAE")].values())) for l in lams]
            y0 = np.mean(list(vw[(f"{cfg}:2.0:0.25:1.0", "HORAE")].values()))
            axes[0].plot(lams, np.array(y)/y0, ls, marker=mk, color=c, ms=2.8, label=lab if eps == 1.0 else None)
            q = f1 if "alert" in cfg else nr
            axes[1].plot(lams, [np.mean(list(q[(f"{cfg}:{l}:0.25:{eps}", "HORAE")].values())) for l in lams], ls, marker=mk, color=c, ms=2.8)
    axes[0].axvline(2.0, color="#777", lw=0.6, ls="--"); axes[1].axvline(2.0, color="#777", lw=0.6, ls="--")
    axes[0].axhline(1.0, color="#333", lw=0.6, ls=":")
    axes[0].set_ylabel(r"VWAoI relative to $\lambda=2$, $\nu=1$"); axes[1].set_ylabel("decision quality")
    for ax, t_ in zip(axes, ("(a) staleness", "(b) alert $F_1$ or tracking NRMSE")):
        ax.set_xlabel(r"threshold multiple $\lambda$"); ax.set_title(t_, pad=4)
    H = axes[0].get_legend_handles_labels()
    H = list(H[0]) + [Line2D([], [], color="#333", marker="o", ms=2.8, label=r"$\nu=1$"), Line2D([], [], color="#333", ls="--", marker="s", ms=2.8, label=r"$\nu=1/2$")]
    fig.legend(handles=H, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), columnspacing=1.0)
    save(fig, "fig_thresh")


def fig_index():
    vw = A.table(A.load("exponents"), "vwaoi")
    As = [0.0, 0.5, 1.0, 1.5, 2.0]; Bs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.55))
    fig.subplots_adjust(wspace=0.2, bottom=0.17, top=0.9, left=0.06, right=0.99)
    cmap = plt.get_cmap("PuOr_r")
    for ax, (cfg, title, _, _) in zip(axes, CFG):
        base = np.mean(list(vw[(f"{cfg}:1.0:1.0", "IDX")].values()))
        for i, a in enumerate(As):
            for j, b in enumerate(Bs):
                v = np.mean(list(vw[(f"{cfg}:{a}:{b}", "IDX")].values()))/base
                col = cmap(np.clip(0.5 + 1.1*np.log(v), 0, 1))
                ax.add_patch(Rectangle((j, i), 1, 1, fc=col, ec="white", lw=0.5))
                lum = 0.299*col[0] + 0.587*col[1] + 0.114*col[2]
                ax.text(j + 0.5, i + 0.5, ("%.2f" % v) if v < 10 else ("%.0f" % v), ha="center", va="center", fontsize=6.6, color="white" if lum < 0.6 else "black")
        ax.add_patch(Rectangle((2, 2), 1, 1, fc="none", ec=C["black"], lw=1.4))
        for b_, a_ in ((1.0, 1.0), (2.0, 1.5), (3.0, 2.0)):
            pass
        ax.set_xlim(0, len(Bs)); ax.set_ylim(0, len(As))
        ax.set_xticks(np.arange(len(Bs)) + 0.5); ax.set_xticklabels([str(b) for b in Bs])
        ax.set_yticks(np.arange(len(As)) + 0.5); ax.set_yticklabels([str(a) for a in As])
        ax.set_xlabel(r"age exponent $\gamma$"); ax.grid(False); ax.set_title(title, pad=4)
    axes[0].set_ylabel(r"value exponent $\alpha$")
    save(fig, "fig_index")


def _load_map():
    
    lm = json.load(open(os.path.join(HERE, "results", "load_map.json"), encoding="utf8"))
    return {int(k): v for k, v in lm["scale"].items()}


def fig_overload():
    recs = A.load("overload")
    rows = [r for r in recs if not r["tag"].startswith("imp")]
    def v(r):
        if r["pol"] == "HORAE" and r["kw"]:
            return "HORAE-L"
        return r["pol"]
    vw = A.table(rows, "vwaoi", v); gs = A.table(rows, "G_over_S", v); f1 = A.table(rows, "alert_f1", v)
    act = _load_map()
    Ss = [4, 11, 22, 44]; Ns = [96, 240, 504, 1008, 2040]
    MK = {4: "o", 11: "s", 22: "^", 44: "D"}
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9))
    fig.subplots_adjust(wspace=0.42, bottom=0.36, top=0.91, left=0.07, right=0.98)

    def curve(S, num, den, table=vw):
        xs, ys = [], []
        for N in Ns:
            k1, k2 = (f"S{S}:N{N}", num), (f"S{S}:N{N}", den)
            if k1 in table and k2 in table:
                xs.append(act[N]/S); ys.append(A.paired(table[k2], table[k1], B=200)["ratio"])
        return xs, ys
    ax = axes[0]
    for S in Ss:
        x, y = curve(S, "ALOHA", "HORAE"); ax.plot(x, y, "-", marker=MK[S], color=PC["HORAE"], ms=3.2)
        x, y = curve(S, "ALOHA", "HORAE-R"); ax.plot(x, y, "-", marker=MK[S], color=PC["HORAE-R"], ms=3.2, mfc="none")
        x, y = curve(S, "ALOHA", "HORAE-L"); ax.plot(x, y, "--", marker=MK[S], color=C["grey"], ms=3.2, zorder=1)
    ax.axhline(1, color="#333", lw=0.7, ls=":"); ax.set_xscale("log"); ax.set_yscale("log")
    plain_log(ax, "both", xt=[0.1, 1, 10], yt=[0.5, 1, 2, 4, 8, 16])
    ax.set_xlabel("offered load"); ax.set_ylabel("S-ALOHA VWAoI\n/ VWAoI of rule"); ax.set_title(r"(a) over slotted ALOHA", pad=4)
    ax = axes[1]
    for S in Ss:
        x, y = curve(S, "ALOHA-ADP", "HORAE"); ax.plot(x, y, "-", marker=MK[S], color=PC["HORAE"], ms=3.2)
        x, y = curve(S, "ALOHA-ADP", "HORAE-R"); ax.plot(x, y, "-", marker=MK[S], color=PC["HORAE-R"], ms=3.2, mfc="none")
    ax.axhline(1, color="#333", lw=0.7, ls=":"); ax.set_xscale("log"); ax.set_ylim(0.6, None)
    plain_log(ax, "x", xt=[0.1, 1, 10])
    ax.set_xlabel("offered load"); ax.set_ylabel("Ad-ALOHA VWAoI\n/ VWAoI of rule"); ax.set_title("(b) over adaptive ALOHA", pad=4)
    ax = axes[2]
    for S in Ss:
        xs = [act[N]/S for N in Ns]
        ax.plot(xs, [np.mean(list(gs[(f"S{S}:N{N}", "HORAE")].values())) for N in Ns], "-", marker=MK[S], color=PC["HORAE"], ms=3.2)
        ax.plot(xs, [np.mean(list(gs[(f"S{S}:N{N}", "HORAE-R")].values())) for N in Ns], "-", marker=MK[S], color=PC["HORAE-R"], ms=3.2, mfc="none")
        ax.plot(xs, [np.mean(list(gs[(f"S{S}:N{N}", "HORAE-L")].values())) for N in Ns], "--", marker=MK[S], color=C["grey"], ms=3.2, zorder=1)
    ax.axhline(1, color="#333", lw=0.7, ls=":"); ax.set_xscale("log"); ax.set_yscale("log")
    plain_log(ax, "both", xt=[0.1, 1, 10], yt=[0.1, 1, 10])
    ax.set_xlabel("offered load"); ax.set_ylabel("attempt rate $G/S$"); ax.set_title("(c) what the controller holds", pad=4)
    H = [Line2D([], [], color=PC["HORAE"], label=r"HORAE, $\beta\geq10^{-12}$"), Line2D([], [], color=C["grey"], ls="--", label=r"HORAE, $\beta\geq10^{-4}$"),
         Line2D([], [], color=PC["HORAE-R"], marker="o", mfc="none", ms=3.2, label="HORAE-R")] + \
        [Line2D([], [], color="#333", marker=MK[S], ls="none", ms=3.5, label=f"$S={S}$") for S in Ss]
    fig.legend(handles=H, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.02), columnspacing=1.0, handletextpad=0.4)
    save(fig, "fig_overload")


def fig_sync():
    recs = [r for r in A.load("synchrony")]
    vw = A.table(recs, "vwaoi"); f1 = A.table(recs, "alert_f1")
    lv = sorted({float(k[0]) for k in vw})
    pols = ["ALOHA-ADP", "ADP-G", "ALOHA", "ATA", "AGE", "HORAE", "HORAE-G", "HORAE-R"]
    MK = dict(zip(pols, ["p", "P", ">", "D", "^", "o", "s", "H"]))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    fig.subplots_adjust(wspace=0.27, bottom=0.3, top=0.91, left=0.08, right=0.98)
    for p in pols:
        st = dict(ls="--", mfc="none") if p == "AGE" else {}
        axes[0].plot(lv, [np.mean(list(vw[(str(s), p)].values())) for s in lv], marker=MK[p], color=PC[p], label=PN[p], ms=3.2, **st)
        axes[1].plot(lv, [np.mean(list(f1[(str(s), p)].values())) for s in lv], marker=MK[p], color=PC[p], ms=3.2, **st)
    axes[0].set_yscale("log"); axes[0].set_ylabel("value-weighted AoI"); axes[1].set_ylabel("alert $F_1$")
    for ax, t in zip(axes, ("(a) staleness", "(b) decisions")):
        ax.set_xlabel("phase-keeping probability $s$"); ax.set_title(t, pad=4)
    fig.legend(loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.01), columnspacing=0.9)
    plain_log(axes[0], "y", yt=[30, 50, 100, 200, 400])
    save(fig, "fig_sync")


def fig_fields():
    recs = A.load("fields")
    if not recs:
        return
    vw = A.table(recs, "vwaoi")
    names = [n for n in ("aqs", "o3", "pems") if any(k[0].startswith(n + ":") for k in vw)]
    LAB = {"aqs": "EPA PM$_{2.5}$", "o3": "EPA ozone", "pems": "PEMS occupancy"}
    fig, axes = plt.subplots(1, len(names), figsize=(7.5, 2.7), sharey=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(wspace=0.1, bottom=0.31, top=0.91, left=0.07, right=0.98)
    loads = [0.5, 1.0, 2.0, 4.0]
    for ax, n in zip(axes, names):
        for kind, ls, mk in (("alert", "-", "o"), ("track", "--", "s")):
            for den, c in (("HORAE", PC["HORAE"]), ("HORAE-G", PC["HORAE-G"]), ("HORAE-R", PC["HORAE-R"])):
                num = "ADP-G" if den == "HORAE-G" else "ALOHA-ADP"
                y = [A.paired(vw[(f"{n}:{kind}:{l}", den)], vw[(f"{n}:{kind}:{l}", num)], B=200)["ratio"] for l in loads]
                ax.plot(loads, y, ls, marker=mk, color=c, ms=3.2)
            y = [A.paired(vw[(f"{n}:{kind}:{l}", "HORAE")], vw[(f"{n}:{kind}:{l}", "ALOHA")], B=200)["ratio"] for l in loads]
            ax.plot(loads, y, ls, marker=mk, color=PC["ALOHA"], ms=3.2)
            sh = [(l, f"{n}:{kind}:{l}:shifted") for l in (1.0, 2.0) if (f"{n}:{kind}:{l}:shifted", "HORAE") in vw]
            if sh:
                ax.plot([l*(0.88 if kind == "alert" else 1.14) for l, _ in sh], [A.paired(vw[(t, "HORAE")], vw[(t, "ALOHA-ADP")], B=200)["ratio"] for _, t in sh], marker="*", ls="none",
                        color=C["black"] if kind == "alert" else C["grey"], mec="white", mew=0.4, ms=7.5, zorder=6)
                ax.plot([l*(0.88 if kind == "alert" else 1.14) for l, _ in sh], [A.paired(vw[(t, "HORAE-R")], vw[(t, "ALOHA-ADP")], B=200)["ratio"] for _, t in sh], marker="*", ls="none",
                        color=C["black"] if kind == "alert" else C["grey"], mfc="white", ms=7.5, zorder=7)
        ax.axhline(1, color="#333", lw=0.7, ls=":"); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(LAB[n], pad=4); ax.set_xlabel("offered load")
        plain_log(ax, "both", xt=[0.5, 1, 2, 4], yt=[0.7, 1, 1.5, 2, 3, 4, 6])
    axes[0].set_ylabel("VWAoI of comparator\n/ VWAoI of rule")
    H = [Line2D([], [], color=PC["HORAE-R"], label="Ad-ALOHA / HORAE-R"), Line2D([], [], color=PC["HORAE"], label="Ad-ALOHA / HORAE"), Line2D([], [], color=PC["HORAE-G"], label="ADP-G / HORAE-G"), Line2D([], [], color=PC["ALOHA"], label="S-ALOHA / HORAE"),
         Line2D([], [], color="#333", marker="o", ms=3, label="alert"), Line2D([], [], color="#333", ls="--", marker="s", ms=3, label="tracking"),
         Line2D([], [], color=C["black"], marker="*", ls="none", ms=6, label="12 sensors shifted, alert"), Line2D([], [], color=C["grey"], marker="*", ls="none", ms=6, label="12 sensors shifted, tracking"),
         Line2D([], [], color=C["black"], marker="*", mfc="none", ls="none", ms=6, label="open star: HORAE-R")]
    fig.legend(handles=H, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), columnspacing=1.0)
    save(fig, "fig_fields")


def fig_feedback():
    vw = A.table(A.load("feedback"), "vwaoi")
    v2 = A.table(A.load("feedback2"), "vwaoi")
    panels = [("loss", "beacon lost per device", "loss probability", [0.01, 0.05, 0.1, 0.2, 0.3, 0.5], 0.0, vw),
              ("delay", "feedback delay", "delay (slots)", [1, 2, 5, 10, 20], 0, vw),
              ("incons", "inconsistent idle count", r"error sd / $S$", [0.05, 0.1, 0.2, 0.3], 0.0, vw),
              ("duty", "beacon duty cycle", "slots between listens", [2, 5, 10, 20, 50, 100], 1, vw),
              ("guard", "clock offsets, 10% guard", "offset range (slots)", [0.05, 0.1, 0.2, 0.25, 0.5, 1.0], 0.0, v2)]
    pols = ["ALOHA-ADP", "AGE", "HORAE-G", "HORAE", "HORAE-R"]
    MK = {"ALOHA-ADP": "p", "AGE": "^", "HORAE-G": "s", "HORAE": "o", "HORAE-R": "H"}
    fig, axes = plt.subplots(2, 5, figsize=(7.5, 3.7), sharey="row")
    fig.subplots_adjust(wspace=0.12, hspace=0.62, bottom=0.2, top=0.93, left=0.075, right=0.99)
    for r_, (cfg, lab) in enumerate((("beijing_alert", "Beijing alert"), ("beijing_track", "Beijing tracking"))):
        base = np.mean(list(vw[(f"{cfg}:ideal", "HORAE")].values()))
        for ax, (key, title, xl, vals, x0, tb) in zip(axes[r_], panels):
            for p in pols:
                xs, ys = [], []
                if (f"{cfg}:ideal", p) in vw:
                    xs.append(x0); ys.append(np.mean(list(vw[(f"{cfg}:ideal", p)].values()))/base)
                for v_ in vals:
                    k = (f"{cfg}:{key}:{v_}", p)
                    if k in tb:
                        xs.append(v_); ys.append(np.mean(list(tb[k].values()))/base)
                if len(xs) > 1:
                    st = dict(ls="--", mfc="none") if p == "AGE" else {}
                    ax.plot(xs, ys, marker=MK[p], color=PC[p], ms=3.0 if p != "ALOHA-ADP" else 2.6, lw=1.1 if p != "ALOHA-ADP" else 0.8, label=PN[p], **st)
            ax.set_title("(%s) %s" % ("abcdefghij"[5*r_ + list(zip(axes[r_], panels)).index((ax, (key, title, xl, vals, x0, tb)))], title), pad=3, fontsize=7.2); ax.set_xlabel(xl, fontsize=7.2)
            ax.axhline(1, color="#333", lw=0.6, ls=":")
            if key == "duty":
                ax.set_xscale("log"); plain_log(ax, "x", xt=[1, 10, 100])
        axes[r_, 0].set_ylabel(lab + "\nVWAoI / ideal HORAE", fontsize=7.6); axes[r_, 0].set_yscale("log")
        plain_log(axes[r_, 0], "y", yt=[1, 3, 10, 30, 100])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_feedback")


def fig_sens():
    S = A.load("sensitivity"); vw = A.table(S, "vwaoi")
    V = A.load("value"); vv = A.table(V, "vwaoi")
    panels = [("v_floor", "floor $v_0$", (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5), True, 0.05),
              ("kappa", r"$\kappa$", (1.0, 2.0, 3.0, 5.0, 8.0), False, 3.0), ("boost", "bonus $b$", (0.0, 1.0, 2.0, 4.0, 8.0, 16.0), False, 4.0),
              ("rho_mul", r"$\rho$ multiplier", (0.25, 0.5, 1.0, 2.0, 4.0), True, 1.0), ("pct", "alert percentile", (80, 85, 90, 92, 95, 98), False, 92),
              ("eta", r"gain $\eta$", (0.02, 0.05, 0.1, 0.25, 0.5, 1.0), True, 0.25)]
    fig, axes = plt.subplots(2, 4, figsize=(7.5, 3.8))
    fig.subplots_adjust(wspace=0.32, hspace=0.62, bottom=0.2, top=0.95, left=0.085, right=0.98)
    axf = axes.ravel()
    for ax_, let_ in zip(axf, "abcdefgh"):
        ax_.set_title("(%s)" % let_, loc="left", pad=2, fontsize=7.6)
    COL = {"intel_alert": C["blue"], "beijing_alert": C["red"], "beijing_track": C["green"]}
    for ax, (key, xl, vals, lg, mark) in zip(axf, panels):
        for cfg, c in COL.items():
            xs, ys = [], []
            for v_ in vals:
                a, b = (f"{cfg}:{key}:{v_}", "HORAE"), (f"{cfg}:{key}:{v_}", "ALOHA-ADP")
                if a in vw and b in vw:
                    xs.append(v_); ys.append(A.paired(vw[a], vw[b], B=200)["ratio"])
            if xs:
                ax.plot(xs, ys, "o-", color=c, ms=2.8)
            xs, ys = [], []
            for v_ in vals:
                a, b = (f"{cfg}:{key}:{v_}", "HORAE-R"), (f"{cfg}:{key}:{v_}", "ALOHA-ADP")
                if a in vw and b in vw:
                    xs.append(v_); ys.append(A.paired(vw[a], vw[b], B=200)["ratio"])
            if xs:
                ax.plot(xs, ys, "H--", color=c, ms=2.6, lw=0.9)
        ax.axhline(1, color="#333", lw=0.6, ls=":"); ax.axvline(mark, color="#777", lw=0.6, ls="--")
        if lg:
            ax.set_xscale("log")
            plain_log(ax, "x", xt=[v_ for v_ in vals if (key != "v_floor" or v_ in (0.005, 0.05, 0.5)) and (key != "eta" or v_ in (0.02, 0.1, 0.5))])
        ax.set_xlabel(xl)
    fig.supylabel("margin over adaptive ALOHA", x=0.012, y=0.58, fontsize=8.5)
    # value errors: HORAE with a damaged score, scored on the true value, against the unaffected comparators
    for ax, (kind, xl, vals) in zip(axf[6:], (("noise", r"per-slot score error $\sigma_{\mathrm{e}}$", (0.25, 0.5, 1.0, 2.0)), ("bias", r"per-source score bias $\sigma_{\mathrm{e}}$", (0.25, 0.5, 1.0, 2.0)))):
        for cfg, c in COL.items():
            base = vv[(f"{cfg}:base", "ALOHA-ADP")]
            ys = [A.paired(vv[(f"{cfg}:true", "HORAE")], base, B=200)["ratio"]] + [A.paired(vv[(f"{cfg}:{kind}:{v_}", "HORAE")], base, B=200)["ratio"] for v_ in vals]
            ax.plot((0.0,) + vals, ys, "o-", color=c, ms=2.8)
            ys = [A.paired(vv[(f"{cfg}:true", "HORAE-R")], base, B=200)["ratio"]] + [A.paired(vv[(f"{cfg}:{kind}:{v_}", "HORAE-R")], base, B=200)["ratio"] for v_ in vals]
            ax.plot((0.0,) + vals, ys, "H--", color=c, ms=2.6, lw=0.9)
        ax.axhline(1, color="#333", lw=0.6, ls=":"); ax.set_xlabel(xl)
    H = [Line2D([], [], color=c, marker="o", ms=3, label=l) for c, l in ((C["blue"], "Intel-Lab, alert"), (C["red"], "Beijing, alert"), (C["green"], "Beijing, tracking"))] + \
        [Line2D([], [], color="#333", marker="o", ms=2.8, label="HORAE"), Line2D([], [], color="#333", marker="H", ls="--", ms=2.6, lw=0.9, label="HORAE-R"),
         Line2D([], [], color="#777", ls="--", lw=0.7, label="value used"), Line2D([], [], color="#333", ls=":", lw=0.7, label="parity")]
    fig.legend(handles=H, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0)
    save(fig, "fig_sens")


def fig_capture():
    recs = A.load("capture")
    vw = A.table(recs, "vwaoi"); f1 = A.table(recs, "alert_f1")
    tags = ["None", "9.0", "6.0", "3.0", "0.0"]
    labels = ["none", "9", "6", "3", "0"]
    pols = ["HORAE-R", "HORAE", "HORAE-G", "AGE", "ATA", "ALOHA", "ALOHA-ADP"]
    MK = dict(zip(pols, ["H", "o", "s", "^", "D", ">", "p"]))
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5))
    fig.subplots_adjust(wspace=0.27, bottom=0.31, top=0.9, left=0.09, right=0.98)
    for p in pols:
        x = np.arange(len(tags))
        axes[0].plot(x, [np.mean(list(vw[(t, p)].values())) for t in tags], marker=MK[p], color=PC[p], label=PN[p], ms=3.2)
        axes[1].plot(x, [np.mean(list(f1[(t, p)].values())) for t in tags], marker=MK[p], color=PC[p], ms=3.2)
    for ax, t in zip(axes, ("(a) staleness", "(b) decisions")):
        ax.set_xticks(np.arange(len(tags))); ax.set_xticklabels(labels); ax.set_xlabel("capture margin (dB)"); ax.set_title(t, pad=4)
    axes[0].set_yscale("log"); axes[0].set_ylabel("value-weighted AoI"); axes[1].set_ylabel("alert $F_1$")
    fig.legend(loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.01), columnspacing=0.9)
    plain_log(axes[0], "y", yt=[10, 20, 50, 100, 200])
    save(fig, "fig_capture")


SIZE = {"intel_alert": 62, "beijing_alert": 21, "beijing_track": 32}


def _ops():
    return EP.read_ops(os.path.join(HERE, "results", "device_loop.txt"))


def _ops_r():
    return EP.read_ops(os.path.join(HERE, "results", "device_loop_r.txt"))


def _mj(tag, p, T, prof="lora", slot=10.0, ops=None, tag0=None):
    n = np.mean(list(T["N"][(tag, p)].values()))*np.mean(list(T["slots"][(tag, p)].values()))
    a, l = EP.device_rates(p, np.mean(list(T["tx"][(tag, p)].values())), np.mean(list(T["succ"][(tag, p)].values())), np.mean(list(T["n_listen"][(tag, p)].values())), n)
    return 1e3*EP.slot_energy_J(prof, slot, a, l, ops=ops, **EP.bits(SIZE[tag0 or tag.split(":")[0]]))["total"]


def fig_energy():
    ops = _ops()
    KEYS = ("vwaoi", "tx", "succ", "n_listen", "N", "slots", "alert_f1", "track_nrmse")
    OP = {k: A.table(A.load("operating"), k) for k in KEYS}
    FB = {k: A.table(A.load("feedback"), k) for k in KEYS}
    MN = {k: A.table(A.load("main"), k, variant) for k in KEYS}
    TH = {k: A.table(A.load("thresh"), k) for k in KEYS}
    opr = _ops_r()
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 3.9))
    fig.subplots_adjust(wspace=0.3, hspace=0.5, bottom=0.23, top=0.94, left=0.08, right=0.98)
    taus = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    duties = [2, 5, 10, 20, 50, 100]
    for j, (cfg, title, q, ql) in enumerate(CFG):
        ax, ay = axes[0, j], axes[1, j]
        Q = q
        for pol, c, mk in (("ALOHA-ADP", PC["ALOHA-ADP"], "p"), ("AGE", PC["AGE"], "^"), ("HORAE", PC["HORAE"], "o")):
            xs = [_mj(f"{cfg}:{t}", pol, OP, ops=ops) for t in taus]
            ax.plot(xs, [np.mean(list(OP["vwaoi"][(f"{cfg}:{t}", pol)].values())) for t in taus], marker=mk, color=c, ms=3, lw=1)
            ay.plot(xs, [np.mean(list(OP[Q][(f"{cfg}:{t}", pol)].values())) for t in taus], marker=mk, color=c, ms=3, lw=1)
        xr = [_mj(f"energy:{cfg}:{t}:1", "HORAE-R", TH, ops=opr, tag0=cfg) for t in (0.25, 0.5, 0.75, 1.0)]
        ax.plot(xr, [np.mean(list(TH["vwaoi"][(f"energy:{cfg}:{t}:1", "HORAE-R")].values())) for t in (0.25, 0.5, 0.75, 1.0)], "H-", color=PC["HORAE-R"], ms=3.2, lw=1.2)
        ay.plot(xr, [np.mean(list(TH[Q][(f"energy:{cfg}:{t}:1", "HORAE-R")].values())) for t in (0.25, 0.5, 0.75, 1.0)], "H-", color=PC["HORAE-R"], ms=3.2, lw=1.2)
        xr = [_mj(f"energy:{cfg}:1.0:{d}", "HORAE-R", TH, ops=opr, tag0=cfg) for d in (1, 5, 20, 100)]
        ax.plot(xr, [np.mean(list(TH["vwaoi"][(f"energy:{cfg}:1.0:{d}", "HORAE-R")].values())) for d in (1, 5, 20, 100)], "H:", color=PC["HORAE-R"], ms=3.2, lw=1.2)
        ay.plot(xr, [np.mean(list(TH[Q][(f"energy:{cfg}:1.0:{d}", "HORAE-R")].values())) for d in (1, 5, 20, 100)], "H:", color=PC["HORAE-R"], ms=3.2, lw=1.2)
        xs = [_mj(f"{cfg}:ideal", "HORAE", FB, ops=ops)] + [_mj(f"{cfg}:duty:{d}", "HORAE", FB, ops=ops) for d in duties]
        ys = [np.mean(list(FB["vwaoi"][(f"{cfg}:ideal", "HORAE")].values()))] + [np.mean(list(FB["vwaoi"][(f"{cfg}:duty:{d}", "HORAE")].values())) for d in duties]
        zs = [np.mean(list(FB[Q][(f"{cfg}:ideal", "HORAE")].values()))] + [np.mean(list(FB[Q][(f"{cfg}:duty:{d}", "HORAE")].values())) for d in duties]
        ax.plot(xs, ys, "s--", color=PC["HORAE-G"], ms=3, lw=1); ay.plot(xs, zs, "s--", color=PC["HORAE-G"], ms=3, lw=1)
        for pol, mk in (("ALOHA", ">"), ("ATA-T", "D"), ("GOAL", "X"), ("AOII", "P")):
            y_ = np.mean(list(MN["vwaoi"][(cfg, pol)].values()))
            if cfg == "intel_alert" and y_ > 28:
                if pol != "AOII":
                    ax.annotate(("GOAL, AoII" if pol == "GOAL" else PN[pol]) + " at %.0f" % y_, xy=(_mj(cfg, pol, MN, ops=ops), 28), xytext=(_mj(cfg, pol, MN, ops=ops) + (0.03 if pol == "ALOHA" else -0.12), 26.6), ha="left", fontsize=6.8,
                                arrowprops=dict(arrowstyle="-|>", color="#333", lw=0.9, mutation_scale=7, shrinkA=1, shrinkB=0), annotation_clip=False)
                # their decision quality lies inside the lower panel's range
                ay.plot(_mj(cfg, pol, MN, ops=ops), np.mean(list(MN[Q][(cfg, pol)].values())), mk, color=PC[pol], ms=5, mec="#222", mew=0.3)
                continue
            ax.plot(_mj(cfg, pol, MN, ops=ops), y_, mk, color=PC[pol], ms=5, mec="#222", mew=0.3)
            ay.plot(_mj(cfg, pol, MN, ops=ops), np.mean(list(MN[Q][(cfg, pol)].values())), mk, color=PC[pol], ms=5, mec="#222", mew=0.3)
        ax.set_yscale("log"); ax.set_title(title, pad=3); ay.set_xlabel("device energy per slot (mJ)")
        ay.set_ylabel(ql); ax.set_xlabel("device energy per slot (mJ)")
        plain_log(ax, "y", yt={"intel_alert": [23, 24, 25, 26, 27, 28], "beijing_alert": [20, 40, 60, 100], "beijing_track": [20, 30, 40, 60]}[cfg])
        if cfg == "intel_alert":
            ax.set_ylim(22.6, 28)
            ay.set_ylim(0.93, 1.005)
        ax.set_ylabel("VWAoI")
    H = [Line2D([], [], color=PC["HORAE-R"], marker="H", ms=3, label=r"HORAE-R, $\tau$ swept"), Line2D([], [], color=PC["HORAE-R"], marker="H", ls=":", ms=3, label="HORAE-R, beacon duty swept"),
         Line2D([], [], color=PC["HORAE"], marker="o", ms=3, label=r"HORAE, target $\tau$ swept"), Line2D([], [], color=PC["AGE"], marker="^", ms=3, label=r"Age-only, $\tau$ swept"),
         Line2D([], [], color=PC["ALOHA-ADP"], marker="p", ms=3, label=r"Ad-ALOHA, $\tau$ swept"), Line2D([], [], color=PC["HORAE-G"], marker="s", ls="--", ms=3, label="HORAE, beacon duty swept")] + \
        [Line2D([], [], color=PC[p_], marker=m_, ls="none", ms=5, mec="#222", mew=0.3, label=PN[p_]) for p_, m_ in (("ALOHA", ">"), ("ATA-T", "D"), ("GOAL", "X"), ("AOII", "P"))]
    fig.legend(handles=H, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04), columnspacing=0.9)
    save(fig, "fig_energy")


if __name__ == "__main__":
    which = sys.argv[1:] or ["main", "frontier", "thresh", "load", "index", "overload", "sync", "fields", "feedback", "sens", "capture", "energy"]
    for w in which:
        globals()["fig_" + w]()
