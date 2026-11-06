
import json, math, os, re, subprocess, sys
from math import e, pi
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
sys.path.insert(0, CODE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import analyze as A
import energy_profiles as EP
from run_studies import tuned_settings, CONFIGS

RES = os.path.join(CODE, "results")
R = json.load(open(os.path.join(RES, "results.json"), encoding="utf8"))
CK = json.load(open(os.path.join(RES, "analysis_checks.json"), encoding="utf8"))
OPS = EP.read_ops(os.path.join(RES, "device_loop.txt"))
M = {}
TB = {}


def put(name, value, fmt="{:.2f}"):
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(name)
    if name in M:
        raise ValueError("duplicate macro " + name)
    M[name] = value if isinstance(value, str) else fmt.format(value)
    return M[name]


def commas(n):
    return "{:,}".format(int(round(n))).replace(",", "{,}")


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


main = A.load("main")
VW = A.table(main, "vwaoi", variant)
F1 = A.table(main, "alert_f1", variant)
NR = A.table(main, "track_nrmse", variant)
EN = A.table(main, "energy_J", variant)
TAG = {"Intel": "intel_alert", "BeijAl": "beijing_alert", "BeijTr": "beijing_track", "BeijAlW": "beijing_alert_w", "BeijTrW": "beijing_track_w"}
PM = {"HORAE": "Horae", "HORAE-G": "HoraeG", "HORAE-R": "HoraeR", "AGE-R": "AgeR", "AGE-RE": "AgeRE", "AGE": "Age", "VOI": "Voi", "ATA": "Ata", "ATA-T": "AtaT", "LTP": "Ltp", "LTP-T": "LtpT", "ALOHA": "Aloha",
      "ALOHA-OPT": "Opt", "ALOHA-ADP": "Adp", "ADP-G": "AdpG", "AOII": "Aoii", "GOAL": "Goal", "TDMA": "Tdma", "ACS": "Acs", "RG": "Rg", "MAXWEIGHT": "Oracle", "MWQ": "Mwq", "IMP": "Imp"}


def mean(t, tag, p):
    return float(np.mean(list(t[(tag, p)].values())))


put("nRepMain", len(VW[("beijing_alert", "HORAE")]), "{:d}")
put("nRepSweep", 10, "{:d}")
put("nRepFive", 5, "{:d}")
for t, cfg in TAG.items():
    for p, pm in PM.items():
        if (cfg, p) in VW:
            put(f"xvw{t}{pm}", mean(VW, cfg, p))
            tbl = F1 if (cfg, p) in F1 else NR
            put(f"xdq{t}{pm}", mean(tbl, cfg, p), "{:.3f}")
    for p, pm in PM.items():
        if p != "HORAE" and (cfg, p) in VW:
            s = A.paired(VW[(cfg, "HORAE")], VW[(cfg, p)])
            put(f"r{t}{pm}", s["ratio"]); put(f"r{t}{pm}Lo", s["lo"]); put(f"r{t}{pm}Hi", s["hi"])


rows = []
HEAD = [("Intel-Lab alert", "intel_alert"), ("Beijing alert", "beijing_alert"), ("Beijing tracking", "beijing_track"),
        ("Beijing alert, $w_i$ mixed", "beijing_alert_w"), ("Beijing tracking, $w_i$ mixed", "beijing_track_w")]
STEPS = [("ALOHA", "ALOHA-ADP", "control"), ("ALOHA-ADP", "AGE", "age"), ("AGE", "HORAE", "value"), ("ALOHA-ADP", "HORAE", "ordering"), ("ALOHA", "HORAE", "total")]


def pfmt(p):
    if p < 1e-3:
        return "$<$0.001"
    return "%.3f" % p


pv = []; pvt = []
for lab, cfg in HEAD:
    cells = []
    for hi, lo, nm in STEPS:
        s = A.paired(VW[(cfg, lo)], VW[(cfg, hi)])
        cells.append(s)
        pv.append(s["p_w"]); pvt.append(s["p_t"])
    rows.append((lab, cfg, cells))
adj = A.holm(pv)
adj_i = iter(adj)
body = []
for lab, cfg, cells in rows:
    tx = [lab]
    for s in cells:
        a = next(adj_i)
        star = "" if a < 0.05 else "$^{\\dagger}$"
        tx.append("%.2f [%.2f, %.2f]%s" % (s["ratio"], s["lo"], s["hi"], star))
    body.append(" & ".join(tx) + r" \\")
TB["decomp"] = "\n".join(body)
put("decompNotSig", int(sum(1 for a in adj if a >= 0.05)), "{:d}")
put("decompTotal", len(adj), "{:d}")
put("decompNotSigT", int(sum(1 for a in A.holm(pvt) if a >= 0.05)), "{:d}")
_wm = "{:.1e}".format(2.0*0.5**len(VW[("beijing_alert", "HORAE")])).split("e")
put("wilMinP", "%s\\times10^{%d}" % (_wm[0], int(_wm[1])))


for t, cfg in TAG.items():
    if (cfg, "ALOHA-OPT") in VW:
        put(f"mult{t}", TUNED[(cfg, "opt")][1][0])

G = [A.paired(VW[(TAG[t], "HORAE")], VW[(TAG[t], "ALOHA")]) for t in ("Intel", "BeijAl", "BeijTr")]
put("absGainLo", min(s["ratio"] for s in G), "{:.1f}"); put("absGainHi", max(s["ratio"] for s in G), "{:.1f}")

# ---- main table
PTAB = [("HORAE-R", "HORAE-R"), ("HORAE", "HORAE"), ("HORAE-G", "HORAE-G"), ("AGE-R", "Age-R"), ("AGE", "Age-only"), ("VOI", "Value-only"), ("ATA", "ATA"), ("ATA-T", "ATA-T"), ("AOII", "AoII"), ("GOAL", "GOAL"),
        ("LTP", "LTP"), ("LTP-T", "LTP-T"), ("ALOHA", "Slotted ALOHA"), ("ALOHA-OPT", "Tuned ALOHA"), ("ALOHA-ADP", "Adaptive ALOHA"), ("ADP-G", "ADP-G"),
        ("TDMA", "TDMA (scheduled)"), ("ACS", "ACS (centralized)"), ("RG", "RG (request-grant)"), ("MAXWEIGHT", "Max-weight (genie)"), ("MWQ", "MW-Q (genie)")]
cfgs3 = ["intel_alert", "beijing_alert", "beijing_track"]
DEC = ["HORAE-R", "AGE-R", "HORAE", "HORAE-G", "AGE", "VOI", "ATA", "ATA-T", "AOII", "GOAL", "LTP", "LTP-T", "ALOHA", "ALOHA-OPT", "ALOHA-ADP", "ADP-G"]
lines = []
for p, lab in PTAB:
    if p == "TDMA":
        lines.append(r"\midrule")
    cells = [lab]
    for cfg in cfgs3:
        m, c = A.mean_ci(VW[(cfg, p)].values())
        best = p in DEC and "%.2f" % m == "%.2f" % min(mean(VW, cfg, q) for q in DEC)
        s = "%.2f $\\pm$ %.2f" % (m, c)
        cells.append("\\textbf{%s}" % s if best else s)
    for cfg in cfgs3:
        t = F1 if cfg != "beijing_track" else NR
        m, c = A.mean_ci(t[(cfg, p)].values())
        vals = [mean(t, cfg, q) for q in DEC]
        target = max(vals) if cfg != "beijing_track" else min(vals)
        best = p in DEC and "%.3f" % m == "%.3f" % target
        s = "%.3f" % m
        cells.append("\\textbf{%s}" % s if best else s)
    lines.append(" & ".join(cells) + r" \\")
TB["main"] = "\n".join(lines)


lines = []
for p, lab in PTAB:
    if (p == "TDMA"):
        lines.append(r"\midrule")
    cells = [lab]
    for cfg in ("beijing_alert_w", "beijing_track_w"):
        m, c = A.mean_ci(VW[(cfg, p)].values())
        cells.append("%.2f" % m)
    lines.append(" & ".join(cells) + r" \\")
TB["importance"] = "\n".join(lines)



V = A.table(A.load("variance"), "vwaoi")
for t, cfg in (("Intel", "intel_alert"), ("BeijAl", "beijing_alert"), ("BeijTr", "beijing_track")):
    a = A.paired(V[(f"{cfg}:chan", "HORAE")], V[(f"{cfg}:chan", "ALOHA-ADP")])
    b = A.paired(V[(f"{cfg}:pop", "HORAE")], V[(f"{cfg}:pop", "ALOHA-ADP")])
    put(f"varChan{t}", a["ratio"]); put(f"varPop{t}", b["ratio"]); put(f"varPop{t}Lo", b["lo"]); put(f"varPop{t}Hi", b["hi"])
    put(f"sdChan{t}", float(np.std(list(V[(f"{cfg}:chan", "HORAE")].values()), ddof=1)))
    put(f"sdPop{t}", float(np.std(list(V[(f"{cfg}:pop", "HORAE")].values()), ddof=1)))


for t, cfg in (("Intel", "intel_alert"), ("BeijAl", "beijing_alert"), ("BeijTr", "beijing_track")):
    a, h, o = mean(VW, cfg, "ALOHA"), mean(VW, cfg, "HORAE"), mean(VW, cfg, "MAXWEIGHT")
    put(f"gap{t}", 100*(a - h)/(a - o), "{:.0f}")
    ad = mean(VW, cfg, "ALOHA-ADP")
    put(f"gapAdp{t}", 100*(a - ad)/(a - o), "{:.0f}")


put("packetBits", 200, "{:d}"); put("headerBits", 80, "{:d}")
put("dtIntel", 31, "{:d}")
put("TIntel", commas(1500)); put("TBeij", commas(6000)); put("TBeijDays", 6000/24, "{:.0f}")
put("TIntelSlots", commas(1499)); put("TBeijSlots", commas(5999)); put("THoursIntel", 1499*31/3600, "{:.1f}")
put("TOverload", commas(2999)); put("TSync", commas(3999)); put("warmSlots", 4, "{:d}")
put("corrN", commas(1000)); put("corrT", commas(2999))
put("betaCeilVal", 50, "{:d}"); put("idleCap", "0.999"); put("idleUpFactor", 1000**0.25, "{:.1f}")
put("ltpEtaDefault", "0.1"); put("ltpPenDefault", "0.5")
put("wrongRatio", 7/3, "{:.1f}"); put("vRange", "140"); put("floorSilence", "60")
put("domViolPct", CK["dominance"]["pct"], "{:.1f}")
put("burstS", CK["burst"]["S"], "{:d}"); put("burstStepS", CK["burst"]["step"], "{:.2f}"); put("burstSlots", CK["burst"]["slots"], "{:.0f}")
put("gThree", CK["g3"], "{:.3f}"); put("mfOneConst", CK["one_const"], "{:.2f}")
FP = CK["fixed_point"]
for s_, nm in (("1", "One"), ("2", "Two"), ("3", "Three"), ("4", "Four"), ("5", "Five"), ("8", "Eight"), ("10", "Ten")):
    put("fp" + nm, FP[s_])


q, a = OPS["inactive"], OPS["active"]
put("opsQuietBasic", q["basic"], "{:d}"); put("opsQuietTrans", q["transcendental"], "{:d}")
put("opsActiveBasic", a["basic"], "{:d}"); put("opsActiveTrans", a["transcendental"], "{:d}"); put("opsActiveRng", a["rng"], "{:d}")
put("stateBytes", OPS["state_bytes"], "{:d}"); put("configBytes", OPS["config_bytes"], "{:d}")
ca, cq = EP.cycles(a), EP.cycles(q)
put("cyclesActive", commas(ca)); put("cyclesQuiet", commas(cq))
put("usActive", 1e6*ca/EP.MCU_HZ, "{:.0f}"); put("usQuiet", 1e6*cq/EP.MCU_HZ, "{:.0f}")
put("uJCpu", 1e6*3.0*EP.MCU_RUN_A_PER_HZ*EP.MCU_HZ*ca/EP.MCU_HZ, "{:.1f}")
put("codeKB", "1"); put("mcuSleepUA", EP.MCU_SLEEP_A*1e6, "{:.1f}"); put("battKJ", EP.BATTERY_J/1e3, "{:.1f}"); put("battMah", "%d{,}%03d" % divmod(int(round(EP.BATTERY_AH*1000)), 1000))
put("profZigFrameMs", 1e3*EP.frame_s("zigbee"), "{:.2f}"); put("profZigBeaconMs", 1e3*EP.beacon_s("zigbee", 6), "{:.2f}")
put("profLoraFrameMs", 1e3*EP.frame_s("lora"), "{:.1f}"); put("profLoraBeaconMs", 1e3*EP.beacon_s("lora", 6), "{:.1f}")


def mm(t, k):
    return float(np.mean(list(t[k].values())))


def ratio(t, den, num):
    return A.paired(t[den], t[num], B=2000)["ratio"]



for t, cfg in TAG.items():
    put(f"ctrl{t}", ratio(VW, (cfg, "ALOHA-ADP"), (cfg, "ALOHA")))
    put(f"ageF{t}", ratio(VW, (cfg, "AGE"), (cfg, "ALOHA-ADP")))
    put(f"val{t}", ratio(VW, (cfg, "HORAE"), (cfg, "AGE")))
    Gm = A.table([r for r in main if r["tag"] == cfg], "G_over_S", variant)
    put(f"gos{t}Horae", mm(Gm, (cfg, "HORAE")))
    put(f"gos{t}Adp", mm(Gm, (cfg, "ALOHA-ADP")))
Gm = A.table(main, "G_over_S", variant)
put("ataGoS", mm(Gm, ("beijing_alert", "ATA")))
put("guardMainGain", 100*(1 - mm(VW, ("beijing_alert", "HORAE-G"))/mm(VW, ("beijing_alert", "HORAE"))), "{:.0f}")


put("popIntelN", 530, "{:d}"); put("popBeijN", 480, "{:d}")
put("segSlots", commas(1199))


raw = A.load("raw")
Rv = A.table(raw, "vwaoi"); Rf = A.table(raw, "alert_f1")
for nm, tag in (("rawIntel", "intel"), ("rawBOne", "beijing:S1"), ("rawBTwo", "beijing:S2"), ("rawBFour", "beijing:S4"), ("rawBThree", "beijing:S3")):
    for p, pm in (("HORAE", "Horae"), ("HORAE-G", "HoraeG"), ("ALOHA-ADP", "Adp"), ("ALOHA", "Aloha"), ("MAXWEIGHT", "Oracle")):
        put(f"x{nm}{pm}", mm(Rv, (tag, p)), "{:.1f}")
        put(f"x{nm}{pm}F", mm(Rf, (tag, p)), "{:.3f}")


seg = A.load("segments")
Sv = A.table(seg, "vwaoi"); Sf = A.table(seg, "alert_f1"); Sn = A.table(seg, "track_nrmse")
put("segSAlert", 5, "{:d}"); put("segSTrack", 16, "{:d}")
for p, pm in (("HORAE", "Horae"), ("ALOHA-ADP", "Adp"), ("ATA", "Ata"), ("ALOHA", "Aloha"), ("HORAE-G", "HoraeG")):
    put(f"segAlert{pm}", mm(Sv, ("alert", p)), "{:.1f}")
    put(f"segAlert{pm}F", mm(Sf, ("alert", p)), "{:.3f}")
put("segAlertRatio", ratio(Sv, ("alert", "HORAE"), ("alert", "ALOHA-ADP")))
put("segTrackRatio", ratio(Sv, ("track", "HORAE"), ("track", "ALOHA-ADP")))
put("segAlertAlohaRatio", ratio(Sv, ("alert", "HORAE"), ("alert", "ALOHA")))
put("segTrackAlohaRatio", ratio(Sv, ("track", "HORAE"), ("track", "ALOHA")))


op = A.load("operating")
Ov = A.table(op, "vwaoi"); Of = A.table(op, "alert_f1"); Oe = A.table(op, "energy_J")
put("opTrackOne", mm(Ov, ("beijing_track:1.0", "HORAE"))); put("opTrackHalf", mm(Ov, ("beijing_track:0.5", "HORAE"))); put("opTrackTwo", mm(Ov, ("beijing_track:2.0", "HORAE")))
put("opAlertOne", mm(Ov, ("beijing_alert:1.0", "HORAE"))); put("opAlertHalf", mm(Ov, ("beijing_alert:0.5", "HORAE")))
put("opAlertGain", mm(Ov, ("beijing_alert:1.0", "HORAE"))/mm(Ov, ("beijing_alert:0.5", "HORAE")))
put("opAlertEnergyPct", 100*mm(Oe, ("beijing_alert:0.5", "HORAE"))/mm(Oe, ("beijing_alert:1.0", "HORAE")), "{:.0f}")
put("opAlertOneF", mm(Of, ("beijing_alert:1.0", "HORAE")), "{:.3f}"); put("opAlertHalfF", mm(Of, ("beijing_alert:0.5", "HORAE")), "{:.3f}")
put("opIntelEnergyPct", 100*mm(Oe, ("intel_alert:0.25", "HORAE"))/mm(Oe, ("intel_alert:1.0", "HORAE")), "{:.0f}")
put("opIntelQuarter", mm(Ov, ("intel_alert:0.25", "HORAE"))); put("opIntelOne", mm(Ov, ("intel_alert:1.0", "HORAE")))
put("opSTenOne", mm(Ov, ("beijing_alert_S10:1.0", "HORAE"))); put("opSFiveOne", mm(Ov, ("beijing_alert_S5:1.0", "HORAE")))
put("opSTenHalf", mm(Ov, ("beijing_alert_S10:0.5", "HORAE"))); put("opSFiveHalf", mm(Ov, ("beijing_alert_S5:0.5", "HORAE")))


Ex = A.table(A.load("exponents"), "vwaoi"); Ef = A.table(A.load("exponents"), "alert_f1"); En = A.table(A.load("exponents"), "track_nrmse")


def idx(cfg, a, b):
    return mm(Ex, (f"{cfg}:{a}:{b}", "IDX"))/mm(Ex, (f"{cfg}:1.0:1.0", "IDX"))


put("idxTrackOneFiveOne", idx("beijing_track", 1.5, 1.0)); put("idxTrackTwoTwo", idx("beijing_track", 2.0, 2.0))
put("idxTrackTwo", idx("beijing_track", 1.5, 2.0)); put("idxTrackThree", idx("beijing_track", 2.0, 3.0))
put("idxPredTwo", CK["g3"]/(2/pi)); put("idxPredThree", CK["g4"]/(2/pi))
put("idxTrackThreeQ", mm(En, ("beijing_track:2.0:3.0", "IDX")), "{:.3f}")
put("idxAlertThree", idx("beijing_alert", 2.0, 3.0)); put("idxAlertShallow", idx("beijing_alert", 0.5, 0.5))
put("idxAlertVoi", idx("beijing_alert", 2.0, 0.0)); put("idxAlertVoiF", mm(Ef, ("beijing_alert:2.0:0.0", "IDX")), "{:.3f}")
_iv = [idx("intel_alert", a, b) for a in (0.0, 0.5, 1.0, 1.5, 2.0) for b in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)]
put("idxIntelLo", min(_iv)); put("idxIntelHi", max(_iv))


MFv = A.table(A.load("meanfield"), "vwaoi"); MFs = A.table(A.load("meanfield"), "succ"); MFg = A.table(A.load("meanfield"), "G_over_S")
MFslots = A.table(A.load("meanfield"), "slots")


def mf_constants(N, sig):
    rng = np.random.default_rng(5)
    c = np.exp(sig*rng.normal(size=N)); c = c/np.mean(np.sqrt(c))**2
    return c


def mf_pred(pol, ex, N, S, sig, Th):
    c = mf_constants(N, sig); sq = np.sum(np.sqrt(c))**2; mc = float(np.mean(c))
    lb = sq/(2*N*Th) + 0.5*mc
    if pol == "HORAE":
        return (2/pi)*sq/(N*Th) + 0.5*mc
    if pol == "AGE":
        return (2/pi)*c.sum()/Th + 0.5*mc
    if pol in ("ALOHA-ADP", "VOI"):
        return c.sum()/Th
    if pol == "MAXWEIGHT":
        return c.sum()/(2*S) + 0.5*mc
    if pol == "MWQ":
        return sq/(2*N*S) + 0.5*mc
    k = ex[1] + 1; s_ = ex[0]/k
    g = math.gamma(1 + 2/k)/(2*math.gamma(1 + 1/k)**2)
    return g*float(np.sum(c**(1 - s_))*np.sum(c**s_))/(N*Th) + 0.5*mc


def mf_row(pol, ex, N, S, sig):
    tag = f"{N}:{S}:{sig}:{ex[0]}:{ex[1]}"
    J = mm(MFv, (tag, pol))
    if pol in ("MAXWEIGHT", "MWQ"):
        Th = float(S)
    else:
        Th = mm(MFs, (tag, pol))/mm(MFslots, (tag, pol))
    return J, mf_pred(pol, ex, N, S, sig, Th), Th


rows = []
all_ratio = {}
for N, S in ((500, 10), (2000, 20), (2000, 50), (5000, 20)):
    for sig in (0.0, 0.5, 1.0, 1.5):
        for pol, ex in (("HORAE", (1.0, 1.0)), ("AGE", (1.0, 1.0)), ("VOI", (1.0, 1.0)), ("ALOHA-ADP", (1.0, 1.0)), ("MAXWEIGHT", (1.0, 1.0)), ("MWQ", (1.0, 1.0)), ("IDX", (1.5, 2.0)), ("IDX", (2.0, 3.0)), ("IDX", (1.0, 2.0)), ("IDX", (0.5, 1.0))):
            J, P, Th = mf_row(pol, ex, N, S, sig)
            all_ratio[(pol, ex, N, S, sig)] = J/P
LAB = [("HORAE", (1.0, 1.0), "HORAE, $h=cA$"), ("AGE", (1.0, 1.0), "Age-only, $h=A$"), ("ALOHA-ADP", (1.0, 1.0), "Adaptive ALOHA"), ("VOI", (1.0, 1.0), "Value-only, $h=c$"),
       ("IDX", (1.5, 2.0), "$h=c^{1.5}A^2$"), ("IDX", (2.0, 3.0), "$h=c^{2}A^3$"), ("IDX", (1.0, 2.0), "$h=cA^2$"), ("MAXWEIGHT", (1.0, 1.0), "Max-weight, grants $cA$"), ("MWQ", (1.0, 1.0), "MW-Q, grants $cA(A+\\frac32)$")]
N0, S0 = 2000, 20
lines = []
for pol, ex, lab in LAB:
    cells = [lab]
    for sig in (0.0, 1.0, 1.5):
        J, P, Th = mf_row(pol, ex, N0, S0, sig)
        cells.append("%.1f & %.1f" % (J, P))
    rr = [v for k, v in all_ratio.items() if k[0] == pol and k[1] == ex]
    cells.append("%.3f--%.3f" % (min(rr), max(rr)))
    lines.append(" & ".join(cells) + r" \\")
TB["mf"] = "\n".join(lines)
J, P, Th = mf_row("HORAE", (1.0, 1.0), N0, S0, 1.0)
put("mfLB", float(np.sum(np.sqrt(mf_constants(N0, 1.0)))**2/(2*N0*Th) + 0.5*np.mean(mf_constants(N0, 1.0))), "{:.1f}")
put("mfHorae", J, "{:.1f}")
put("mfRatioToLB", (J - 0.5*np.mean(mf_constants(N0, 1.0)))/(float(np.sum(np.sqrt(mf_constants(N0, 1.0)))**2/(2*N0*Th))))
_core = [v for k, v in all_ratio.items() if k[0] in ("HORAE", "AGE", "ALOHA-ADP", "IDX", "MAXWEIGHT", "MWQ")]
put("mfCoreLo", min(_core), "{:.3f}"); put("mfCoreHi", max(_core), "{:.3f}")
_voi = [v for k, v in all_ratio.items() if k[0] == "VOI"]
put("mfVoiLo", min(_voi), "{:.3f}"); put("mfVoiHi", max(_voi), "{:.3f}")
put("mfN", commas(N0)); put("mfS", S0, "{:d}")
_tau = {t: mm(MFv, (f"tau:{t}", "HORAE")) for t in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)}
put("mfTauBest", min(_tau, key=_tau.get), "{:g}")
put("mfTauHalf", _tau[0.5]/_tau[1.0]); put("mfTauTwo", _tau[2.0]/_tau[1.0])


E11 = R["E11"]
for i, l_ in enumerate("abc"):
    put(f"thFp{l_}", FP.get(str(E11["S"][i]), float("nan")))


ovr = A.load("overload")
def _ov(r):
    return "HORAE-L" if r["pol"] == "HORAE" and r["kw"] else r["pol"]
_rows = [r for r in ovr if not r["tag"].startswith("imp")]
Ovv = A.table(_rows, "vwaoi", _ov); Ovg = A.table(_rows, "G_over_S", _ov); Ovf = A.table(_rows, "alert_f1", _ov); Ovs = A.table(_rows, "succ", _ov); Ovb = A.table(_rows, "beta_at_floor", _ov); Ovj = A.table(_rows, "jain_deliv", _ov)

LM = json.load(open(os.path.join(RES, "load_map.json"), encoding="utf8"))
ACT = {int(k_): v_ for k_, v_ in LM["scale"].items()}
for k_, c_ in (("Intel", "intel_alert"), ("BeijAl", "beijing_alert"), ("BeijTr", "beijing_track")):
    put(f"load{k_}", LM["headline"][c_]["active"]/LM["headline"][c_]["S"], "{:.2f}")
put("actIntelPct", 100*LM["headline"]["intel_alert"]["active"]/LM["headline"]["intel_alert"]["N"], "{:.1f}")
put("actBeijPct", 100*LM["headline"]["beijing_alert"]["active"]/LM["headline"]["beijing_alert"]["N"], "{:.1f}")
put("rawIntelLoad", LM["raw"]["intel"]/10, "{:.2f}"); put("rawBeijOne", 1/LM["raw"]["beijing"], "{:.1f}"); put("rawBeijFour", 4/LM["raw"]["beijing"], "{:.1f}")
for k_, ds_ in (("Intel", "intel"), ("Beij", "beijing")):
    E_ = R["E2"][ds_]; i_ = E_["loads"].index(4)
    put(f"loadSweep{k_}Four", E_["by_pol"]["ALOHA-ADP"][i_]/E_["by_pol"]["HORAE"][i_])
    put(f"loadSweep{k_}Top", LM["headline"][ds_ + "_alert"]["active"]/E_["S"][i_], "{:.1f}")
_e = R["E2"]["beijing"]; _r = [a_/h_ for a_, h_ in zip(_e["by_pol"]["ALOHA"], _e["by_pol"]["HORAE"])]
put("loadBeijPeakWhole", LM["headline"]["beijing_alert"]["active"]/_e["S"][_r.index(max(_r))], "{:.1f}")
put("scaleNhi", commas(2040))
put("vMinIntel", 3*math.exp(-3*0.3/0.5)); put("vMinBeij", 3*math.exp(-3*4/8))
put("vRatioIntel", 7/(3*math.exp(-3*0.3/0.5)), "{:.0f}"); put("vRatioBeij", 7/(3*math.exp(-3*4/8)), "{:.1f}")
def _lk(S_, N_):
    return f"S{S_}:N{N_}"
_all_a = []
for S_ in (4, 11, 22, 44):
    for N_ in (96, 240, 504, 1008, 2040):
        _all_a.append((ACT[N_]/S_, ratio(Ovv, (_lk(S_, N_), "HORAE"), (_lk(S_, N_), "ALOHA")), S_, N_))
put("ovAlohaMin", min(a[1] for a in _all_a)); put("ovLoadMax", max(a[0] for a in _all_a), "{:.1f}")
put("ovAlohaAtMax", [a[1] for a in _all_a if a[0] == max(b[0] for b in _all_a)][0])
_L = _lk(4, 504)
put("ovLimFloorPct", 100*mm(Ovb, (_L, "HORAE-L")), "{:.0f}"); put("ovLimGoS", mm(Ovg, (_L, "HORAE-L")), "{:.1f}"); put("ovLimVw", mm(Ovv, (_L, "HORAE-L")), "{:.0f}")
put("ovFixVw", mm(Ovv, (_L, "HORAE")), "{:.0f}"); put("ovFixGoS", mm(Ovg, (_L, "HORAE")), "{:.2f}"); put("ovLoadA", ACT[504]/4, "{:.1f}")
_L2 = _lk(4, 2040)
put("ovLimGoSMax", mm(Ovg, (_L2, "HORAE-L")), "{:.0f}"); put("ovLimVwB", mm(Ovv, (_L2, "HORAE-L")), "{:.0f}"); put("ovFixVwB", mm(Ovv, (_L2, "HORAE")), "{:.0f}"); put("ovLoadB", ACT[2040]/4, "{:.1f}")
put("ovAdpVwB", mm(Ovv, (_L2, "ALOHA-ADP")), "{:.0f}"); put("ovAdpRatioB", ratio(Ovv, (_L2, "HORAE"), (_L2, "ALOHA-ADP")))
_L3 = _lk(4, 1008)
put("ovAdpRatioC", ratio(Ovv, (_L3, "HORAE"), (_L3, "ALOHA-ADP"))); put("ovLoadC", ACT[1008]/4, "{:.1f}")
put("ovGAdpRatioB", ratio(Ovv, (_L2, "HORAE-G"), (_L2, "ADP-G"))); put("ovGAdpRatioC", ratio(Ovv, (_L3, "HORAE-G"), (_L3, "ADP-G")))
put("ovAtaB", mm(Ovv, (_L2, "ATA")), "{:.0f}"); put("ovAgeB", mm(Ovv, (_L2, "AGE")), "{:.0f}")
_L4 = _lk(11, 2040)
put("ovAdpRatioD", ratio(Ovv, (_L4, "HORAE"), (_L4, "ALOHA-ADP"))); put("ovLoadD", ACT[2040]/11, "{:.1f}")
_L5 = _lk(22, 2040)
put("ovAdpRatioE", ratio(Ovv, (_L5, "HORAE"), (_L5, "ALOHA-ADP"))); put("ovLoadE", ACT[2040]/22, "{:.1f}")
put("ovFoneE", mm(Ovf, (_L5, "HORAE")), "{:.3f}"); put("ovFoneEAdp", mm(Ovf, (_L5, "ALOHA-ADP")), "{:.3f}")
put("ovFoneB", mm(Ovf, (_L2, "HORAE")), "{:.3f}"); put("ovFoneBAdp", mm(Ovf, (_L2, "ALOHA-ADP")), "{:.3f}")
put("ovJainB", mm(Ovj, (_L2, "HORAE")), "{:.2f}"); put("ovJainBAdp", mm(Ovj, (_L2, "ALOHA-ADP")), "{:.2f}")
_L6 = _lk(4, 96)
put("ovAdpLowE", ratio(Ovv, (_lk(11, 96), "HORAE"), (_lk(11, 96), "ALOHA-ADP"))); put("ovLoadLowE", ACT[96]/11, "{:.2f}")
put("ovAdpLowT", ratio(Ovv, (_lk(22, 240), "HORAE"), (_lk(22, 240), "ALOHA-ADP"))); put("ovLoadLowT", ACT[240]/22, "{:.2f}")
put("ovLowRatio", ratio(Ovv, (_L6, "HORAE"), (_L6, "ALOHA-ADP"))); put("ovLowHorae", mm(Ovv, (_L6, "HORAE")), "{:.1f}"); put("ovLowAdp", mm(Ovv, (_L6, "ALOHA-ADP")), "{:.1f}"); put("ovLowG", mm(Ovv, (_L6, "HORAE-G")), "{:.1f}"); put("ovLowGoS", mm(Ovg, (_L6, "HORAE")), "{:.2f}"); put("ovLoadF", ACT[96]/4, "{:.1f}")
put("ovLimRatioA", ratio(Ovv, (_lk(4, 504), "HORAE-L"), (_lk(4, 504), "ALOHA")))
put("nOvRatioA", ratio(Ovv, (_lk(4, 504), "HORAE"), (_lk(4, 504), "ALOHA"))); put("nOvRatioB", ratio(Ovv, (_lk(4, 2040), "HORAE"), (_lk(4, 2040), "ALOHA"))); put("nOvRatioC", ratio(Ovv, (_lk(11, 2040), "HORAE"), (_lk(11, 2040), "ALOHA")))
_E5 = R["E5"]


imp = [r for r in ovr if r["tag"].startswith("imp")]
It = {k: A.table(imp, k) for k in ("vwaoi", "age_active_routine", "age_active_critical", "age_active_p95", "jain_deliv", "deliv_min", "alert_f1", "S")}
PF = [("HORAE-R", "HORAE-R"), ("HORAE", "HORAE"), ("HORAE-G", "HORAE-G"), ("AGE", "Age-only"), ("VOI", "Value-only"), ("ATA", "ATA"), ("ALOHA", "Slotted ALOHA"), ("ALOHA-ADP", "Adaptive ALOHA")]
lines = []
for p, lab in PF:
    cells = [lab]
    for l_ in ("imp:2.0", "imp:4.0", "imp:8.0"):
        cells += ["%.0f" % mm(It["age_active_routine"], (l_, p)), "%.0f" % mm(It["age_active_critical"], (l_, p)), "%.0f" % mm(It["age_active_p95"], (l_, p)), "%.2f" % mm(It["jain_deliv"], (l_, p))]
    lines.append(" & ".join(cells) + r" \\")
TB["fair"] = "\n".join(lines)
for nm, l_ in (("A", "imp:2.0"), ("B", "imp:4.0"), ("C", "imp:8.0")):
    put(f"fairS{nm}", mm(It["S"], (l_, "HORAE")), "{:.0f}")
    put(f"fairRoutine{nm}", mm(It["age_active_routine"], (l_, "HORAE")), "{:.0f}")
    put(f"fairCritical{nm}", mm(It["age_active_critical"], (l_, "HORAE")), "{:.0f}")
    put(f"fairRoutineAdp{nm}", mm(It["age_active_routine"], (l_, "ALOHA-ADP")), "{:.0f}")
    put(f"fairRoutineVoi{nm}", mm(It["age_active_routine"], (l_, "VOI")), "{:.0f}")
    put(f"fairJain{nm}", mm(It["jain_deliv"], (l_, "HORAE")), "{:.2f}")
    put(f"fairJainAdp{nm}", mm(It["jain_deliv"], (l_, "ALOHA-ADP")), "{:.2f}")
    put(f"fairJainAge{nm}", mm(It["jain_deliv"], (l_, "AGE")), "{:.2f}")
    put(f"fairJainVoi{nm}", mm(It["jain_deliv"], (l_, "VOI")), "{:.2f}")
    put(f"fairMin{nm}", mm(It["deliv_min"], (l_, "HORAE")), "{:.1f}")
    put(f"fairMinAdp{nm}", mm(It["deliv_min"], (l_, "ALOHA-ADP")), "{:.1f}")
    put(f"fairPninefive{nm}", mm(It["age_active_p95"], (l_, "HORAE")), "{:.0f}")
    put(f"fairPninefiveAdp{nm}", mm(It["age_active_p95"], (l_, "ALOHA-ADP")), "{:.0f}")
    put(f"fairVwRatio{nm}", ratio(It["vwaoi"], (l_, "HORAE"), (l_, "ALOHA-ADP")))
    put(f"fairVwAta{nm}", ratio(It["vwaoi"], (l_, "HORAE"), (l_, "ATA")))
put("fairRoutineAge", mm(It["age_active_routine"], ("imp:4.0", "AGE")), "{:.0f}")
for nm, l_ in (("D", "impov:N1008"), ("E", "impov:N2040")):
    put(f"fairRoutine{nm}", mm(It["age_active_routine"], (l_, "HORAE")), "{:.0f}")
    put(f"fairRoutineAdp{nm}", mm(It["age_active_routine"], (l_, "ALOHA-ADP")), "{:.0f}")
    put(f"fairRoutineAge{nm}", mm(It["age_active_routine"], (l_, "AGE")), "{:.0f}")
    put(f"fairRoutineVoi{nm}", mm(It["age_active_routine"], (l_, "VOI")), "{:.0f}")
    put(f"fairJain{nm}", mm(It["jain_deliv"], (l_, "HORAE")), "{:.2f}")
    put(f"fairJainAdp{nm}", mm(It["jain_deliv"], (l_, "ALOHA-ADP")), "{:.2f}")
    put(f"fairPninefive{nm}", mm(It["age_active_p95"], (l_, "HORAE")), "{:,.0f}".replace(",", ""))
    put(f"fairPninefiveAdp{nm}", mm(It["age_active_p95"], (l_, "ALOHA-ADP")), "{:.0f}")


syn = A.load("synchrony")
Yv = A.table(syn, "vwaoi"); Yf = A.table(syn, "alert_f1"); Yg = A.table(syn, "G_over_S")
PSI = {r["tag"]: r["res"]["sync_index"] for r in syn if r["pol"] == "HORAE" and r["rep"] == 0 and "sync_index" in r["res"]}
put("psiSyncZero", PSI["0.0"], "{:.0f}"); put("psiSyncOne", PSI["1.0"], "{:.0f}")
for nm, s_ in (("Zero", "0.0"), ("Quarter", "0.25"), ("Half", "0.5"), ("One", "1.0"), ("Tq", "0.75"), ("Fiveeighths", "0.625")):
    for p, pm in (("HORAE", "Horae"), ("HORAE-G", "HoraeG"), ("ALOHA-ADP", "Adp"), ("ADP-G", "AdpG"), ("ALOHA", "Aloha"), ("AGE", "Age"), ("ATA", "Ata")):
        put(f"sy{nm}{pm}", mm(Yv, (s_, p)), "{:.1f}")
    put(f"sy{nm}HoraeF", mm(Yf, (s_, "HORAE")), "{:.3f}"); put(f"sy{nm}AdpF", mm(Yf, (s_, "ALOHA-ADP")), "{:.3f}"); put(f"sy{nm}HoraeGF", mm(Yf, (s_, "HORAE-G")), "{:.3f}"); put(f"sy{nm}AdpGF", mm(Yf, (s_, "ADP-G")), "{:.3f}")
_lv = ["0.0", "0.125", "0.25", "0.375", "0.5", "0.625", "0.75", "0.875", "1.0"]
_gw = [ratio(Yv, (s_, "HORAE-G"), (s_, "ADP-G")) for s_ in _lv]
put("syGuardLo", min(_gw)); put("syGuardHi", max(_gw))
_first_lose = [s_ for s_ in _lv if mm(Yv, (s_, "HORAE")) > mm(Yv, (s_, "ALOHA-ADP"))][0]
put("syLoseAdp", float(_first_lose), "{:g}")
_last_win = [s_ for s_ in _lv if mm(Yv, (s_, "HORAE")) <= mm(Yv, (s_, "ALOHA-ADP"))][-1]
put("syWinAdp", float(_last_win), "{:g}")
_first_lose2 = [s_ for s_ in _lv if mm(Yv, (s_, "HORAE")) > mm(Yv, (s_, "ALOHA"))][0]
put("syLoseAloha", float(_first_lose2), "{:g}")
put("syGoSOne", mm(Yg, ("1.0", "HORAE")))
put("sySval", max(r["res"]["S"] for r in syn), "{:d}")
put("burstSlotsSync", math.log(50/1e-3)/(0.25*math.log(math.log(2*max(r["res"]["S"] for r in syn)))), "{:.0f}")


gd = A.load("guards")
Gv = A.table(gd, "vwaoi", lambda r: r["pol"]); Gf = A.table(gd, "alert_f1", lambda r: r["pol"]); Gn = A.table(gd, "track_nrmse", lambda r: r["pol"])
REG = [("beijing_alert", "Beijing alert, $S=21$"), ("beijing_track", "Beijing tracking, $S=32$"), ("sync0.5", "Synchrony $s=0.5$"), ("sync1.0", "Synchrony $s=1$"),
       ("over_S4_N96", "$S=4$, load %.1f" % (ACT[96]/4)), ("over_S4_N1008", "$S=4$, load %.1f" % (ACT[1008]/4)), ("over_S4_N2040", "$S=4$, load %.1f" % (ACT[2040]/4)),
       ("raw_S2", "Raw Beijing, $S=2$"), ("raw_S1", "Raw Beijing, $S=1$")]
COLS = [("HORAE-R", "HORAE-R"), ("base", "HORAE"), ("ceil1", "HORAE"), ("freeze", "HORAE"), ("fast3", "HORAE"), ("G", "HORAE"), ("pmin1e-2", "HORAE"), ("expl10", "HORAE"), ("ALOHA-ADP", "ALOHA-ADP"), ("ADP-G", "ADP-G"), ("AGE-G", "AGE-G")]
lines = []
for reg, lab in REG:
    cells = [lab]
    for g_, p in COLS:
        v_ = mm(Gv, (f"{reg}:{g_}", p))
        cells.append(("%.1f" % v_) if v_ < 100 else ("%.0f" % v_))
    lines.append(" & ".join(cells) + r" \\")
TB["guard"] = "\n".join(lines)
put("gdSyncFastThree", mm(Gv, ("sync1.0:fast3", "HORAE")), "{:.0f}"); put("gdOverFastThree", mm(Gv, ("over_S4_N2040:fast3", "HORAE")), "{:.0f}")
put("gdSyncFastTen", mm(Gv, ("sync1.0:fast10", "HORAE")), "{:.0f}"); put("gdOverFastTen", mm(Gv, ("over_S4_N2040:fast10", "HORAE")), "{:.0f}")
put("gdPminOverB", mm(Gv, ("over_S4_N2040:pmin1e-2", "HORAE")), "{:.0f}"); put("gdPminFiveOverB", mm(Gv, ("over_S4_N2040:pmin5e-2", "HORAE")), "{:.0f}"); put("gdBaseOverB", mm(Gv, ("over_S4_N2040:base", "HORAE")), "{:.0f}")
put("gdPminSync", mm(Gv, ("sync1.0:pmin1e-2", "HORAE")), "{:.0f}"); put("gdExplSync", mm(Gv, ("sync1.0:expl10", "HORAE")), "{:.0f}"); put("gdBaseSync", mm(Gv, ("sync1.0:base", "HORAE")), "{:.0f}")
put("gdRawOneG", mm(Gv, ("raw_S1:G", "HORAE")), "{:.0f}")


cor = A.load("correlation")
Cv = A.table(cor, "vwaoi"); Cf = A.table(cor, "alert_f1")
_cp = {}
for r in cor:
    if r["pol"] == "HORAE" and "sync_index" in r["res"]:
        _cp.setdefault(r["tag"], []).append(r["res"]["sync_index"])
CPSI = {k_: float(np.mean(v_)) for k_, v_ in _cp.items()}
lines = []
for phi in ("0.9", "0.99", "0.999"):
    for ell, lab in (("None", "indep."), ("0.05", r"$d_c=0.05$"), ("0.3", r"$d_c=0.3$")):
        cells = [phi if ell == "None" else "", lab]
        for c_ in ("0.0", "0.25", "0.5", "0.75"):
            t_ = f"{phi}:{ell}:{c_}"
            cells.append("%.2f, %.2f (%.0f)" % (ratio(Cv, (t_, "HORAE-R"), (t_, "ALOHA-ADP")), ratio(Cv, (t_, "HORAE"), (t_, "ALOHA-ADP")), CPSI[t_]))
        lines.append(" & ".join(cells) + r" \\")
TB["corr"] = "\n".join(lines)
_ci = [ratio(Cv, (f"{phi}:{ell}:0.0", "HORAE"), (f"{phi}:{ell}:0.0", "ALOHA-ADP")) for phi in ("0.9", "0.99", "0.999") for ell in ("None", "0.05")]
put("corrIndepLo", min(_ci)); put("corrIndepHi", max(_ci))
_cc = [ratio(Cv, (f"{phi}:None:0.75", "HORAE"), (f"{phi}:None:0.75", "ALOHA-ADP")) for phi in ("0.9", "0.99", "0.999")]
put("corrCommonLo", min(_cc)); put("corrCommonHi", max(_cc))
put("corrCommonFast", _cc[0]); put("corrCommonSlow", _cc[2])
put("corrLongFast", ratio(Cv, ("0.9:0.3:0.0", "HORAE"), ("0.9:0.3:0.0", "ALOHA-ADP")))
put("corrLongFastG", ratio(Cv, ("0.9:0.3:0.0", "HORAE-G"), ("0.9:0.3:0.0", "ADP-G")))
put("corrZeroFast", ratio(Cv, ("0.9:None:0.0", "HORAE"), ("0.9:None:0.0", "ALOHA-ADP"))); put("corrZeroSlow", ratio(Cv, ("0.999:None:0.0", "HORAE"), ("0.999:None:0.0", "ALOHA-ADP")))
put("corrLongCommonFast", ratio(Cv, ("0.9:0.3:0.75", "HORAE"), ("0.9:0.3:0.75", "ALOHA-ADP")))
for nm, m_ in (("One", "1"), ("Ten", "10"), ("Hun", "100"), ("All", "1000")):
    t_ = f"burst:{m_}"
    put(f"corrBurst{nm}", ratio(Cv, (t_, "HORAE"), (t_, "ALOHA-ADP")))
    put(f"corrBurstG{nm}", ratio(Cv, (t_, "HORAE-G"), (t_, "ADP-G")))
    put(f"corrBurstPsi{nm}", CPSI[t_], "{:.0f}")
_pts = [(CPSI[t_], ratio(Cv, (t_, "HORAE"), (t_, "ALOHA-ADP")), t_) for t_ in CPSI if not t_.startswith("burst")]
_above = [p_[0] for p_ in _pts if p_[1] >= 1.0]; _below = [p_[0] for p_ in _pts if p_[1] < 1.0]
put("corrPsiAllAbove", min(_below), "{:.0f}"); put("corrPsiAllBelow", max(_above), "{:.0f}")
_gw = [ratio(Cv, (t_, "HORAE-G"), (t_, "ADP-G")) for t_ in CPSI if not t_.startswith("burst")]
put("corrGuardLo", min(_gw)); put("corrGuardHi", max(_gw))


cap = A.load("capture")
Kv = A.table(cap, "vwaoi"); Kf = A.table(cap, "alert_f1")
for p, pm in (("HORAE", "Horae"), ("HORAE-G", "HoraeG"), ("ALOHA-ADP", "Adp"), ("ALOHA", "Aloha"), ("ATA", "Ata"), ("AGE", "Age")):
    for t_, nm in (("None", "Pure"), ("6.0", "Six"), ("0.0", "Zero")):
        put(f"cap{pm}{nm}", mm(Kv, (t_, p)))
        put(f"cap{pm}{nm}F", mm(Kf, (t_, p)), "{:.3f}")
for t_, nm in (("None", "Pure"), ("9.0", "Nine"), ("6.0", "Six"), ("3.0", "Three"), ("0.0", "Zero")):
    put(f"capRatio{nm}", ratio(Kv, (t_, "HORAE"), (t_, "ALOHA-ADP")))


ho = A.load("heldout")
Hv = A.table(ho, "vwaoi"); Hf = A.table(ho, "alert_f1")
for nm, ds in (("Intel", "intel"), ("Beij", "beijing")):
    for lab, tg in (("Calib", "calib"), ("Whole", "whole"), ("Test", "test")):
        put(f"ho{nm}{tg.capitalize()}", ratio(Hv, (f"{ds}:{tg}", "HORAE"), (f"{ds}:{tg}", "ALOHA-ADP")))
        put(f"ho{nm}{tg.capitalize()}Aloha", ratio(Hv, (f"{ds}:{tg}", "HORAE"), (f"{ds}:{tg}", "ALOHA")))
        put(f"ho{nm}{tg.capitalize()}F", mm(Hf, (f"{ds}:{tg}", "HORAE")), "{:.3f}")
        put(f"ho{nm}{tg.capitalize()}FAdp", mm(Hf, (f"{ds}:{tg}", "ALOHA-ADP")), "{:.3f}")
for nm, cfg, T_ in (("Intel", "intel_alert", 1500), ("BeijAl", "beijing_alert", 6000), ("BeijTr", "beijing_track", 6000)):
    a0 = mm(Hv, (f"{cfg}:warm:5", "HORAE")); b0 = mm(Hv, (f"{cfg}:warm:{T_//2}", "HORAE"))
    put(f"warm{nm}Short", a0); put(f"warm{nm}Long", b0)
    put(f"warm{nm}RatioShort", ratio(Hv, (f"{cfg}:warm:5", "HORAE"), (f"{cfg}:warm:5", "ALOHA-ADP")))
    put(f"warm{nm}RatioLong", ratio(Hv, (f"{cfg}:warm:{T_//2}", "HORAE"), (f"{cfg}:warm:{T_//2}", "ALOHA-ADP")))
    put(f"len{nm}RatioQuarter", ratio(Hv, (f"{cfg}:len:{T_//4}", "HORAE"), (f"{cfg}:len:{T_//4}", "ALOHA-ADP")))
    put(f"len{nm}RatioFull", ratio(Hv, (f"{cfg}:len:{T_}", "HORAE"), (f"{cfg}:len:{T_}", "ALOHA-ADP")))
    put(f"len{nm}Quarter", mm(Hv, (f"{cfg}:len:{T_//4}", "HORAE"))); put(f"len{nm}Full", mm(Hv, (f"{cfg}:len:{T_}", "HORAE")))


fb = A.load("feedback")
Fv = A.table(fb, "vwaoi"); Ff = A.table(fb, "alert_f1"); Fn = A.table(fb, "track_nrmse"); Fsd = A.table(fb, "beta_log_sd"); Fg = A.table(fb, "G_over_S"); Fl = A.table(fb, "n_listen"); Ftx = A.table(fb, "tx"); Fs = A.table(fb, "slots"); FN = A.table(fb, "N"); Fe = A.table(fb, "energy_J")
for nm, cfg in (("Al", "beijing_alert"), ("Tr", "beijing_track"), ("In", "intel_alert")):
    base = mm(Fv, (f"{cfg}:ideal", "HORAE")); baseA = mm(Fv, (f"{cfg}:ideal", "ALOHA-ADP"))
    put(f"fb{nm}Ideal", base); put(f"fb{nm}IdealAdp", baseA)
    for k_, tg in (("LossA", "loss:0.1"), ("LossB", "loss:0.3"), ("LossC", "loss:0.5"), ("DelayA", "delay:1"), ("DelayB", "delay:5"), ("DelayC", "delay:20"), ("InconsA", "incons:0.1"), ("InconsB", "incons:0.3"),
                   ("CommonA", "common:0.3"), ("CommonB", "common:0.5"), ("SharedB", "shared:0.3"), ("DutyA", "duty:5"), ("DutyB", "duty:20"), ("DutyC", "duty:100"), ("MisA", "misalign:0.25"), ("MisB", "misalign:1.0"), ("Noack", "noack")):
        put(f"fb{nm}{k_}", mm(Fv, (f"{cfg}:{tg}", "HORAE")))
        put(f"fb{nm}{k_}Adp", mm(Fv, (f"{cfg}:{tg}", "ALOHA-ADP")))
        put(f"fb{nm}{k_}Ratio", ratio(Fv, (f"{cfg}:{tg}", "HORAE"), (f"{cfg}:{tg}", "ALOHA-ADP")))
        put(f"fb{nm}{k_}Sd", mm(Fsd, (f"{cfg}:{tg}", "HORAE")))
        put(f"fb{nm}{k_}Gs", mm(Fg, (f"{cfg}:{tg}", "HORAE")))
        put(f"fb{nm}{k_}Rel", 100*(mm(Fv, (f"{cfg}:{tg}", "HORAE"))/base - 1), "{:+.0f}")
        if (f"{cfg}:{tg}", "HORAE-G") in Fv:
            put(f"fb{nm}{k_}G", mm(Fv, (f"{cfg}:{tg}", "HORAE-G")))
    t_ = Ff if "alert" in cfg else Fn
    put(f"fb{nm}IdealQ", mm(t_, (f"{cfg}:ideal", "HORAE")), "{:.3f}")
    put(f"fb{nm}NoackQ", mm(t_, (f"{cfg}:noack", "HORAE")), "{:.3f}")
    put(f"fb{nm}DutyBQ", mm(t_, (f"{cfg}:duty:20", "HORAE")), "{:.3f}")
    _ls = 100*(1 - mm(Fl, (f"{cfg}:duty:20", "HORAE"))/mm(Fl, (f"{cfg}:ideal", "HORAE")))
    put(f"fb{nm}DutyBListenCut", _ls, "{:.0f}")


va = A.load("value")
Vv = A.table(va, "vwaoi"); Vf = A.table(va, "alert_f1"); Vn = A.table(va, "track_nrmse")
for nm, cfg in (("In", "intel_alert"), ("Al", "beijing_alert"), ("Tr", "beijing_track"), ("TrW", "beijing_track_w"), ("AlW", "beijing_alert_w")):
    b_ = (f"{cfg}:base", "ALOHA-ADP"); g_ = (f"{cfg}:base", "AGE")
    put(f"vaAdp{nm}", mm(Vv, b_)); put(f"vaAge{nm}", mm(Vv, g_)); put(f"vaTrue{nm}", mm(Vv, (f"{cfg}:true", "HORAE")))
    for k_, tg in (("NoiseA", "noise:0.5"), ("NoiseB", "noise:1.0"), ("NoiseC", "noise:2.0"), ("BiasB", "bias:1.0"), ("BiasC", "bias:2.0"), ("BitsA", "bits:1"), ("BitsB", "bits:2"), ("Inv", "invert"), ("Shuf", "shuffle")):
        put(f"va{k_}{nm}", mm(Vv, (f"{cfg}:{tg}", "HORAE")))
        put(f"va{k_}{nm}Ratio", A.paired(Vv[(f"{cfg}:{tg}", "HORAE")], Vv[b_], B=2000)["ratio"])
for nm, cfg in (("In", "intel_alert"), ("Al", "beijing_alert"), ("Tr", "beijing_track")):
    for k_, m_ in (("XA", "0.25"), ("XB", "1.0"), ("XC", "2.0")):
        t_ = f"{cfg}:xnoise:{m_}"
        put(f"va{k_}{nm}", mm(Vv, (t_, "HORAE"))); put(f"va{k_}{nm}Adp", mm(Vv, (t_, "ALOHA-ADP")))
        put(f"va{k_}{nm}Ratio", ratio(Vv, (t_, "HORAE"), (t_, "ALOHA-ADP")))
for nm, cfg in (("In", "intel_alert"), ("Al", "beijing_alert")):
    for k_, m_ in (("ThA", "-1.0"), ("ThB", "1.0"), ("ThC", "2.0")):
        t_ = f"{cfg}:thetaerr:{m_}"
        put(f"va{k_}{nm}", mm(Vv, (t_, "HORAE"))); put(f"va{k_}{nm}Adp", mm(Vv, (t_, "ALOHA-ADP")))
        put(f"va{k_}{nm}F", mm(Vf, (t_, "HORAE")), "{:.3f}"); put(f"va{k_}{nm}AdpF", mm(Vf, (t_, "ALOHA-ADP")), "{:.3f}")


se = A.load("sensitivity")
Sev = A.table(se, "vwaoi"); Sef = A.table(se, "alert_f1"); Sen = A.table(se, "track_nrmse")


def rng_ratio(cfg, name, vals):
    rr = [ratio(Sev, (f"{cfg}:{name}:{v_}", "HORAE"), (f"{cfg}:{name}:{v_}", "ALOHA-ADP")) for v_ in vals]
    return min(rr), max(rr), rr


GR = {"v_floor": (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5), "kappa": (1.0, 2.0, 3.0, 5.0, 8.0), "boost": (0.0, 1.0, 2.0, 4.0, 8.0, 16.0), "rho_mul": (0.25, 0.5, 1.0, 2.0, 4.0),
      "delta_mul": (0.5, 1.0, 2.0), "pct": (80, 85, 90, 92, 95, 98), "eta": (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)}
NM = {"v_floor": "Vf", "kappa": "Ka", "boost": "Bo", "rho_mul": "Rh", "delta_mul": "De", "pct": "Pc", "eta": "Et"}
for cfg, cn in (("intel_alert", "In"), ("beijing_alert", "Al"), ("beijing_track", "Tr")):
    for name, vals in GR.items():
        if (f"{cfg}:{name}:{vals[0]}", "HORAE") not in Sev:
            continue
        lo, hi, rr = rng_ratio(cfg, name, vals)
        put(f"se{NM[name]}{cn}Lo", lo); put(f"se{NM[name]}{cn}Hi", hi)
        vw_ = [mm(Sev, (f"{cfg}:{name}:{v_}", "HORAE")) for v_ in vals]
        put(f"se{NM[name]}{cn}VwLo", min(vw_)); put(f"se{NM[name]}{cn}VwHi", max(vw_))
    for bc, nm_ in ((0.1, "A"), (1.0, "B"), (10.0, "C"), (50.0, "D")):
        put(f"seBc{nm_}{cn}", mm(Sev, (f"{cfg}:beta_ceil:{bc}", "HORAE")))
    _b0 = [mm(Sev, (f"{cfg}:beta0:{b_}", "HORAE")) for b_ in (1e-4, 1e-3, 0.02, 0.1, 1.0)]
    put(f"seBz{cn}Lo", min(_b0)); put(f"seBz{cn}Hi", max(_b0))
for cfg, cn in (("beijing_track", "Tr"), ("beijing_alert", "Al")):
    for wc, nm_ in ((1.0, "A"), (4.0, "B"), (8.0, "C"), (16.0, "D"), (32.0, "E")):
        t_ = f"{cfg}:w:{wc}:0.15"
        put(f"seW{nm_}{cn}Adp", ratio(Sev, (t_, "HORAE"), (t_, "ALOHA-ADP")))
        put(f"seW{nm_}{cn}Age", ratio(Sev, (t_, "HORAE"), (t_, "AGE")))
        put(f"seW{nm_}{cn}Voi", ratio(Sev, (t_, "HORAE"), (t_, "VOI")))


mainE = {k: A.table(main, k, variant) for k in ("tx", "n_listen", "slots", "N", "energy_J", "val_delivered", "succ", "energy_per_dev_slot_uJ", "listen_J", "tx_J", "rx_J", "comp_J", "vwaoi")}


def rates(cfg, p):
    N_ = mm(mainE["N"], (cfg, p)); sl = mm(mainE["slots"], (cfg, p))
    return EP.device_rates(p, mm(mainE["tx"], (cfg, p)), mm(mainE["succ"], (cfg, p)), mm(mainE["n_listen"], (cfg, p)), N_*sl)


PE = [("HORAE-R", "HORAE-R"), ("AGE-R", "Age-R"), ("HORAE", "HORAE"), ("HORAE-G", "HORAE-G"), ("AGE", "Age-only"), ("ATA-T", "ATA-T"), ("GOAL", "GOAL"), ("AOII", "AoII"), ("ALOHA", "Slotted ALOHA"), ("ALOHA-ADP", "Adaptive ALOHA"), ("RG", "RG")]
lines = []
for cfg, cn in (("intel_alert", "Intel-Lab alert"), ("beijing_alert", "Beijing alert"), ("beijing_track", "Beijing tracking")):
    lines.append(("" if cfg == "intel_alert" else r"\addlinespace[3pt]") + r"\multicolumn{8}{l}{\emph{%s}}\\" % cn)
    for p, lab in PE:
        a_, l_ = rates(cfg, p)
        e1 = mm(mainE["energy_per_dev_slot_uJ"], (cfg, p))
        eps = 1e6*mm(mainE["energy_J"], (cfg, p))/mm(mainE["succ"], (cfg, p))
        valmj = mm(mainE["val_delivered"], (cfg, p))/(1e3*mm(mainE["energy_J"], (cfg, p)))
        vwe = mm(mainE["vwaoi"], (cfg, p))*e1
        Bb = EP.bits(CONFIGS[cfg]["S"])
        ops_ = EP.read_ops(os.path.join(RES, "device_loop_r.txt")) if p in ("HORAE-R", "AGE-R") else OPS
        lf1 = EP.lifetime_years("lora", 1.0, a_, l_, ops=ops_, **Bb); lf10 = EP.lifetime_years("lora", 10.0, a_, l_, ops=ops_, **Bb)
        lines.append(" & ".join([lab, "%.4f" % a_, "%.2f" % e1, "%.0f" % eps, "%.1f" % valmj, "%.1f" % vwe, "%.2f" % lf1, "%.1f" % lf10]) + r" \\")
TB["energy"] = "\n".join(lines)
for cfg, cn in (("intel_alert", "In"), ("beijing_alert", "Al"), ("beijing_track", "Tr")):
    for p, pm in (("HORAE", "Horae"), ("ALOHA", "Aloha"), ("ALOHA-ADP", "Adp"), ("ATA-T", "AtaT"), ("GOAL", "Goal"), ("HORAE-G", "HoraeG")):
        a_, l_ = rates(cfg, p)
        Bb = EP.bits(CONFIGS[cfg]["S"])
        put(f"lf{cn}{pm}One", EP.lifetime_years("lora", 1.0, a_, l_, ops=OPS, **Bb)); put(f"lf{cn}{pm}Ten", EP.lifetime_years("lora", 10.0, a_, l_, ops=OPS, **Bb), "{:.1f}")
        put(f"lf{cn}{pm}Sl", EP.lifetime_years("lora", 31.0, a_, l_, ops=OPS, **Bb), "{:.1f}")
        put(f"lfz{cn}{pm}One", EP.lifetime_years("zigbee", 1.0, a_, l_, ops=OPS, **Bb), "{:.1f}")
        put(f"att{cn}{pm}", a_, "{:.4f}")
        put(f"enFo{cn}{pm}", mm(mainE["energy_per_dev_slot_uJ"], (cfg, p)))
        put(f"enPs{cn}{pm}", 1e6*mm(mainE["energy_J"], (cfg, p))/mm(mainE["succ"], (cfg, p)), "{:.0f}")
        put(f"enVm{cn}{pm}", mm(mainE["val_delivered"], (cfg, p))/(1e3*mm(mainE["energy_J"], (cfg, p))), "{:.1f}")
_ei = EP.slot_energy_J("lora", 10.0, *rates("intel_alert", "HORAE"), ops=OPS, **EP.bits(62))
put("shareRxIn", 100*_ei["rx"]/_ei["total"], "{:.0f}"); put("shareTxIn", 100*_ei["tx"]/_ei["total"], "{:.0f}"); put("shareCpuIn", 100*_ei["cpu"]/_ei["total"], "{:.2f}"); put("shareSleepIn", 100*_ei["sleep"]/_ei["total"], "{:.0f}"); put("shareWakeIn", 100*_ei["wake"]/_ei["total"], "{:.1f}")

_a0 = rates("intel_alert", "HORAE")[0]
_ops = _a0*(OPS["active"]["basic"] + OPS["active"]["transcendental"]) + (1 - _a0)*(OPS["inactive"]["basic"] + OPS["inactive"]["transcendental"])
_full = _ops*1e-9*mm(mainE["N"], ("intel_alert", "HORAE"))*mm(mainE["slots"], ("intel_alert", "HORAE"))
put("cpuFullShareFo", 100*_full/(mm(mainE["energy_J"], ("intel_alert", "HORAE")) - mm(mainE["comp_J"], ("intel_alert", "HORAE")) + _full), "{:.1f}")

_h, _a = ("intel_alert", "HORAE"), ("intel_alert", "ALOHA")
_Eh, _Ea = mm(mainE["energy_J"], _h), mm(mainE["energy_J"], _a)
_Lh, _La = mm(mainE["listen_J"], _h), mm(mainE["listen_J"], _a)
put("listenFracIntelPct", 100*_Lh/_Eh, "{:.0f}"); put("listenFracIntelAlohaPct", 100*_La/_Ea, "{:.0f}")
put("compFracIntelPct", 100*mm(mainE["comp_J"], _h)/_Eh, "{:.1f}")
put("enRatioIntel", _Eh/_Ea, "{:.1f}"); put("enRatioIntelNoBeacon", _Eh/(_Ea - _La), "{:.1f}"); put("enIntelHorae", _Eh, "{:.2f}")
_Nh, _Sl, _Tx = mm(mainE["N"], _h), mm(mainE["slots"], _h), mm(mainE["tx"], _h)
_idle, _map = math.ceil(math.log2(62 + 1)), math.ceil(62*math.log2(3))
put("enIntelHoraeRealFb", _Eh - _Lh + (_Nh*_Sl*_idle + _Tx*_map)*50e-9, "{:.2f}")
put("enIntelHoraeFullMap", _Eh - _Lh + _Nh*_Sl*_map*50e-9, "{:.2f}")

for nm, cfg in (("Al", "beijing_alert"), ("In", "intel_alert")):
    for d_, k_ in ((1, "One"), (5, "A"), (20, "B"), (100, "C")):
        tg = f"{cfg}:ideal" if d_ == 1 else f"{cfg}:duty:{d_}"
        N_ = mm(FN, (tg, "HORAE")); sl_ = mm(Fs, (tg, "HORAE"))
        a_ = mm(Ftx, (tg, "HORAE"))/(N_*sl_); l_ = mm(Fl, (tg, "HORAE"))/(N_*sl_)
        put(f"dc{nm}{k_}Life", EP.lifetime_years("lora", 10.0, a_, l_, ops=OPS), "{:.1f}")
        put(f"dc{nm}{k_}Listen", l_, "{:.3f}")


fl = A.load("fields")
if fl:
    Zv = A.table(fl, "vwaoi"); Zf = A.table(fl, "alert_f1"); Zn = A.table(fl, "track_nrmse")
    ZS = {}; ZPSI = {}; ZA = {}
    for r in fl:
        ZS[r["tag"]] = r["res"]["S"]
        if r["pol"] == "HORAE" and r["rep"] == 0:
            ZPSI[r["tag"]] = r["res"].get("sync_index", float("nan")); ZA[r["tag"]] = r["res"].get("active_frac", float("nan"))
    FIELDS = [n_ for n_ in ("aqs", "o3", "pems") if any(k[0].startswith(n_ + ":") for k in Zv)]
    LABF = {"aqs": "EPA PM$_{2.5}$", "o3": "EPA ozone", "pems": "PEMS occupancy"}
    z_ = {}
    for n_ in FIELDS:
        z = np.load(os.path.join(CODE, "data", f"{n_}_field.npz"), allow_pickle=True)
        z_[n_] = (z["X"].shape, int(z["n_all"]) if "n_all" in z else z["X"].shape[1])
    tag_ = {"aqs": "aqs", "o3": "othree", "pems": "pems"}
    for n_ in FIELDS:
        put(f"{tag_[n_]}N", z_[n_][0][1], "{:d}"); put(f"{tag_[n_]}T", commas(z_[n_][0][0] - 1)); put(f"{tag_[n_]}All", commas(z_[n_][1]))
    if "pems" in FIELDS:
        put("pemsDays", z_["pems"][0][0]/144, "{:.0f}")
    PLC = [("HORAE-R", "HORAE-R"), ("HORAE", "HORAE"), ("HORAE-G", "HORAE-G"), ("AGE", "Age-only"), ("ATA", "ATA"), ("GOAL", "GOAL"), ("ALOHA", "S-ALOHA"), ("ALOHA-ADP", "Ad-ALOHA"), ("RG", "RG"), ("MAXWEIGHT", "Max-weight")]
    lines = []
    rr_all = []
    for n_ in FIELDS:
        for kind, kl in (("alert", "alert"), ("track", "tracking")):
            for l_ in ("1.0", "2.0"):
                t_ = f"{n_}:{kind}:{l_}"
                qt = Zf if kind == "alert" else Zn
                cells = [LABF[n_] if (kind == "alert" and l_ == "1.0") else "", kl if l_ == "1.0" else "", "%s (%d)" % (l_.rstrip("0").rstrip("."), ZS[t_]), "%.0f" % ZPSI[t_]]
                cells += ["%.1f" % mm(Zv, (t_, p)) for p, _ in PLC]
                cells += ["%.2f" % ratio(Zv, (t_, "HORAE-R"), (t_, "ALOHA-ADP")), "%.2f" % ratio(Zv, (t_, "HORAE"), (t_, "ALOHA-ADP")), "%.3f" % mm(qt, (t_, "HORAE-R")), "%.3f" % mm(qt, (t_, "ALOHA-ADP"))]
                lines.append(" & ".join(cells) + r" \\")
    TB["fields"] = "\n".join(lines)
    for n_ in FIELDS:
        for kind in ("alert", "track"):
            for l_ in ("0.5", "1.0", "2.0", "4.0"):
                rr_all.append(ratio(Zv, (f"{n_}:{kind}:{l_}", "HORAE"), (f"{n_}:{kind}:{l_}", "ALOHA-ADP")))
    put("fieldsLo", min(rr_all)); put("fieldsHi", max(rr_all))
    for n_ in FIELDS:
        tn = tag_[n_]
        for kind, kn in (("alert", "Al"), ("track", "Tr")):
            t1, t2 = f"{n_}:{kind}:1.0", f"{n_}:{kind}:2.0"
            put(f"f{tn}{kn}PsiOne", ZPSI[t1], "{:.0f}"); put(f"f{tn}{kn}ActPct", 100*ZA[t1], "{:.1f}")
            for p, pm in (("HORAE", "Horae"), ("HORAE-G", "HoraeG"), ("ALOHA-ADP", "Adp"), ("ADP-G", "AdpG"), ("ALOHA", "Aloha"), ("ATA", "Ata"), ("AGE", "Age"), ("GOAL", "Goal"), ("RG", "Rg"), ("TDMA", "Tdma")):
                put(f"f{tn}{kn}{pm}One", mm(Zv, (t1, p))); put(f"f{tn}{kn}{pm}Two", mm(Zv, (t2, p)))
            put(f"f{tn}{kn}RatioOne", ratio(Zv, (t1, "HORAE"), (t1, "ALOHA-ADP"))); put(f"f{tn}{kn}RatioTwo", ratio(Zv, (t2, "HORAE"), (t2, "ALOHA-ADP")))
            put(f"f{tn}{kn}RatioGOne", ratio(Zv, (t1, "HORAE-G"), (t1, "ADP-G"))); put(f"f{tn}{kn}RatioGTwo", ratio(Zv, (t2, "HORAE-G"), (t2, "ADP-G")))
            put(f"f{tn}{kn}RatioAlohaOne", ratio(Zv, (t1, "HORAE"), (t1, "ALOHA"))); put(f"f{tn}{kn}RatioAlohaTwo", ratio(Zv, (t2, "HORAE"), (t2, "ALOHA")))
            for l_, ln in (("0.5", "Half"), ("4.0", "Four")):
                put(f"f{tn}{kn}Ratio{ln}", ratio(Zv, (f"{n_}:{kind}:{l_}", "HORAE"), (f"{n_}:{kind}:{l_}", "ALOHA-ADP")))
            s1, s2 = t1 + ":shifted", t2 + ":shifted"
            put(f"f{tn}{kn}ShiftOne", ratio(Zv, (s1, "HORAE"), (s1, "ALOHA-ADP"))); put(f"f{tn}{kn}ShiftTwo", ratio(Zv, (s2, "HORAE"), (s2, "ALOHA-ADP")))
            put(f"f{tn}{kn}ShiftPsi", ZPSI[s1], "{:.1f}")
            put(f"f{tn}{kn}ShiftHoraeOne", mm(Zv, (s1, "HORAE")))
            put(f"f{tn}{kn}ShiftAloha", ratio(Zv, (s1, "HORAE"), (s1, "ALOHA")))
            q_ = Zf if kind == "alert" else Zn
            put(f"f{tn}{kn}QOne", mm(q_, (t1, "HORAE")), "{:.3f}"); put(f"f{tn}{kn}QOneAdp", mm(q_, (t1, "ALOHA-ADP")), "{:.3f}"); put(f"f{tn}{kn}QOneGoal", mm(q_, (t1, "GOAL")), "{:.3f}")

if fl:
    keys_ = [(n_, k_) for n_ in FIELDS for k_ in ("alert", "track")]
    psi_ = [ZPSI[f"{n_}:{k_}:1.0"] for n_, k_ in keys_]; spsi_ = [ZPSI[f"{n_}:{k_}:1.0:shifted"] for n_, k_ in keys_]
    put("fieldPsiLo", min(psi_), "{:.0f}"); put("fieldPsiHi", max(psi_), "{:.0f}")
    put("fieldShiftPsiLo", min(spsi_), "{:.1f}"); put("fieldShiftPsiHi", max(spsi_), "{:.1f}")
    all_ = [f"{n_}:{k_}:{l_}" for n_, k_ in keys_ for l_ in ("0.5", "1.0", "2.0", "4.0")]
    a_ = [ratio(Zv, (t_, "HORAE"), (t_, "ALOHA")) for t_ in all_]
    put("fieldAlohaLo", min(a_)); put("fieldAlohaHi", max(a_))
    a_ = [ratio(Zv, (t_, "HORAE"), (t_, "AGE")) for t_ in all_]
    put("fieldAgeLo", min(a_)); put("fieldAgeHi", max(a_))
    a_ = [ratio(Zv, (t_, "HORAE-G"), (t_, "ADP-G")) for t_ in all_ if not t_.endswith(":0.5")]
    put("fieldGuardAbLo", min(a_)); put("fieldGuardAbHi", max(a_))
    a_ = [ratio(Zv, (t_, "HORAE-G"), (t_, "ADP-G")) for t_ in ("o3:alert:0.5", "pems:alert:0.5", "pems:track:0.5")]
    put("fieldGuardHalfLo", min(a_)); put("fieldGuardHalfHi", max(a_))

put("seWShareLo", ratio(Sev, ("beijing_track:w:16.0:0.05", "HORAE"), ("beijing_track:w:16.0:0.05", "ALOHA-ADP")))
put("seWShareHi", ratio(Sev, ("beijing_track:w:16.0:0.3", "HORAE"), ("beijing_track:w:16.0:0.3", "ALOHA-ADP")))
_wa = [ratio(Sev, (t_, "HORAE"), (t_, "ALOHA-ADP")) for t_ in sorted({k_[0] for k_ in Sev if k_[0].startswith("beijing_alert:w:")})]
put("seWAlLo", min(_wa)); put("seWAlHi", max(_wa))


main_act = A.table(main, "vwaoi_active", variant)
for t, cfg in TAG.items():
    a_ = mm(main_act, (cfg, "HORAE")); v_ = mm(VW, (cfg, "HORAE"))
    put(f"activeQuiet{t}", 100*(1 - a_/v_), "{:.0f}")
    put(f"activeShare{t}", 100*a_/v_, "{:.0f}")
    s_ = A.paired(main_act[(cfg, "HORAE")], main_act[(cfg, "ALOHA-ADP")], B=2000)
    put(f"rActive{t}Adp", s_["ratio"]); put(f"rActive{t}AdpLo", s_["lo"]); put(f"rActive{t}AdpHi", s_["hi"])
fb2 = A.load("feedback2")
if fb2:
    Bv = A.table(fb2, "vwaoi"); Bsd = A.table(fb2, "beta_log_sd"); Bf = A.table(fb2, "alert_f1")
    put("lowGainEta", "0.02")
    put("lowGainAl", mm(Bv, ("beijing_alert:gain:20:0.02", "HORAE")))
    put("lowGainTr", mm(Bv, ("beijing_track:gain:20:0.02", "HORAE")))
    put("incsSdTr", mm(Bsd, ("beijing_track:incs:0.3", "HORAE")))
    put("bcSdLossTr", mm(Bsd, ("beijing_track:bc:loss:0.3", "HORAE")))
    put("bcLossBTr", mm(Bv, ("beijing_track:bc:loss:0.3", "HORAE")))
    put("bcLossBAl", mm(Bv, ("beijing_alert:bc:loss:0.3", "HORAE")))
    put("guardAlSmall", mm(Bv, ("beijing_alert:guard:0.1", "HORAE")))
    put("guardAlLarge", mm(Bv, ("beijing_alert:guard:0.5", "HORAE")))
    put("guardTrSmall", mm(Bv, ("beijing_track:guard:0.1", "HORAE")))
else:
    for k_ in ("lowGainEta", "lowGainAl", "lowGainTr", "incsSdTr", "bcSdLossTr", "bcLossBTr", "bcLossBAl", "guardAlSmall", "guardAlLarge", "guardTrSmall"):
        put(k_, "0")
put("fbTrDutyTwo", mm(Fv, ("beijing_track:duty:2", "HORAE")))
put("sharedDriftAl", 100*(mm(Fv, ("beijing_alert:shared:0.3", "HORAE"))/mm(Fv, ("beijing_alert:ideal", "HORAE")) - 1), "{:.0f}")
put("xnoiseAlValue", "10"); put("thetaAlValue", "10")
put("corrGuardCommonFast", ratio(Cv, ("0.9:None:0.75", "HORAE-G"), ("0.9:None:0.75", "ADP-G")))


import run_studies as RR

OPSR = EP.read_ops(os.path.join(RES, "device_loop_r.txt"))
HR_T = {"Intel": "intel_alert", "BeijAl": "beijing_alert", "BeijTr": "beijing_track", "BeijAlW": "beijing_alert_w", "BeijTrW": "beijing_track_w"}
for t, cfg in HR_T.items():
    for p, pm in (("ALOHA", "Aloha"), ("ALOHA-ADP", "Adp"), ("ADP-G", "AdpG"), ("HORAE", "Horae"), ("HORAE-G", "HoraeG"), ("AGE", "Age"), ("AGE-R", "AgeR"), ("VOI", "Voi"), ("ATA", "Ata"), ("ATA-T", "AtaT"), ("AOII", "Aoii"),
                  ("GOAL", "Goal"), ("LTP-T", "LtpT"), ("RG", "Rg"), ("TDMA", "Tdma"), ("MAXWEIGHT", "Oracle")):
        if (cfg, p) in VW:
            s = A.paired(VW[(cfg, "HORAE-R")], VW[(cfg, p)])
            put(f"hr{t}{pm}", s["ratio"]); put(f"hr{t}{pm}Lo", s["lo"]); put(f"hr{t}{pm}Hi", s["hi"])


HEAD_R = [("Intel-Lab alert", "intel_alert"), ("Beijing alert", "beijing_alert"), ("Beijing tracking", "beijing_track"),
          ("Beijing alert, $w_i$ mixed", "beijing_alert_w"), ("Beijing tracking, $w_i$ mixed", "beijing_track_w")]
COMP_R = ["HORAE", "AGE-R", "ATA-T", "ALOHA-ADP", "ALOHA"]
_pv = []; _rows = []
for lab, cfg in HEAD_R:
    cells = [A.paired(VW[(cfg, "HORAE-R")], VW[(cfg, q)]) for q in COMP_R]
    _pv += [c["p_w"] for c in cells]
    _rows.append((lab, cells))
_adj = iter(A.holm(_pv))
_body = []
for lab, cells in _rows:
    tx = [lab]
    for c in cells:
        a_ = next(_adj)
        tx.append("%.2f [%.2f, %.2f]%s" % (c["ratio"], c["lo"], c["hi"], "" if a_ < 0.05 else "$^{\\dagger}$"))
    _body.append(" & ".join(tx) + r" \\")
TB["thresh"] = "\n".join(_body)
_ap = A.holm(_pv)
assert all(a_ < 0.05 for a_ in _ap), "the caption of the table says that every entry is significant"
put("threshTotal", len(_ap), "{:d}")


_rk, _gm = RR.select_r("hr")
put("calGridN", len(_gm), "{:d}")
_best = _rk[0]; _sel = (2.0, 0.25, 1.0)
assert _gm[_sel]/_gm[_best] <= 1.01, "the setting used is not among those tied with the best"
_tied = [k_ for k_ in _rk if _gm[k_]/_gm[_best] <= 1.01]
put("calTiedN", len(_tied), "{:d}")
put("calExcessPct", 100*(_gm[_sel]/_gm[_best] - 1), "{:.1f}")
put("calBestLambda", _best[0], "{:g}"); put("calBestMu", _best[1], "{:g}"); put("calBestNu", _best[2], "{:g}")
_win, _t2 = RR.selected_r()
assert _win == _sel and _t2 == _tied, "the tie-break does not select the setting used: %s" % (_win,)

assert RR.selected_r(0.005)[0] == (2.0, 0.25, 0.5) and RR.selected_r(0.015)[0] == (3.0, 0.25, 1.0)

_q = {}
for line in open(os.path.join(RES, "ext_calib.jsonl"), encoding="utf8"):
    r_ = json.loads(line); t_ = r_["tag"].split(":")
    if t_[1] == "hr":
        q_ = r_["res"].get("alert_f1"); k_ = "al" if q_ is not None else "tr"
        q_ = q_ if q_ is not None else 1.0/r_["res"]["track_nrmse"]
        _q.setdefault((k_, tuple(float(x_) for x_ in t_[2:])), []).append(math.log(q_))
assert max(_tied, key=lambda k_: np.mean(_q[("al", k_)]))[2] == 0.5 and max(_tied, key=lambda k_: np.mean(_q[("tr", k_)]))[2] == 1.0
_rk, _gm = RR.select_r("ar")
assert _rk[0] == (0.5, 1.0), "the age-only setting is not the one used: %s" % (_rk[0],)
put("lambdaVal", 2, "{:d}"); put("ageRLambda", "0.5")


TH = A.load("thresh")
Tv = A.table(TH, "vwaoi"); Tf = A.table(TH, "alert_f1"); Tn = A.table(TH, "track_nrmse"); Te = A.table(TH, "energy_J"); Ts = A.table(TH, "succ"); Tval = A.table(TH, "val_delivered"); Ted = A.table(TH, "energy_per_dev_slot_uJ"); Tl = A.table(TH, "n_listen"); Ttx = A.table(TH, "tx"); TN = A.table(TH, "N"); Tsl = A.table(TH, "slots")
for t, cfg in HR_T.items():
    for pn, pm in (("full", "Full"), ("noexempt", "NoEx"), ("nothresh", "NoTh"), ("noguard", "NoGd"), ("noceil", "NoCe")):
        put(f"pt{t}{pm}", mm(Tv, (f"part:{cfg}:{pn}", "HORAE")))
        q_ = Tf if "alert" in cfg else Tn
        put(f"pt{t}{pm}Q", mm(q_, (f"part:{cfg}:{pn}", "HORAE")), "{:.3f}")
for reg, nm in (("sync1.0", "SyncOne"), ("sync0.5", "SyncHalf"), ("o3_a1", "Othree"), ("pems_a1", "Pems"), ("aqs_a1", "Aqs"), ("over4_96", "Low")):
    for pn, pm in (("full", "Full"), ("noexempt", "NoEx"), ("nothresh", "NoTh"), ("noguard", "NoGd"), ("noceil", "NoCe")):
        put(f"pt{nm}{pm}", mm(Tv, (f"part:{reg}:{pn}", "HORAE")))

for t, cfg in HR_T.items():
    q_ = Tf if "alert" in cfg else Tn
    for l_, nm in ((0.5, "Half"), (1.0, "One"), (2.0, "Two"), (3.0, "Three"), (4.0, "Four")):
        put(f"tv{t}{nm}", mm(Tv, (f"{cfg}:{l_}:0.25:1.0", "HORAE")))
        put(f"tv{t}{nm}Q", mm(q_, (f"{cfg}:{l_}:0.25:1.0", "HORAE")), "{:.3f}")
        put(f"tv{t}{nm}Rt", mm(Tv, (f"{cfg}:{l_}:0.25:0.5", "HORAE")))
        put(f"tv{t}{nm}RtQ", mm(q_, (f"{cfg}:{l_}:0.25:0.5", "HORAE")), "{:.3f}")


PE_R = [("HORAE-R", "HoraeR")]
for cfg, cn in (("intel_alert", "In"), ("beijing_alert", "Al"), ("beijing_track", "Tr")):
    for p, pm in PE_R + [("TDMA", "Tdma"), ("RG", "Rg"), ("AOII", "Aoii")]:
        a_, l_ = rates(cfg, p)
        Bb = EP.bits(CONFIGS[cfg]["S"])
        ops_ = OPSR if p == "HORAE-R" else OPS
        put(f"lf{cn}{pm}One", EP.lifetime_years("lora", 1.0, a_, l_, ops=ops_, **Bb)); put(f"lf{cn}{pm}Ten", EP.lifetime_years("lora", 10.0, a_, l_, ops=ops_, **Bb), "{:.1f}")
        put(f"att{cn}{pm}", a_, "{:.4f}")
        put(f"enFo{cn}{pm}", mm(mainE["energy_per_dev_slot_uJ"], (cfg, p)))
        put(f"enPs{cn}{pm}", 1e6*mm(mainE["energy_J"], (cfg, p))/mm(mainE["succ"], (cfg, p)), "{:.0f}")
        put(f"enVm{cn}{pm}", mm(mainE["val_delivered"], (cfg, p))/(1e3*mm(mainE["energy_J"], (cfg, p))), "{:.1f}")
    for p, pm in (("HORAE-R", "HoraeR"), ("HORAE", "Horae"), ("ALOHA", "Aloha"), ("ATA-T", "AtaT"), ("RG", "Rg"), ("GOAL", "Goal")):
        put(f"enVe{cn}{pm}", mm(mainE["vwaoi"], (cfg, p))*mm(mainE["energy_per_dev_slot_uJ"], (cfg, p)), "{:.1f}")
    for tau, tn in ((1.0, "One"), (0.75, "Tq"), (0.5, "Half"), (0.25, "Quarter")):
        for M_, mn in ((1, "A"), (20, "B"), (100, "C")):
            tg = f"energy:{cfg}:{tau}:{M_}"
            put(f"eg{cn}{tn}{mn}Vw", mm(Tv, (tg, "HORAE-R")))
            q_ = Tf if cfg != "beijing_track" else Tn
            put(f"eg{cn}{tn}{mn}Q", mm(q_, (tg, "HORAE-R")), "{:.3f}")
            put(f"eg{cn}{tn}{mn}Vm", mm(Tval, (tg, "HORAE-R"))/(1e3*mm(Te, (tg, "HORAE-R"))), "{:.1f}")
            put(f"eg{cn}{tn}{mn}Ed", mm(Ted, (tg, "HORAE-R")))
            N_ = mm(TN, (tg, "HORAE-R")); sl_ = mm(Tsl, (tg, "HORAE-R"))
            Bb = EP.bits(CONFIGS[cfg]["S"])
            put(f"eg{cn}{tn}{mn}Life", EP.lifetime_years("lora", 10.0, mm(Ttx, (tg, "HORAE-R"))/(N_*sl_), mm(Tl, (tg, "HORAE-R"))/(N_*sl_), ops=OPSR, **Bb), "{:.1f}")


UH = A.load("unheard")
Uu = A.table(UH, "frac_unheard"); Uv = A.table(UH, "vwaoi"); Uj = A.table(UH, "jain_deliv"); Uf = A.table(UH, "alert_f1")
for N_, nm in ((504, "A"), (1008, "C"), (2040, "B")):
    for p, pm in (("HORAE-R", "HoraeR"), ("HORAE", "Horae"), ("HORAE-G", "HoraeG"), ("ALOHA-ADP", "Adp"), ("ADP-G", "AdpG"), ("ALOHA", "Aloha"), ("ATA", "Ata"), ("AGE", "Age"), ("VOI", "Voi"), ("RG", "Rg")):
        put(f"uh{nm}{pm}", 100*mm(Uu, (f"S4:N{N_}", p)), "{:.1f}")
    put(f"uhJain{nm}HoraeR", mm(Uj, (f"S4:N{N_}", "HORAE-R")))
_a_ = []; _b_ = []
for S_ in (4, 11, 22, 44):
    for N_ in (96, 240, 504, 1008, 2040):
        _a_.append(ratio(Ovv, (_lk(S_, N_), "HORAE-R"), (_lk(S_, N_), "ALOHA")))
        _b_.append((ratio(Ovv, (_lk(S_, N_), "HORAE-R"), (_lk(S_, N_), "ALOHA-ADP")), S_, N_))
put("ovRAlohaMin", min(_a_)); put("ovRAlohaMax", max(_a_))
put("ovRAdpMin", min(x_[0] for x_ in _b_)); put("ovRAdpMax", max(x_[0] for x_ in _b_))
put("ovRAdpBelowN", int(sum(1 for x_ in _b_ if x_[0] < 1.0)), "{:d}"); put("ovRAdpN", len(_b_), "{:d}")
for N_, nm in ((96, "F"), (504, "A"), (1008, "C"), (2040, "B")):
    put(f"ovRAdp{nm}", ratio(Ovv, (_lk(4, N_), "HORAE-R"), (_lk(4, N_), "ALOHA-ADP")))
    put(f"ovRVw{nm}", mm(Ovv, (_lk(4, N_), "HORAE-R")), "{:.0f}")
    put(f"ovRGs{nm}", mm(Ovg, (_lk(4, N_), "HORAE-R")))
put("ovRAdpD", ratio(Ovv, (_lk(11, 2040), "HORAE-R"), (_lk(11, 2040), "ALOHA-ADP"))); put("ovRAdpE", ratio(Ovv, (_lk(22, 2040), "HORAE-R"), (_lk(22, 2040), "ALOHA-ADP")))
put("ovRFoneE", mm(Ovf, (_lk(22, 2040), "HORAE-R")), "{:.3f}")


for nm, l_ in (("A", "imp:2.0"), ("B", "imp:4.0"), ("C", "imp:8.0"), ("D", "impov:N1008"), ("E", "impov:N2040")):
    put(f"fairRRoutine{nm}", mm(It["age_active_routine"], (l_, "HORAE-R")), "{:.0f}")
    put(f"fairRCritical{nm}", mm(It["age_active_critical"], (l_, "HORAE-R")), "{:.0f}")
    put(f"fairRJain{nm}", mm(It["jain_deliv"], (l_, "HORAE-R")), "{:.2f}")
    put(f"fairRPninefive{nm}", mm(It["age_active_p95"], (l_, "HORAE-R")), "{:.0f}")


for nm, s_ in (("Zero", "0.0"), ("Quarter", "0.25"), ("Half", "0.5"), ("Tq", "0.75"), ("One", "1.0")):
    put(f"sy{nm}HoraeR", mm(Yv, (s_, "HORAE-R")), "{:.1f}"); put(f"sy{nm}HoraeRF", mm(Yf, (s_, "HORAE-R")), "{:.3f}")
_sr = [ratio(Yv, (s_, "HORAE-R"), (s_, "ALOHA-ADP")) for s_ in _lv]
put("syRAdpLo", min(_sr)); put("syRAdpHi", max(_sr))
_sr = [ratio(Yv, (s_, "HORAE-R"), (s_, "ADP-G")) for s_ in _lv]
put("syRAdpGLo", min(_sr)); put("syRAdpGHi", max(_sr))
_sr = [ratio(Yv, (s_, "HORAE-R"), (s_, "ALOHA")) for s_ in _lv]
put("syRAlohaLo", min(_sr)); put("syRAlohaHi", max(_sr))
_sr = [ratio(Yv, (s_, "HORAE-R"), (s_, "ATA")) for s_ in _lv]
put("syRAtaLo", min(_sr)); put("syRAtaHi", max(_sr))


_cr = {t_: ratio(Cv, (t_, "HORAE-R"), (t_, "ALOHA-ADP")) for t_ in CPSI}
_g = [v_ for t_, v_ in _cr.items() if not t_.startswith("burst")]
put("corrRLo", min(_g)); put("corrRHi", max(_g)); put("corrRBelowN", int(sum(1 for v_ in _g if v_ < 1.0)), "{:d}"); put("corrRN", len(_g), "{:d}")
put("corrRCommonFast", _cr["0.9:None:0.75"]); put("corrRLongFast", _cr["0.9:0.3:0.0"]); put("corrRLongCommonFast", _cr["0.9:0.3:0.75"])
for nm, m_ in (("One", "1"), ("Ten", "10"), ("Hun", "100"), ("All", "1000")):
    put(f"corrRBurst{nm}", _cr[f"burst:{m_}"])


_fr = {}
for n_ in FIELDS:
    tn = tag_[n_]
    for kind, kn in (("alert", "Al"), ("track", "Tr")):
        for l_, ln in (("0.5", "Half"), ("1.0", "One"), ("2.0", "Two"), ("4.0", "Four")):
            t_ = f"{n_}:{kind}:{l_}"
            _fr[t_] = ratio(Zv, (t_, "HORAE-R"), (t_, "ALOHA-ADP"))
            put(f"fr{tn}{kn}Ratio{ln}", _fr[t_])
            put(f"fr{tn}{kn}Vw{ln}", mm(Zv, (t_, "HORAE-R")))
            q_ = Zf if kind == "alert" else Zn
            put(f"fr{tn}{kn}Q{ln}", mm(q_, (t_, "HORAE-R")), "{:.3f}")
            put(f"fr{tn}{kn}Ata{ln}", ratio(Zv, (t_, "HORAE-R"), (t_, "ATA")))
            put(f"fr{tn}{kn}Rg{ln}", ratio(Zv, (t_, "HORAE-R"), (t_, "RG")))
            put(f"fr{tn}{kn}Aloha{ln}", ratio(Zv, (t_, "HORAE-R"), (t_, "ALOHA")))
            put(f"fr{tn}{kn}Horae{ln}", ratio(Zv, (t_, "HORAE-R"), (t_, "HORAE")))
        for l_, ln in (("1.0", "One"), ("2.0", "Two")):
            s_ = f"{n_}:{kind}:{l_}:shifted"
            put(f"fr{tn}{kn}Shift{ln}", ratio(Zv, (s_, "HORAE-R"), (s_, "ALOHA-ADP")))
put("frLo", min(_fr.values())); put("frHi", max(_fr.values()))
put("frBelowN", int(sum(1 for v_ in _fr.values() if v_ < 1.0)), "{:d}"); put("frN", len(_fr), "{:d}")
_u = {k_: v_ for k_, v_ in _fr.items() if k_.endswith(":1.0")}
put("frOneLo", min(_u.values())); put("frOneHi", max(_u.values()))
_abv = {k_: v_ for k_, v_ in _fr.items() if not k_.endswith(":0.5")}
put("frAbLo", min(_abv.values())); put("frAbHi", max(_abv.values()))
_hlf = {k_: v_ for k_, v_ in _fr.items() if k_.endswith(":0.5")}
put("frHalfLo", min(_hlf.values())); put("frHalfHi", max(_hlf.values()))
_a_ = [ratio(Zv, (t_, "HORAE-R"), (t_, "ALOHA")) for t_ in _fr]
put("frAlohaLo", min(_a_)); put("frAlohaHi", max(_a_))
_a_ = [ratio(Zv, (t_, "HORAE-R"), (t_, "ATA")) for t_ in _fr]
put("frAtaLo", min(_a_)); put("frAtaHi", max(_a_)); put("frAtaBelowN", int(sum(1 for v_ in _a_ if v_ < 1.0)), "{:d}")
_a_ = [ratio(Zv, (t_, "HORAE-R"), (t_, "RG")) for t_ in _fr]
put("frRgLo", min(_a_)); put("frRgHi", max(_a_)); put("frRgAboveN", int(sum(1 for v_ in _a_ if v_ >= 1.0)), "{:d}")


for nm, tag in (("rawIntel", "intel"), ("rawBOne", "beijing:S1"), ("rawBTwo", "beijing:S2"), ("rawBThree", "beijing:S3"), ("rawBFour", "beijing:S4")):
    put(f"x{nm}HoraeR", mm(Rv, (tag, "HORAE-R")), "{:.1f}"); put(f"x{nm}HoraeRF", mm(Rf, (tag, "HORAE-R")), "{:.3f}")
put("segAlertHoraeR", mm(Sv, ("alert", "HORAE-R")), "{:.1f}"); put("segAlertHoraeRF", mm(Sf, ("alert", "HORAE-R")), "{:.3f}")
put("segTrackHoraeR", mm(Sv, ("track", "HORAE-R")), "{:.1f}"); put("segTrackHoraeRF", mm(Sn, ("track", "HORAE-R")), "{:.3f}")
put("segAlertRatioR", ratio(Sv, ("alert", "HORAE-R"), ("alert", "ALOHA-ADP"))); put("segTrackRatioR", ratio(Sv, ("track", "HORAE-R"), ("track", "ALOHA-ADP")))
put("segAlertAlohaRatioR", ratio(Sv, ("alert", "HORAE-R"), ("alert", "ALOHA")))


for cfg, cn in (("beijing_alert", "Al"), ("beijing_track", "Tr"), ("intel_alert", "In")):
    for tau, tn in ((0.25, "Quarter"), (0.5, "Half"), (0.75, "Tq"), (1.0, "One"), (1.5, "OneFive"), (2.0, "Two")):
        put(f"opR{cn}{tn}", mm(Ov, (f"{cfg}:{tau}", "HORAE-R")))
for S_, nm in ((10, "Ten"), (5, "Five")):
    put(f"opRS{nm}One", mm(Ov, (f"beijing_alert_S{S_}:1.0", "HORAE-R"))); put(f"opRS{nm}Half", mm(Ov, (f"beijing_alert_S{S_}:0.5", "HORAE-R")))
    put(f"opRS{nm}Adp", mm(Ov, (f"beijing_alert_S{S_}:1.0", "ALOHA-ADP")))


LSW = A.load("loadsweep")
LSv = A.table(LSW, "vwaoi")
for ds, nm in (("intel", "Intel"), ("beijing", "Beij")):
    E_ = R["E2"][ds]
    _r = [np.mean(list(LSv[(f"{ds}:{s_}", "HORAE")].values())) for s_ in E_["S"]]
    
    assert all(abs(a_ - b_) < 1e-6*a_ for a_, b_ in zip(_r, E_["by_pol"]["HORAE"])), "the load sweep does not reproduce the stored base rule"
    _q = [np.mean(list(LSv[(f"{ds}:{s_}", "HORAE-R")].values())) for s_ in E_["S"]]
    _adp = [a_/b_ for a_, b_ in zip(E_["by_pol"]["ALOHA-ADP"], _q)]
    put(f"lsR{nm}Lo", min(_adp)); put(f"lsR{nm}Hi", max(_adp))
    i_ = E_["loads"].index(4)
    put(f"lsR{nm}Top", _adp[i_])
    i_ = E_["loads"].index(0.5)
    put(f"lsR{nm}Low", _adp[i_])


for t_, nm in (("None", "Pure"), ("6.0", "Six"), ("0.0", "Zero")):
    put(f"capHoraeR{nm}", mm(Kv, (t_, "HORAE-R"))); put(f"capHoraeR{nm}F", mm(Kf, (t_, "HORAE-R")), "{:.3f}")
_cp_ = [ratio(Kv, (t_, "HORAE-R"), (t_, "ALOHA-ADP")) for t_ in ("None", "9.0", "6.0", "3.0", "0.0")]
put("capRRatioLo", min(_cp_)); put("capRRatioHi", max(_cp_))


for nm, ds in (("Beij", "beijing"), ("Intel", "intel")):
    for tg in ("calib", "whole", "test"):
        put(f"ho{nm}{tg.capitalize()}R", ratio(Hv, (f"{ds}:{tg}", "HORAE-R"), (f"{ds}:{tg}", "ALOHA-ADP")))
        put(f"ho{nm}{tg.capitalize()}RF", mm(Hf, (f"{ds}:{tg}", "HORAE-R")), "{:.3f}")
for nm, cfg, T_ in (("Intel", "intel_alert", 1500), ("BeijAl", "beijing_alert", 6000), ("BeijTr", "beijing_track", 6000)):
    put(f"warm{nm}RatioShortR", ratio(Hv, (f"{cfg}:warm:5", "HORAE-R"), (f"{cfg}:warm:5", "ALOHA-ADP")))
    put(f"warm{nm}RatioLongR", ratio(Hv, (f"{cfg}:warm:{T_//2}", "HORAE-R"), (f"{cfg}:warm:{T_//2}", "ALOHA-ADP")))


for nm, cfg in (("Al", "beijing_alert"), ("Tr", "beijing_track"), ("In", "intel_alert")):
    put(f"fbR{nm}Ideal", mm(Fv, (f"{cfg}:ideal", "HORAE-R")))
    for k_, tg in (("LossA", "loss:0.1"), ("LossB", "loss:0.3"), ("LossC", "loss:0.5"), ("DelayA", "delay:1"), ("DelayB", "delay:5"), ("DelayC", "delay:20"), ("InconsA", "incons:0.1"), ("InconsB", "incons:0.3"),
                   ("SharedB", "shared:0.3"), ("DutyA", "duty:5"), ("DutyB", "duty:20"), ("DutyC", "duty:100"), ("MisB", "misalign:1.0"), ("Noack", "noack")):
        put(f"fbR{nm}{k_}", mm(Fv, (f"{cfg}:{tg}", "HORAE-R")))
        put(f"fbR{nm}{k_}Ratio", ratio(Fv, (f"{cfg}:{tg}", "HORAE-R"), (f"{cfg}:{tg}", "ALOHA-ADP")))
    put(f"fbR{nm}DutyBQ", mm(Ff if "alert" in cfg else Fn, (f"{cfg}:duty:20", "HORAE-R")), "{:.3f}")
put("bcRLossBAl", mm(Bv, ("beijing_alert:bc:loss:0.3", "HORAE-R"))); put("bcRLossBTr", mm(Bv, ("beijing_track:bc:loss:0.3", "HORAE-R")))
put("lowGainRAl", mm(Bv, ("beijing_alert:gain:20:0.02", "HORAE-R"))); put("lowGainRTr", mm(Bv, ("beijing_track:gain:20:0.02", "HORAE-R")))
put("guardRAlSmall", mm(Bv, ("beijing_alert:guard:0.1", "HORAE-R"))); put("guardRAlLarge", mm(Bv, ("beijing_alert:guard:0.5", "HORAE-R")))


for nm, cfg in (("In", "intel_alert"), ("Al", "beijing_alert"), ("Tr", "beijing_track"), ("TrW", "beijing_track_w"), ("AlW", "beijing_alert_w")):
    put(f"vaRTrue{nm}", mm(Vv, (f"{cfg}:true", "HORAE-R")))
    for k_, tg in (("NoiseC", "noise:2.0"), ("BiasB", "bias:1.0"), ("BiasC", "bias:2.0"), ("BitsA", "bits:1"), ("Inv", "invert"), ("Shuf", "shuffle")):
        put(f"vaR{k_}{nm}", mm(Vv, (f"{cfg}:{tg}", "HORAE-R")))
put("vaXBAlR", mm(Vv, ("beijing_alert:xnoise:1.0", "HORAE-R")))
for nm, cfg in (("Al", "beijing_alert"),):
    for k_, m_ in (("ThA", "-1.0"), ("ThB", "1.0")):
        t_ = f"{cfg}:thetaerr:{m_}"
        put(f"vaR{k_}{nm}", mm(Vv, (t_, "HORAE-R"))); put(f"vaR{k_}{nm}F", mm(Vf, (t_, "HORAE-R")), "{:.3f}")
for cfg, cn in (("beijing_alert", "Al"), ("beijing_track", "Tr"), ("intel_alert", "In")):
    for name, vals in GR.items():
        if (f"{cfg}:{name}:{vals[0]}", "HORAE-R") not in Sev:
            continue
        rr_ = [ratio(Sev, (f"{cfg}:{name}:{v_}", "HORAE-R"), (f"{cfg}:{name}:{v_}", "ALOHA-ADP")) for v_ in vals]
        put(f"seR{NM[name]}{cn}Lo", min(rr_)); put(f"seR{NM[name]}{cn}Hi", max(rr_))
for wc, nm_ in ((1.0, "A"), (8.0, "C"), (16.0, "D"), (32.0, "E")):
    t_ = f"beijing_track:w:{wc}:0.15"
    put(f"seRW{nm_}TrAdp", ratio(Sev, (t_, "HORAE-R"), (t_, "ALOHA-ADP"))); put(f"seRW{nm_}TrHorae", ratio(Sev, (t_, "HORAE-R"), (t_, "HORAE")))


put("gdRRawOne", mm(Gv, ("raw_S1:HORAE-R", "HORAE-R")), "{:.0f}"); put("gdRRawTwo", mm(Gv, ("raw_S2:HORAE-R", "HORAE-R")), "{:.1f}")
put("gdROverB", mm(Gv, ("over_S4_N2040:HORAE-R", "HORAE-R")), "{:.0f}"); put("gdRSyncOne", mm(Gv, ("sync1.0:HORAE-R", "HORAE-R")), "{:.0f}")


_thr = {}
MFw = A.table(A.load("meanfield"), "succ_after_warmup"); MFt = A.table(A.load("meanfield"), "T")
for N_, S_ in ((2000, 20), (500, 10), (5000, 20)):
    for sig in (0.0, 1.0, 1.5):
        c_ = mf_constants(N_, sig)
        for eps in (0.5, 1.0):
            for rf in (0.7, 0.85, 0.95, 1.0, 1.2):
                tag = f"thr:{N_}:{S_}:{sig}:{eps}:{rf}"
                
                J_ = mm(MFv, (tag, "HORAE")); Th_ = mm(MFw, (tag, "HORAE"))/mm(MFt, (tag, "HORAE"))
                pred = float(np.sum(c_**eps)*np.sum(c_**(1 - eps)))/(2*N_*Th_) + 0.5*float(np.mean(c_))
                lb = float(np.sum(np.sqrt(c_))**2)/(2*N_*Th_) + 0.5*float(np.mean(c_))
                _thr[(N_, S_, sig, eps, rf)] = (J_, pred, lb, Th_)
_best_rf = {}
for N_, S_ in ((2000, 20), (500, 10), (5000, 20)):
    for sig in (0.0, 1.0, 1.5):
        for eps in (0.5, 1.0):
            _best_rf[(N_, S_, sig, eps)] = _thr[(N_, S_, sig, eps, 0.95)]
_rr = [v_[0]/v_[1] for v_ in _best_rf.values()]
put("mfThrLo", min(_rr), "{:.3f}"); put("mfThrHi", max(_rr), "{:.3f}")
_v = _best_rf[(2000, 20, 1.0, 0.5)]
put("mfThrRt", _v[0], "{:.1f}"); put("mfThrRtToLB", _v[0]/_v[2]); put("mfThrRtRate", _v[3])
_v = _best_rf[(2000, 20, 1.0, 1.0)]
put("mfThrLin", _v[0], "{:.1f}"); put("mfThrLinToLB", _v[0]/_v[2])
_lines = []
for eps, lab in ((0.5, r"Threshold on $\sqrt{c}A$"), (1.0, r"Threshold on $cA$")):
    cells = [lab]
    for sig in (0.0, 1.0, 1.5):
        v_ = _best_rf[(2000, 20, sig, eps)]
        cells.append("%.1f & %.1f" % (v_[0], v_[1]))
    rr_ = [v_[0]/v_[1] for k_, v_ in _best_rf.items() if k_[3] == eps]
    cells.append("%.3f--%.3f" % (min(rr_), max(rr_)))
    _lines.append(" & ".join(cells) + r" \\")
TB["mf"] = TB["mf"] + "\n" + "\n".join(_lines)
put("hazOpsQuiet", OPSR["inactive"]["basic"], "{:d}"); put("hazOpsActive", OPSR["active"]["basic"], "{:d}"); put("hazCfgBytes", OPSR["config_bytes"], "{:d}")
put("hazCyclesActive", commas(EP.cycles(OPSR["active"]))); put("hazCyclesQuiet", commas(EP.cycles(OPSR["inactive"])))


put("frAboveN", int(sum(1 for v_ in _fr.values() if v_ >= 1.0)), "{:d}")
_a_ = {t_: ratio(Zv, (t_, "HORAE-R"), (t_, "ATA")) for t_ in _fr}
put("frAtaAboveN", int(sum(1 for v_ in _a_.values() if v_ >= 1.0)), "{:d}")
put("frAtaBehindPct", math.ceil(1000*max(1.0/v_ - 1 for v_ in _a_.values()))/10, "{:.1f}")
_a_ = [ratio(Zv, (t_, "HORAE-R"), (t_, "RG")) for t_ in _fr]
put("frRgBehindN", int(sum(1 for v_ in _a_ if v_ < 1.0)), "{:d}")
put("ovRAdpAboveN", int(sum(1 for x_ in _b_ if x_[0] >= 1.0)), "{:d}")
put("ovRGsMax", max(mm(Ovg, (_lk(S_, N_), "HORAE-R")) for S_ in (4, 11, 22, 44) for N_ in (96, 240, 504, 1008, 2040)))
put("ovActB", ACT[2040], "{:.0f}")
put("ovRVwFs", mm(Ovv, (_lk(4, 96), "HORAE-R")), "{:.1f}")
_up = [s_ for s_ in _lv if mm(Yf, (s_, "ALOHA-ADP")) > mm(Yf, (s_, "HORAE-R"))]
put("syRFoneTop", float(_up[-1]), "{:g}")
put("syOneAlohaF", mm(Yf, ("1.0", "ALOHA")), "{:.3f}")
_Oe = A.table(op, "energy_per_dev_slot_uJ")
put("opREAlHalf", mm(_Oe, ("beijing_alert:0.5", "HORAE-R"))); put("opREAlOne", mm(_Oe, ("beijing_alert:1.0", "HORAE-R")))
for t, cfg in (("Intel", "intel_alert"), ("BeijAl", "beijing_alert"), ("BeijTr", "beijing_track")):
    a_, h_, o_ = mean(VW, cfg, "ALOHA"), mean(VW, cfg, "HORAE-R"), mean(VW, cfg, "MAXWEIGHT")
    put(f"gapR{t}", 100*(a_ - h_)/(a_ - o_), "{:.0f}")
_Gm = A.table(main, "G_over_S", variant)
for t, cfg in TAG.items():
    put(f"gos{t}HoraeR", mm(_Gm, (cfg, "HORAE-R")))
_l = [100*(mean(VW, "intel_alert", "HORAE-R")/mean(VW, "intel_alert", "ATA-T") - 1), 100*(mean(VW, "beijing_alert", "HORAE-R")/mean(VW, "beijing_alert", "AGE-R") - 1)]
put("hrLeadPct", math.ceil(10*max(_l))/10, "{:.1f}")
for cfg, cn in (("intel_alert", "In"), ("beijing_alert", "Al"), ("beijing_track", "Tr")):
    _k = (cfg, "ALOHA")
    put(f"enVm{cn}AlohaNb", mm(mainE["val_delivered"], _k)/(1e3*(mm(mainE["energy_J"], _k) - mm(mainE["listen_J"], _k))), "{:.1f}")
_rr = [v_[0]/v_[1] for k_, v_ in _best_rf.items() if k_[3] == 1.0 and k_[2] > 0]
put("mfThrLinLo", min(_rr), "{:.2f}"); put("mfThrLinHi", max(_rr), "{:.2f}")
put("gdGRawTwo", mm(Gv, ("raw_S2:G", "HORAE")), "{:.1f}")


for t, cfg in HR_T.items():
    s = A.paired(VW[(cfg, "AGE-R")], VW[(cfg, "ALOHA-ADP")])
    put(f"ar{t}Adp", s["ratio"]); put(f"ar{t}AdpLo", s["lo"]); put(f"ar{t}AdpHi", s["hi"])


_words = {7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven"}
_b = [p for p in DEC if p != "HORAE-R" and mean(F1, "beijing_alert", p) > mean(F1, "beijing_alert", "HORAE-R")]
put("hrBeatFoneN", _words[len(_b)])
_b = [p for p in DEC if p != "HORAE-R" and mean(VW, "intel_alert", p) < mean(VW, "intel_alert", "HORAE-R")]
put("hrBeatVwIntelN", _words[len(_b)])
put("hrBeatVwIntelPct", math.ceil(1000*max(mean(VW, "intel_alert", "HORAE-R")/mean(VW, "intel_alert", p) - 1 for p in _b))/10, "{:.1f}")
_rc = ["HORAE-R", "AGE-R", "HORAE", "HORAE-G", "AGE", "VOI", "ATA", "ATA-T", "ALOHA-OPT", "ALOHA-ADP", "ADP-G"]
put("intelBandVwLo", min(mean(VW, "intel_alert", p) for p in _rc)); put("intelBandVwHi", max(mean(VW, "intel_alert", p) for p in _rc))
put("intelBandFLo", min(mean(F1, "intel_alert", p) for p in _rc), "{:.3f}"); put("intelBandFHi", max(mean(F1, "intel_alert", p) for p in _rc), "{:.3f}")
_a2 = mm(mainE["tx"], ("intel_alert", "HORAE"))/mm(mainE["tx"], ("intel_alert", "HORAE-R"))
put("hrIntelAttRatio", _a2, "{:.0f}")
_aa = [mm(mainE["tx"], ("intel_alert", p))/mm(mainE["tx"], ("intel_alert", "HORAE-R")) for p in ("HORAE", "HORAE-G", "AGE", "VOI", "ALOHA-ADP", "ADP-G")]
put("hrIntelAttLo", min(_aa), "{:.0f}"); put("hrIntelAttHi", max(_aa), "{:.0f}")
for cfg in ("intel_alert", "beijing_alert"):
    assert abs(mean(VW, cfg, "ATA") - mean(VW, cfg, "ATA-T")) < 1e-9, "the caption says that tuning selected the default threshold on the alert goals"
put("enFoInAgeR", mm(mainE["energy_per_dev_slot_uJ"], ("intel_alert", "AGE-R")))
put("enFoInAlohaNb", mm(mainE["energy_per_dev_slot_uJ"], ("intel_alert", "ALOHA"))*(1 - mm(mainE["listen_J"], ("intel_alert", "ALOHA"))/mm(mainE["energy_J"], ("intel_alert", "ALOHA"))))


_vm = {"In": 3*math.exp(-3*0.3/0.5), "Al": 3*math.exp(-3*4/8), "Tr": 0.45}  
_tn = {"In": "intel_alert", "Al": "beijing_alert", "Tr": "beijing_track"}
for cn in ("In", "Al", "Tr"):
    N_ = mm(mainE["N"], (_tn[cn], "HORAE-R")); S_ = CONFIGS[_tn[cn]]["S"]
    put(f"rw{cn}", math.ceil(2.0*N_/(S_*_vm[cn])), "{:d}")


Tg = A.table(TH, "G_over_S"); Tc = A.table(TH, "beta_at_ceil"); Tu = A.table(TH, "frac_unheard"); Tw = A.table(TH, "frac_wrong_end")
_pn = (("full", "Full"), ("nothresh", "NoTh"), ("noceil", "NoCe"))
for reg, nm, fld in (("aqs_t05", "Aqs", "aqs"), ("o3_t05", "Othree", "o3")):
    for pn, pm in _pn:
        put(f"ptHalf{nm}{pm}", mm(Tv, (f"part:{reg}:{pn}", "HORAE")))
    put(f"ptHalf{nm}Gs", mm(Tg, (f"part:{reg}:full", "HORAE")))
    put(f"ptHalf{nm}Adp", mm(Zv, (f"{fld}:track:0.5", "ALOHA-ADP")))
    assert mm(Tv, (f"part:{reg}:noceil", "HORAE")) < mm(Zv, (f"{fld}:track:0.5", "ALOHA-ADP")) < mm(Tv, (f"part:{reg}:full", "HORAE"))
    assert mm(Tc, (f"part:{reg}:full", "HORAE")) > 0.99, "the text says that the cap binds"
for cfg, cn in (("beijing_alert", "Al"), ("beijing_track", "Tr")):
    for s_, sn in ((1.0, "One"), (2.0, "Two")):
        for pn, pm in _pn + (("rt", "Rt"),):
            put(f"ptBias{cn}{sn}{pm}", mm(Tv, (f"part:{cfg}:bias{s_}:{pn}", "HORAE")))
    put(f"ptBias{cn}TwoUh", 100*mm(Tu, (f"part:{cfg}:bias2.0:full", "HORAE")), "{:.1f}")
put("vaRBiasBTrRatio", mm(Vv, ("beijing_track:base", "ALOHA-ADP"))/mm(Vv, ("beijing_track:bias:1.0", "HORAE-R")))
for te, nm in ((0.5, "Hi"), (-0.5, "Lo")):
    for pn, pm in _pn:
        put(f"ptIn{nm}{pm}", mm(Tv, (f"part:intel_alert:th{te}:{pn}", "HORAE"))); put(f"ptIn{nm}{pm}Q", mm(Tf, (f"part:intel_alert:th{te}:{pn}", "HORAE")), "{:.3f}")
        put(f"wrIn{nm}{pm}", 100*mm(Tw, (f"part:intel_alert:th{te}:{pn}", "HORAE")), "{:.1f}")
    put(f"wrIn{nm}Adp", 100*mm(Tw, (f"part:intel_alert:th{te}:adp", "ALOHA-ADP")), "{:.1f}")
    for p, pm in (("HORAE-R", "R"), ("HORAE", "H"), ("ALOHA-ADP", "A")):
        put(f"vaIn{nm}{pm}", mm(Vv, (f"intel_alert:thetaerr:{te}", p))); put(f"vaIn{nm}{pm}Q", mm(Vf, (f"intel_alert:thetaerr:{te}", p)), "{:.3f}")
for p, pm in (("HORAE-R", "R"), ("HORAE", "H"), ("ALOHA-ADP", "A")):
    put(f"vaInFar{pm}", mm(Vv, ("intel_alert:thetaerr:1.0", p)), "{:.0f}")
put("exAlEightEx", mm(Tv, ("exempt:beijing_alert:8.0:True", "HORAE"))); put("exAlEightNo", mm(Tv, ("exempt:beijing_alert:8.0:False", "HORAE")))
put("exAlEightExQ", mm(Tf, ("exempt:beijing_alert:8.0:True", "HORAE")), "{:.3f}"); put("exAlEightNoQ", mm(Tf, ("exempt:beijing_alert:8.0:False", "HORAE")), "{:.3f}")


_c1 = mf_constants(2000, 1.0)
_lbs = float(np.sum(np.sqrt(_c1))**2)/(2*2000*20/math.e) + 0.5*float(np.mean(_c1))
for rf, nm in ((0.7, "Seven"), (0.85, "Eight"), (0.95, "Nine"), (1.0, "Ten"), (1.2, "Twelve")):
    put(f"mfRtSe{nm}", _thr[(2000, 20, 1.0, 0.5, rf)][0]/_lbs)
put("mfLinSeven", _thr[(2000, 20, 1.0, 1.0, 0.7)][0], "{:.1f}"); put("mfLinSevenLB", _thr[(2000, 20, 1.0, 1.0, 0.7)][0]/_thr[(2000, 20, 1.0, 1.0, 0.7)][2])
_w = [_thr[(2000, 20, 1.5, 1.0, rf)][0] for rf in (0.7, 0.85, 0.95, 1.0, 1.2)]
put("mfLinWideLo", min(_w), "{:.1f}"); put("mfLinWideHi", max(_w), "{:.1f}")
put("mfHoraeWide", mm(MFv, ("2000:20:1.5:1.0:1.0", "HORAE")), "{:.1f}")
assert min(_w) > mm(MFv, ("2000:20:1.5:1.0:1.0", "HORAE")), "the text says that the threshold on cA is above HORAE at that spread at every multiple"
put("mfExcessOne", float(np.mean(_c1))/float(np.mean(np.sqrt(_c1)))**2)
put("mfThrLinOffPct", math.ceil(100*(1 - min(v_[0]/v_[1] for k_, v_ in _best_rf.items() if k_[3] == 1.0 and k_[2] > 0))), "{:d}")
_c2 = mf_constants(2000, 1.5)
put("mfExcessWide", float(np.mean(_c2))/float(np.mean(np.sqrt(_c2)))**2)


for nm, cfg in (("Al", "beijing_alert"), ("Tr", "beijing_track")):
    put(f"fb{nm}DelayD", mm(Fv, (f"{cfg}:delay:10", "HORAE"))); put(f"fb{nm}DelayDAdp", mm(Fv, (f"{cfg}:delay:10", "ALOHA-ADP"))); put(f"fbR{nm}DelayD", mm(Fv, (f"{cfg}:delay:10", "HORAE-R")))
    put(f"fbR{nm}DelayCGs", mm(Fg, (f"{cfg}:delay:20", "HORAE-R")))
    put(f"fbR{nm}DutyBListenCut", 100*(1 - mm(Fl, (f"{cfg}:duty:20", "HORAE-R"))/mm(Fl, (f"{cfg}:ideal", "HORAE-R"))), "{:.1f}")
put("fbAlDelayCGGs", mm(Fg, ("beijing_alert:delay:20", "HORAE-G")))
put("lowGainAdpAl", mm(Bv, ("beijing_alert:gain:20:0.02", "ALOHA-ADP")))


_Of = A.table(op, "alert_f1")
put("opRFAlHalf", mm(_Of, ("beijing_alert:0.5", "HORAE-R")), "{:.3f}"); put("opRFAlOne", mm(_Of, ("beijing_alert:1.0", "HORAE-R")), "{:.3f}")


_fa = [t_ for t_ in _fr if ":alert:" in t_]
put("frFoneAdpN", _words.get(sum(1 for t_ in _fa if mm(Zf, (t_, "ALOHA-ADP")) > mm(Zf, (t_, "HORAE-R"))), sum(1 for t_ in _fa if mm(Zf, (t_, "ALOHA-ADP")) > mm(Zf, (t_, "HORAE-R")))))
put("frFoneN", len(_fa), "{:d}")
_sp = [(max(Zv[(t_, "HORAE-R")].values()) - min(Zv[(t_, "HORAE-R")].values()))/mm(Zv, (t_, "HORAE-R")) for t_ in _fr]
put("frSpreadPct", math.ceil(100*max(_sp)), "{:d}")


for S_, nm in ((1, "One"), (2, "Two"), (3, "Three"), (4, "Four")):
    for p, pm in (("HORAE-R", "HoraeR"), ("HORAE", "Horae"), ("ALOHA-ADP", "Adp"), ("RG", "Rg")):
        put(f"xrawI{nm}{pm}", mm(Rv, (f"intel:S{S_}", p)), "{:.1f}")
put("rawIntelActive", LM["raw"]["intel"], "{:.1f}")


_m = [ratio(Cv, (f"0.99:0.3:{x_}", "HORAE"), (f"0.99:0.3:{x_}", "ALOHA-ADP")) for x_ in ("0.25", "0.5", "0.75")]
put("corrLongMidLo", min(_m)); put("corrLongMidHi", max(_m))
assert max(_m) < 1

_m = [ratio(Cv, (f"0.99:{d_}:0.75", "HORAE"), (f"0.99:{d_}:0.75", "ALOHA-ADP")) for d_ in ("None", "0.05")]
put("corrMidCommonLo", min(_m)); put("corrMidCommonHi", max(_m))
put("enFoAlAgeR", mm(mainE["energy_per_dev_slot_uJ"], ("beijing_alert", "AGE-R")))

_decf = ["HORAE-R", "HORAE", "HORAE-G", "AGE", "VOI", "ATA", "GOAL", "ALOHA", "ALOHA-ADP", "ADP-G"]
_tt = [t_ for t_ in _fr if ":track:" in t_]
put("frTrN", len(_tt), "{:d}")
put("frTrBestVwN", sum(1 for t_ in _tt if min(_decf, key=lambda p_: mm(Zv, (t_, p_))) == "HORAE-R"), "{:d}")
put("frTrBestQN", sum(1 for t_ in _tt if min(_decf, key=lambda p_: mm(Zn, (t_, p_))) == "HORAE-R"), "{:d}")


_z = EP.slot_charge_C("zigbee", 1.0, *rates("intel_alert", "HORAE"), ops=OPS, **EP.bits(CONFIGS["intel_alert"]["S"]))
put("lfzWakeShareIn", 100*_z["wake"]/_z["total"], "{:.0f}"); put("lfzRxShareIn", 100*_z["rx"]/_z["total"], "{:.0f}")

import copy as _copy
_P0 = _copy.deepcopy(EP.PROFILES); _S0, _C0 = EP.MCU_SLEEP_A, EP.CYCLES_EXP_LOG


def _lifes():
    out = {}
    for cfg in ("intel_alert", "beijing_alert", "beijing_track"):
        Bb_ = EP.bits(CONFIGS[cfg]["S"])
        for p, _lab in PE:
            a_, l_ = rates(cfg, p)
            out[(cfg, p)] = EP.lifetime_years("lora", 10.0, a_, l_, ops=OPSR if p in ("HORAE-R", "AGE-R") else OPS, **Bb_)
    return out


_ref = _lifes()
_worst = 0.0; _flips = []
for key, val in (("I_tx", 20e-3), ("I_tx", 87e-3), ("I_tx", 120e-3), ("I_rx", 10.8e-3), ("I_rx", 12.0e-3), ("sleep", 1.1e-6), ("sleep", 2.0e-6), ("cyc", 1000)):
    if key in ("I_tx", "I_rx"):
        EP.PROFILES["lora"][key] = val
    elif key == "sleep":
        EP.MCU_SLEEP_A = val
    else:
        EP.CYCLES_EXP_LOG = val
    L_ = _lifes()
    _worst = max(_worst, max(abs(L_[k_]/_ref[k_] - 1) for k_ in L_))
    for cfg in ("intel_alert", "beijing_alert", "beijing_track"):
        ks = [k_ for k_ in _ref if k_[0] == cfg]
        for i_, a_ in enumerate(ks):
            for b_ in ks[i_ + 1:]:
                
                if abs(_ref[a_]/_ref[b_] - 1) > 0.05 and (_ref[a_] > _ref[b_]) != (L_[a_] > L_[b_]):
                    _flips.append((key, val, a_, b_))
    EP.PROFILES.clear(); EP.PROFILES.update(_copy.deepcopy(_P0)); EP.MCU_SLEEP_A, EP.CYCLES_EXP_LOG = _S0, _C0
assert not _flips, "a comparison of lifetimes reverses inside the data sheets' operating points: %s" % _flips[:3]
put("lfSensPct", 100*_worst, "{:.0f}")
put("battRatedMah", "2{,}950")


put("hoTrSecondR", ratio(Hv, ("beijing_track:second", "HORAE-R"), ("beijing_track:second", "ALOHA-ADP"))); put("hoTrSecondH", ratio(Hv, ("beijing_track:second", "HORAE"), ("beijing_track:second", "ALOHA-ADP")))


for v_, nm in (("0.0", "Zero"), ("4.0", "Four"), ("16.0", "Sixteen")):
    put(f"seBo{nm}F", mm(Sef, (f"beijing_alert:boost:{v_}", "HORAE")), "{:.3f}")
put("seBoSixteenVw", mm(Sev, ("beijing_alert:boost:16.0", "HORAE")))
for v_, nm in (("1.0", "One"), ("8.0", "Eight")):
    put(f"seKa{nm}F", mm(Sef, (f"beijing_alert:kappa:{v_}", "HORAE")), "{:.3f}"); put(f"seKa{nm}R", ratio(Sev, (f"beijing_alert:kappa:{v_}", "HORAE"), (f"beijing_alert:kappa:{v_}", "ALOHA-ADP")))


for v_, nm in (("0.0", "Zero"), ("4.0", "Four"), ("16.0", "Sixteen")):
    put(f"seRBo{nm}F", mm(Sef, (f"beijing_alert:boost:{v_}", "HORAE-R")), "{:.3f}")
_adpb = [mm(Sef, (f"beijing_alert:boost:{v_}", "ALOHA-ADP")) for v_ in ("0.0", "1.0", "2.0", "4.0", "8.0", "16.0")]
assert max(_adpb) - min(_adpb) < 5e-4, "adaptive ALOHA's F1 moves with the bonus: %s" % _adpb
put("seBoAdpF", _adpb[0], "{:.3f}")
for v_, nm in (("1.0", "One"), ("8.0", "Eight")):
    put(f"seKa{nm}AdpF", mm(Sef, (f"beijing_alert:kappa:{v_}", "ALOHA-ADP")), "{:.3f}")
    put(f"seKaR{nm}F", mm(Sef, (f"beijing_alert:kappa:{v_}", "HORAE-R")), "{:.3f}")


_words2 = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
_bh = [p for p in DEC if p != "HORAE" and mean(F1, "beijing_alert", p) > mean(F1, "beijing_alert", "HORAE")]
assert "ALOHA-ADP" in _bh, _bh
put("hBeatFoneN", _words2[len(_bh)])


_short = {k_: v_[3]/(k_[1]/math.e) for k_, v_ in _best_rf.items()}
assert all(v_ >= 0.99 for k_, v_ in _short.items() if k_[0] in (2000, 5000) and k_[3] == 0.5), _short
assert any(v_ < 0.99 for k_, v_ in _short.items() if k_[3] == 1.0), _short
put("mfThrShortLo", min(_short.values()), "{:.2f}")


put("enFoTrRSavePct", 100*(1 - float(M["enFoTrHoraeR"])/float(M["enFoTrHorae"])), "{:.0f}")


env = dict(os.environ)
subprocess.run([sys.executable, os.path.join(CODE, "make_macros.py")], check=True, env=env, stdout=subprocess.DEVNULL)
legacy = open(os.path.join(ROOT, "macros.tex"), encoding="utf8").read()
legacy_defs = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}", legacy))
for k, v in legacy_defs.items():
    if k not in M:
        M[k] = v
os.makedirs(os.path.join(ROOT, "generated"), exist_ok=True)
with open(os.path.join(ROOT, "generated", "numbers.tex"), "w", encoding="utf8", newline="\n") as f:
    f.write("% generated by make_numbers.py\n")
    for k in sorted(M):
        f.write("\\newcommand{\\%s}{%s}\n" % (k, M[k]))
    for k in sorted(TB):
        f.write("\\newcommand{\\tab%s}{%%\n%s\n}\n" % (k, TB[k]))
print(len(M), "macros,", len(TB), "table bodies")
