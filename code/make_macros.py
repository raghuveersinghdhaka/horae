#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "results.json")
OUT = os.path.join(HERE, os.pardir, "macros.tex")

R = json.load(open(RES, encoding="utf-8"))

M = {}          
SRC = {}        


def put(name, value, prov, fmt="{:.1f}"):
    
    s = value if isinstance(value, str) else fmt.format(value)
    M[name] = s
    SRC[name] = prov
    return s


def vw(exp, ds, goal, pol):
   
    node = R[exp][ds]
    if goal:
        node = node[goal]
    return node["results"][pol]["vwaoi"][0]


def met(exp, ds, goal, pol, key):
    node = R[exp][ds]
    if goal:
        node = node[goal]
    return node["results"][pol][key][0]


def closes(exp, ds, goal):
    
    h = vw(exp, ds, goal, "HORAE")
    a = vw(exp, ds, goal, "ALOHA")
    o = vw(exp, ds, goal, "MAXWEIGHT")
    return 100.0 * (a - h) / (a - o)



put("EzeroIntelHorae", vw("E0", "intel", None, "HORAE"), "E0.intel HORAE vwaoi")
put("EzeroIntelAloha", vw("E0", "intel", None, "ALOHA"), "E0.intel ALOHA vwaoi")
put("EzeroIntelOracle", vw("E0", "intel", None, "MAXWEIGHT"), "E0.intel MAXWEIGHT vwaoi")
put("EzeroIntelGain",
    vw("E0", "intel", None, "ALOHA") / vw("E0", "intel", None, "HORAE"),
    "E0.intel ALOHA/HORAE vwaoi", "{:.2f}")
put("EzeroIntelOracleGap",
    math.ceil(100.0 * (vw("E0", "intel", None, "HORAE") / vw("E0", "intel", None, "MAXWEIGHT") - 1.0)),
    "E0.intel HORAE excess over oracle, per cent, rounded UP so the text's "
    "'within N%' is a true bound (exact value 5.4%)", "{:d}")
put("EzeroIntelHoraeFone", met("E0", "intel", None, "HORAE", "alert_f1"),
    "E0.intel HORAE alert_f1", "{:.3f}")
put("EzeroIntelAlohaFone", met("E0", "intel", None, "ALOHA", "alert_f1"),
    "E0.intel ALOHA alert_f1", "{:.3f}")
put("EzeroIntelN", R["E0"]["intel"]["N"], "E0.intel N", "{:d}")
put("EzeroBeijingN", R["E0"]["beijing"]["N"], "E0.beijing N", "{:d}")


CELLS = [("Intel", "intel", "alert"), ("BeijAl", "beijing", "alert"), ("BeijTr", "beijing", "track")]
POLMAC = {"HORAE": "Horae", "AGE": "Age", "VOI": "Voi", "ATA": "Ata", "LTP": "Ltp",
          "ALOHA": "Aloha", "ALOHA-ADP": "Adp", "TDMA": "Tdma", "MAXWEIGHT": "Oracle"}
for tag, ds, goal in CELLS:
    for pol, pm in POLMAC.items():
        put(f"vw{tag}{pm}", vw("E1", ds, goal, pol), f"E1.{ds}.{goal} {pol} vwaoi")
    put(f"gain{tag}", vw("E1", ds, goal, "ALOHA") / vw("E1", ds, goal, "HORAE"),
        f"E1.{ds}.{goal} ALOHA/HORAE vwaoi", "{:.2f}")
    put(f"closes{tag}", closes("E1", ds, goal), f"E1.{ds}.{goal} oracle-gap closed, per cent", "{:.0f}")

for tag, ds, goal in CELLS:
    key = "alert_f1" if goal == "alert" else "track_nrmse"
    for pol, pm in (("HORAE", "Horae"), ("ATA", "Ata"), ("ALOHA", "Aloha")):
        put(f"dq{tag}{pm}", met("E1", ds, goal, pol, key), f"E1.{ds}.{goal} {pol} {key}", "{:.3f}")


put("dqBeijAlHoraeAbs", met("E1", "beijing", "alert", "HORAE", "alert_f1"),
    "E1.beijing.alert HORAE alert_f1, 2 dp for the abstract", "{:.2f}")
put("dqBeijAlAlohaAbs", met("E1", "beijing", "alert", "ALOHA", "alert_f1"),
    "E1.beijing.alert ALOHA alert_f1, 2 dp for the abstract", "{:.2f}")

put("ciIntelHorae", R["E1"]["intel"]["alert"]["results"]["HORAE"]["vwaoi"][1],
    "E1.intel.alert HORAE vwaoi 95% CI half-width", "{:.2f}")
put("gainBeijTrOverAge",
    vw("E1", "beijing", "track", "AGE") / vw("E1", "beijing", "track", "HORAE"),
    "E1.beijing.track AGE/HORAE vwaoi", "{:.2f}")
put("gainBeijTrOverAta",
    vw("E1", "beijing", "track", "ATA") / vw("E1", "beijing", "track", "HORAE"),
    "E1.beijing.track ATA/HORAE vwaoi", "{:.2f}")
put("gainLo", min(vw("E1", d, g, "ALOHA") / vw("E1", d, g, "HORAE") for _, d, g in CELLS),
    "min over E1 cells of ALOHA/HORAE vwaoi", "{:.1f}")
put("gainHi", max(vw("E1", d, g, "ALOHA") / vw("E1", d, g, "HORAE") for _, d, g in CELLS),
    "max over E1 cells of ALOHA/HORAE vwaoi", "{:.1f}")
put("closesLo", min(closes("E1", d, g) for _, d, g in CELLS),
    "min over E1 cells of oracle-gap closed", "{:.0f}")
put("closesHi", max(closes("E1", d, g) for _, d, g in CELLS),
    "max over E1 cells of oracle-gap closed", "{:.0f}")


for tag, ds in (("Intel", "intel"), ("Beij", "beijing")):
    E = R["E2"][ds]
    ratios = [E["by_pol"]["ALOHA"][i] / E["by_pol"]["HORAE"][i] for i in range(len(E["loads"]))]
    byload = dict(zip(E["loads"], ratios))
    put(f"load{tag}Half", byload[0.5], f"E2.{ds} ALOHA/HORAE at load 0.5", "{:.1f}")
    put(f"load{tag}Four", byload[4], f"E2.{ds} ALOHA/HORAE at load 4", "{:.1f}")
    put(f"load{tag}Lo", min(ratios), f"E2.{ds} min ALOHA/HORAE over the sweep", "{:.1f}")
    put(f"load{tag}Hi", max(ratios), f"E2.{ds} max ALOHA/HORAE over the sweep", "{:.1f}")
    put(f"load{tag}PeakAt", E["loads"][ratios.index(max(ratios))],
        f"E2.{ds} load at which ALOHA/HORAE peaks", "{:g}")


E4 = R["E4"]
put("impLo", E4["HORAE_over_AGE"][0], "E4 HORAE_over_AGE at spread 1", "{:.2f}")
put("impHi", E4["HORAE_over_AGE"][-1], "E4 HORAE_over_AGE at spread 16", "{:.2f}")
put("impSpreadLo", E4["spread"][0], "E4 lowest importance spread", "{:g}")
put("impSpreadHi", E4["spread"][-1], "E4 highest importance spread", "{:g}")


E5 = R["E5"]
put("scaleNlo", E5["N"][0], "E5 smallest population", "{:d}")
put("scaleNhi", E5["N"][-1], "E5 largest population", "{:d}")
put("scaleVwLo", E5["HORAE_vwaoi"][0], "E5 HORAE vwaoi at the smallest N")
put("scaleVwHi", E5["HORAE_vwaoi"][-1], "E5 HORAE vwaoi at the largest N")
put("scaleNplateau", E5["N"][2], "E5 N at which vwaoi has plateaued", "{:d}")
put("scaleCollLo", min(E5["HORAE_coll"]), "E5 min HORAE collision rate", "{:.2f}")
put("scaleCollHi", max(E5["HORAE_coll"]), "E5 max HORAE collision rate", "{:.2f}")


S6 = R["E6"]["sync"]
put("syncHi", max(S6["HORAE_over_ALOHA"]), "E6.sync max HORAE/ALOHA advantage", "{:.1f}")
put("syncLo", min(S6["HORAE_over_ALOHA"]), "E6.sync min HORAE/ALOHA advantage", "{:.1f}")
P6 = R["E6"]["ploss"]
put("plossHi", 100.0 * max(P6["p"]), "E6.ploss largest feedback-loss rate, per cent", "{:.0f}")
put("plossDrift", 100.0 * abs(P6["HORAE_vwaoi"][-1] / P6["HORAE_vwaoi"][0] - 1.0),
    "E6.ploss vwaoi change from no loss to the largest loss, per cent", "{:.0f}")


put("enIntelHorae", R["E7"]["intel"]["HORAE"]["energy"], "E7.intel HORAE energy_J", "{:.2f}")
put("enIntelAloha", R["E7"]["intel"]["ALOHA"]["energy"], "E7.intel ALOHA energy_J", "{:.2f}")
put("enBeijHorae", R["E7"]["beijing"]["HORAE"]["energy"], "E7.beijing HORAE energy_J")
put("enBeijAloha", R["E7"]["beijing"]["ALOHA"]["energy"], "E7.beijing ALOHA energy_J")
put("succRatio", met("E1", "intel", "alert", "HORAE", "succ") / met("E1", "intel", "alert", "ALOHA", "succ"),
    "E1.intel.alert HORAE/ALOHA successful deliveries")
put("valRatio", met("E1", "intel", "alert", "HORAE", "val_delivered") / met("E1", "intel", "alert", "ALOHA", "val_delivered"),
    "E1.intel.alert HORAE/ALOHA delivered value")
put("epsHorae", 1e6 * met("E1", "intel", "alert", "HORAE", "energy_per_succ"),
    "E1.intel.alert HORAE energy per delivery, microjoule", "{:.0f}")
put("epsAloha", 1e6 * met("E1", "intel", "alert", "ALOHA", "energy_per_succ"),
    "E1.intel.alert ALOHA energy per delivery, microjoule", "{:.0f}")


G8 = R["E8"]["G"]
put("ctrlGoverS", sum(G8[len(G8) // 2:]) / (len(G8) - len(G8) // 2) / R["E8"]["S"],
    "E8 mean G/S over the second half of the trace", "{:.2f}")


E9 = R["E9"]["by_pol"]
put("capVwPure", E9["HORAE"]["vwaoi"][0], "E9 HORAE vwaoi at the pure-collision point")
put("capVwFull", E9["HORAE"]["vwaoi"][-1], "E9 HORAE vwaoi at 0 dB capture")
put("capFoneLo", E9["HORAE"]["f1"][0], "E9 HORAE alert_f1 at the pure-collision point", "{:.2f}")
put("capFoneHi", E9["HORAE"]["f1"][-1], "E9 HORAE alert_f1 at 0 dB capture", "{:.2f}")
put("capAlohaFoneMax", max(E9["ALOHA"]["f1"]), "E9 ALOHA max alert_f1 over the sweep", "{:.2f}")
put("capAtaFoneMax", max(E9["ATA"]["f1"]), "E9 ATA max alert_f1 over the sweep", "{:.2f}")
put("capDbLo", min(x for x in R["E9"]["capture_db"] if x >= 0), "E9 strongest capture threshold, dB", "{:.0f}")


put("realN", R["E10"]["N"], "E10 number of real-segment sources", "{:d}")


def rvw(tag, pol):
    return R["E10"][tag]["results"][pol]["vwaoi"][0]


def rcloses(tag):
    h, a, o = rvw(tag, "HORAE"), rvw(tag, "ALOHA"), rvw(tag, "MAXWEIGHT")
    return 100.0 * (a - h) / (a - o)


put("realAlertGain", rvw("alert", "ALOHA") / rvw("alert", "HORAE"),
    "E10.alert ALOHA/HORAE vwaoi", "{:.2f}")
put("realTrackGain", rvw("track", "ALOHA") / rvw("track", "HORAE"),
    "E10.track ALOHA/HORAE vwaoi", "{:.2f}")
put("realAlertCloses", rcloses("alert"), "E10.alert oracle-gap closed, per cent", "{:.0f}")
put("realTrackCloses", rcloses("track"), "E10.track oracle-gap closed, per cent", "{:.0f}")
put("realFoneAloha", R["E10"]["alert"]["results"]["ALOHA"]["alert_f1"][0],
    "E10.alert ALOHA alert_f1", "{:.3f}")
put("realFoneHorae", R["E10"]["alert"]["results"]["HORAE"]["alert_f1"][0],
    "E10.alert HORAE alert_f1", "{:.3f}")
put("realImpOverAge", rvw("alert+imp", "AGE") / rvw("alert+imp", "HORAE"),
    "E10.alert+imp AGE/HORAE vwaoi", "{:.2f}")
put("realImpOverAloha", rvw("alert+imp", "ALOHA") / rvw("alert+imp", "HORAE"),
    "E10.alert+imp ALOHA/HORAE vwaoi", "{:.2f}")


E11 = R["E11"]
put("thE", E11["e"], "E11 Euler's number", "{:.3f}")
for i, S in enumerate(E11["S"]):
    put(f"thS{'abc'[i]}", S, f"E11 S at sweep point {i}", "{:d}")
    put(f"thMeas{'abc'[i]}", E11["succ_per_slot"][i], f"E11 measured deliveries per slot at S={S}", "{:.2f}")
    put(f"thPred{'abc'[i]}", E11["S_over_e"][i], f"E11 predicted S/e at S={S}", "{:.2f}")
    
    put(f"thRatio{'abc'[i]}", E11["succ_per_slot"][i] / E11["S_over_e"][i],
        f"E11 measured/idealised throughput at S={S} (retained for the cross-check)", "{:.2f}")
put("thRatioLo", min(m / p for m, p in zip(E11["succ_per_slot"], E11["S_over_e"])),
    "E11 min measured/predicted throughput", "{:.2f}")
put("thRatioHi", max(m / p for m, p in zip(E11["succ_per_slot"], E11["S_over_e"])),
    "E11 max measured/predicted throughput", "{:.2f}")
put("thGoverSLo", min(E11["meanGoverS"]), "E11 min mean G/S", "{:.2f}")
put("thGoverSHi", max(E11["meanGoverS"]), "E11 max mean G/S", "{:.2f}")
for i, (S, age) in enumerate(E11["sat_age"]):
    put(f"thSatS{'abc'[i]}", S, f"E11 S at saturated-age point {i}", "{:d}")
    put(f"thSatAge{'abc'[i]}", age, f"E11 saturated-source mean age at S={S}", "{:.2f}")


put("nSeeds", len(R["meta"]["seeds"]), "meta.seeds count", "{:d}")
put("nSeedsSweep", 3, "seeds used for the load, ablation, feedback-loss, capture and "
                      "real-segment studies (run_experiments.py seeds=[0,1,2])", "{:d}")
put("nSeedsScale", 2, "seeds used for the scale and synchrony sweeps "
                      "(run_experiments.py seeds=[0,1])", "{:d}")
put("noisePct", 3, "hdata.build_population measurement noise, per cent of field sd", "{:d}")
put("alertPct", 92, "run_experiments.make_goal alert percentile", "{:d}")


put("kappaVal", 3.0, "horae_sim.Goal kappa", "{:g}")
put("boostVal", 4.0, "horae_sim.Goal boost (wrong-side bonus b)", "{:g}")
put("vfloorVal", 0.05, "horae_sim.Goal v_floor (value floor v0)", "{:g}")
put("rhoIntel", 0.5, "run_experiments.make_goal intel alert tol (value width rho), degC", "{:g}")
put("rhoBeij", 8.0, "run_experiments.make_goal beijing alert tol (rho), ug/m3", "{:g}")
put("deltaIntel", 0.3, "run_experiments.make_goal intel alert act_tol (delta), degC", "{:g}")
put("deltaBeij", 4.0, "run_experiments.make_goal beijing alert act_tol (delta), ug/m3", "{:g}")
put("deltaTrackFrac", 0.45, "run_experiments.make_goal track act_tol as a fraction of sigma", "{:g}")
put("etaVal", 0.25, "horae_sim.simulate gain_up, the constant entry-controller gain", "{:g}")
put("actIntelPct", 100.0*R["E1"]["intel"]["alert"]["active_frac"],
    "E1.intel.alert measured active fraction, per cent", "{:.1f}")
put("actBeijPct", 100.0*R["E1"]["beijing"]["alert"]["active_frac"],
    "E1.beijing.alert measured active fraction, per cent", "{:.1f}")


for tag, ds, goal in CELLS:
    put(f"gain{tag}OverAdp", vw("E1", ds, goal, "ALOHA-ADP") / vw("E1", ds, goal, "HORAE"),
        f"E1.{ds}.{goal} ALOHA-ADP/HORAE vwaoi", "{:.2f}")
    put(f"gain{tag}OverLtp", vw("E1", ds, goal, "LTP") / vw("E1", ds, goal, "HORAE"),
        f"E1.{ds}.{goal} LTP/HORAE vwaoi", "{:.2f}")


EI = R["E1"]["intel"]["alert"]["results"]
EB = R["E1"]["beijing"]["alert"]["results"]
FB_BITS = 4                      # horae_sim.Radio fb_bits
E_ELEC = 50e-9                   # horae_sim.Radio E_elec, J/bit
put("fbBits", FB_BITS, "horae_sim.Radio fb_bits, the charged beacon width", "{:d}")
for tag, node in (("Intel", EI), ("Beij", EB)):
    for pol, pm in (("HORAE", "Horae"), ("ALOHA", "Aloha")):
        put(f"uJ{tag}{pm}", node[pol]["energy_per_dev_slot_uJ"][0],
            f"E1.{tag.lower()}.alert {pol} energy per device per slot, microjoule", "{:.3f}")
put("listenIntelHorae", EI["HORAE"]["listen_J"][0], "E1.intel.alert HORAE beacon-listen energy, J", "{:.3f}")
put("listenFracIntelPct", 100.0*EI["HORAE"]["listen_frac"][0],
    "E1.intel.alert beacon-listen share of HORAE total, per cent", "{:.0f}")
put("listenFracIntelAlohaPct", 100.0*EI["ALOHA"]["listen_frac"][0],
    "E1.intel.alert beacon-listen share of slotted-ALOHA total, per cent", "{:.0f}")
put("compIntelHorae", EI["HORAE"]["comp_J"][0], "E1.intel.alert HORAE compute energy, J", "{:.4f}")
put("compFracIntelPct", 100.0*EI["HORAE"]["comp_J"][0]/EI["HORAE"]["energy_J"][0],
    "E1.intel.alert compute share of HORAE total, per cent", "{:.1f}")

_aloha_nb = EI["ALOHA"]["energy_J"][0] - EI["ALOHA"]["listen_J"][0]
put("enIntelAlohaNoBeacon", _aloha_nb,
    "E1.intel.alert slotted-ALOHA energy with the beacon it does not need removed, J", "{:.2f}")
put("enRatioIntel", EI["HORAE"]["energy_J"][0]/EI["ALOHA"]["energy_J"][0],
    "E1.intel.alert HORAE/ALOHA total energy as charged", "{:.1f}")
put("enRatioIntelNoBeacon", EI["HORAE"]["energy_J"][0]/_aloha_nb,
    "E1.intel.alert HORAE/ALOHA total energy with ALOHA's beacon removed", "{:.1f}")


_S_int = R["E1"]["intel"]["alert"]["S"]
_idle_bits = math.ceil(math.log2(_S_int + 1))
_map_bits = math.ceil(_S_int * math.log2(3))
put("headlineS", _S_int, "E1.intel.alert S, the headline resource count", "{:d}")
put("overheadIdleBits", _idle_bits,
    f"ceil(log2(S+1)) at S={_S_int}: bits to convey the idle count the controller needs", "{:d}")
put("overheadMapBits", _map_bits,
    f"ceil(S*log2 3) at S={_S_int}: bits for the full per-resource ternary outcome map", "{:d}")
_N_int = R["E1"]["intel"]["alert"]["N"]
_slots_int = EI["HORAE"]["slots"][0]
_ntx_int = EI["HORAE"]["tx"][0]
_charged = EI["HORAE"]["listen_J"][0]

_fb_real = (_N_int*_slots_int*_idle_bits + _ntx_int*_map_bits) * E_ELEC

_fb_worst = _N_int*_slots_int*_map_bits * E_ELEC
put("enIntelHoraeRealFb", EI["HORAE"]["energy_J"][0] - _charged + _fb_real,
    "E1.intel.alert HORAE energy with an idle-count header for all devices and the "
    "outcome map only for transmitters", "{:.2f}")
put("enIntelHoraeFullMap", EI["HORAE"]["energy_J"][0] - _charged + _fb_worst,
    "E1.intel.alert HORAE energy if every device decoded the full outcome map every slot",
    "{:.2f}")


put("scaleLoad", R["E5"]["load"], "E5 offered load held fixed while N grows", "{:g}")
put("scaleSlo", R["E5"]["S"][0], "E5 S at the smallest population", "{:d}")
put("scaleShi", R["E5"]["S"][-1], "E5 S at the largest population", "{:d}")


E12B = R["E12b"]
_GROW = "22"
_g = E12B["by_S"][_GROW]
put("fixedS", int(_GROW), "E12b resource count pinned for the growth sweep", "{:d}")
put("fixedVwLo", _g["HORAE_vwaoi"][0], "E12b S=22 HORAE vwaoi at the smallest N")
put("fixedVwHi", _g["HORAE_vwaoi"][-1], "E12b S=22 HORAE vwaoi at the largest N")
put("fixedVwRatio", _g["HORAE_vwaoi"][-1]/_g["HORAE_vwaoi"][0],
    "E12b S=22 growth factor in HORAE vwaoi at fixed channel", "{:.1f}")
put("fixedNratio", _g["N"][-1]/_g["N"][0], "E12b population growth factor", "{:.0f}")
put("fixedLoadLo", _g["load"][0], "E12b S=22 offered load at the smallest N", "{:.2f}")
put("fixedLoadHi", _g["load"][-1], "E12b S=22 offered load at the largest N", "{:.1f}")
put("fixedCollLo", min(_g["HORAE_coll"]), "E12b S=22 min HORAE collision rate", "{:.2f}")
put("fixedCollHi", max(_g["HORAE_coll"]), "E12b S=22 max HORAE collision rate", "{:.2f}")
put("fixedGainLo", min(_g["HORAE_over_ALOHA"]), "E12b S=22 min ALOHA/HORAE vwaoi", "{:.1f}")
put("fixedGainHi", max(_g["HORAE_over_ALOHA"]), "E12b S=22 max ALOHA/HORAE vwaoi", "{:.1f}")


E14 = R["E14"]
put("impLoadLo", E14["loads"][0], "E14 lighter congestion point", "{:g}")
put("impLoadHi", E14["loads"][1], "E14 heavier congestion point", "{:g}")
put("impWcrit", E14["w_critical"], "E14 importance weight of the critical class", "{:g}")
put("impFracCritPct", 100.0*E14["frac_critical"], "E14 critical share of the population, per cent", "{:.0f}")
IMPPOL = (("HORAE", "Horae"), ("AGE", "Age"), ("VOI", "Voi"), ("ATA", "Ata"),
          ("ALOHA", "Aloha"), ("ALOHA-ADP", "Adp"))
IMPMET = (("rt", "age_active_routine", "{:.1f}"), ("cr", "age_active_critical", "{:.1f}"),
          ("p", "age_active_p95", "{:.0f}"), ("j", "jain_deliv", "{:.2f}"))
for li, lt in ((0, "A"), (1, "B")):
    for pol, pm in IMPPOL:
        for mt, key, fmt in IMPMET:
            put(f"imp{lt}{mt}{pm}", E14["by_pol"][pol][key][li],
                f"E14 load {E14['loads'][li]}x {pol} {key}", fmt)
_ihi = 1
_H = E14["by_pol"]["HORAE"]; _A = E14["by_pol"]["ALOHA"]
_D = E14["by_pol"]["ALOHA-ADP"]; _V = E14["by_pol"]["VOI"]
put("impRoutineHorae", _H["age_active_routine"][_ihi], "E14 heavy load HORAE routine-class age")
put("impCriticalHorae", _H["age_active_critical"][_ihi], "E14 heavy load HORAE critical-class age")
put("impRatioHorae", _H["age_active_routine"][_ihi]/_H["age_active_critical"][_ihi],
    "E14 heavy load HORAE routine/critical age ratio", "{:.1f}")
put("impRoutineAloha", _A["age_active_routine"][_ihi], "E14 heavy load slotted-ALOHA routine-class age")
put("impRoutineAdp", _D["age_active_routine"][_ihi], "E14 heavy load adaptive-ALOHA routine-class age")
put("impRoutineVsAloha", _A["age_active_routine"][_ihi]/_H["age_active_routine"][_ihi],
    "E14 heavy load how much fresher HORAE's routine class is than slotted ALOHA's", "{:.1f}")
put("impRoutineVsAdp", _D["age_active_routine"][_ihi]/_H["age_active_routine"][_ihi],
    "E14 heavy load how much fresher HORAE's routine class is than adaptive ALOHA's", "{:.1f}")
put("impRoutineVoi", _V["age_active_routine"][_ihi], "E14 heavy load VoI-only routine-class age")
put("impRatioVoi", _V["age_active_routine"][_ihi]/_V["age_active_critical"][_ihi],
    "E14 heavy load VoI-only routine/critical age ratio", "{:.1f}")
put("impJainHorae", _H["jain_deliv"][_ihi], "E14 heavy load HORAE Jain index", "{:.2f}")
put("impJainVoi", _V["jain_deliv"][_ihi], "E14 heavy load VoI-only Jain index", "{:.2f}")
put("impJainAdp", _D["jain_deliv"][_ihi], "E14 heavy load adaptive-ALOHA Jain index", "{:.2f}")
put("impVoiOverHorae", _V["vwaoi"][_ihi]/_H["vwaoi"][_ihi],
    "E14 heavy load VoI-only/HORAE vwaoi", "{:.1f}")


E13 = R["E13"]
_hi = 1   
_h = E13["by_pol"]["HORAE"]; _a = E13["by_pol"]["ALOHA"]
put("fairLoadHi", E13["loads"][_hi], "E13 heavier congestion point", "{:g}")
put("fairLoHorae", _h["age_active_lowval"][_hi], "E13 heavy load HORAE low-value-third age")
put("fairHiHorae", _h["age_active_highval"][_hi], "E13 heavy load HORAE high-value-third age")
put("fairBlowAloha", _a["age_active_lowval"][_hi], "E13 heavy load slotted-ALOHA low-value-third age")
put("fairBhighAloha", _a["age_active_highval"][_hi], "E13 heavy load slotted-ALOHA high-value-third age")


for i, S in enumerate(E11["S"]):
    _g = E11["meanGoverS"][i] * S
    _pred = _g * math.exp(-_g / S)
    put(f"thPredG{'abc'[i]}", _pred,
        f"E11 G e^(-G/S) at the measured attempt rate, S={S}", "{:.2f}")
    put(f"thRatioG{'abc'[i]}", E11["succ_per_slot"][i] / _pred,
        f"E11 measured/exact-prediction throughput at S={S}", "{:.2f}")
    put(f"thGoverS{'abc'[i]}", E11["meanGoverS"][i], f"E11 measured G/S at S={S}", "{:.2f}")
_ratG = [E11["succ_per_slot"][i] / (E11["meanGoverS"][i]*S*math.exp(-E11["meanGoverS"][i]))
         for i, S in enumerate(E11["S"])]
put("thRatioGLo", min(_ratG), "E11 min measured/exact-prediction throughput", "{:.2f}")
put("thRatioGHi", max(_ratG), "E11 max measured/exact-prediction throughput", "{:.2f}")


def _assert_code_constants():
    import horae_sim
    g = horae_sim.Goal("alert", tol=1.0, theta=0.0)
    r = horae_sim.Radio([[0.0, 0.0]])
    checks = [("kappa", g.kappa, 3.0), ("boost", g.boost, 4.0), ("v_floor", g.v_floor, 0.05),
              ("fb_bits", r.fb_bits, FB_BITS), ("E_elec", r.E_elec, E_ELEC)]
    for name, got, want in checks:
        assert got == want, f"macro constant {name}={want} but horae_sim gives {got}"


_assert_code_constants()


lines = ["% macros.tex --- GENERATED by code/make_macros.py from code/results/results.json.",
         "% Do not edit by hand: every value below is derived, not typed.",
         ""]
for name in M:
    lines.append(f"\\newcommand{{\\{name}}}{{{M[name]}}}%  <- {SRC[name]}")
lines.append("")
with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(lines))
print(f"wrote {os.path.normpath(OUT)}: {len(M)} macros")
