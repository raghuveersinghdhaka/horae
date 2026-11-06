
from math import ceil, log2

FRAME_BITS = 280           

BATTERY_AH = 2.0
BATTERY_V = 3.0
BATTERY_J = BATTERY_AH*3600*BATTERY_V
MCU_HZ = 48e6
MCU_RUN_A_PER_HZ = 100e-6/1e6   
MCU_SLEEP_A = 1.4e-6            
CYCLES_EXP_LOG = 200
CYCLES_DIV = 14                 
CYCLES_RNG = 10


def lora_airtime(n_bytes, sf=7, bw=125e3, cr=1, preamble=8, crc=1, header=1):
    tsym = 2**sf/bw
    pl = 8 + max(ceil((8*n_bytes - 4*sf + 28 + 16*crc - 20*(1 - header))/(4*sf))*(cr + 4), 0)
    return (preamble + 4.25 + pl)*tsym


def zig_airtime(n_bytes):
    return (n_bytes + 6)*8/250e3         


#
PROFILES = {
    "zigbee": dict(name="IEEE 802.15.4", V=3.0, I_tx=17.4e-3, I_rx=18.8e-3, I_sleep=0.02e-6, airtime=zig_airtime,
                   
                   wake=((0.6e-3, 20e-6), (1.0e-3, 426e-6), (192e-6, 18.8e-3)), turn=((192e-6, 18.8e-3),)),
    "lora": dict(name="LoRa", V=3.3, I_tx=29e-3, I_rx=11.5e-3, I_sleep=0.2e-6, airtime=lora_airtime,
                 
                 wake=((250e-6, 1.6e-3), (60e-6, 5.8e-3)), turn=((60e-6, 5.8e-3),)),
}


def frame_s(p):
    return PROFILES[p]["airtime"](ceil(FRAME_BITS/8))


def beacon_s(p, bits):
    return PROFILES[p]["airtime"](ceil(bits/8))


def cycles(ops):
    return ops["add"] + ops["mul"] + ops["cmp"] + ops["div"]*CYCLES_DIV + (ops["exp"] + ops["log"])*CYCLES_EXP_LOG + ops["rng"]*CYCLES_RNG


def read_ops(path):
    out = {}
    for line in open(path):
        w = line.split()
        if w and w[0] in ("inactive", "active"):
            out[w[0]] = {w[i]: int(w[i + 1]) for i in range(1, len(w) - 1, 2)}
        elif w and w[0] == "state_bytes":
            out["state_bytes"] = int(w[1]); out["config_bytes"] = int(w[3])
    return out


def bits(S):
    
    return dict(header_bits=ceil(log2(S + 1)), map_bits=ceil(S*log2(3)))


def device_rates(policy, tx, succ, listens, n_slots_devices):
    
    a, l = tx/n_slots_devices, listens/n_slots_devices
    if policy == "RG":
        s = succ/n_slots_devices
        a = s + (a - s)/8.0
    if policy == "ALOHA":
        l = 0.0
    return a, l


def slot_charge_C(p, slot_s, attempts_per_slot, listens_per_slot, header_bits=6, map_bits=99, ops=None):
    
    P = PROFILES[p]
    t_f = frame_s(p); t_h = beacon_s(p, header_bits); t_m = beacon_s(p, map_bits)
    wakes = min(1.0, listens_per_slot + attempts_per_slot)
    q_tx = P["I_tx"]*t_f*attempts_per_slot
    q_rx = P["I_rx"]*(t_h*listens_per_slot + t_m*attempts_per_slot)
    q_wake = sum(t*i for t, i in P["wake"])*wakes + sum(t*i for t, i in P["turn"])*attempts_per_slot
    c = cycles(ops["inactive"]) + (cycles(ops["active"]) - cycles(ops["inactive"]))*attempts_per_slot if ops else 0.0
    t_cpu = c/MCU_HZ
    q_cpu = MCU_RUN_A_PER_HZ*MCU_HZ*t_cpu
    awake = (t_f + t_m)*attempts_per_slot + t_h*listens_per_slot + t_cpu + sum(t for t, _ in P["wake"])*wakes + sum(t for t, _ in P["turn"])*attempts_per_slot
    q_sleep = (P["I_sleep"] + MCU_SLEEP_A)*max(slot_s - awake, 0.0)
    return dict(total=q_tx + q_rx + q_wake + q_cpu + q_sleep, tx=q_tx, rx=q_rx, wake=q_wake, cpu=q_cpu, sleep=q_sleep)


def slot_energy_J(p, slot_s, attempts_per_slot, listens_per_slot, header_bits=6, map_bits=99, ops=None):
    
    V = PROFILES[p]["V"]
    q = slot_charge_C(p, slot_s, attempts_per_slot, listens_per_slot, header_bits=header_bits, map_bits=map_bits, ops=ops)
    return {k: V*v for k, v in q.items()}


def lifetime_years(p, slot_s, attempts_per_slot, listens_per_slot, ops=None, **kw):
    
    q = slot_charge_C(p, slot_s, attempts_per_slot, listens_per_slot, ops=ops, **kw)["total"]
    return BATTERY_AH*3600/(q/slot_s)/(365.25*86400)


if __name__ == "__main__":
    for p in PROFILES:
        print(p, "frame ms %.2f header ms %.2f map ms %.2f" % (1e3*frame_s(p), 1e3*beacon_s(p, 6), 1e3*beacon_s(p, 99)))
