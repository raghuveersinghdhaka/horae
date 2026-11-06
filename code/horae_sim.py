
from collections import deque
import numpy as np

BETA_FLOOR = 1e-12          
BETA_CEIL = 50.0            
BETA_CEIL_UNIFORM = 0.99    
IDLE_CAP = 0.999            


class Radio:
    
    def __init__(self, xy, packet_bits=200, fb_bits=4):
        self.xy = np.asarray(xy, float); self.N = len(xy)
        self.bs = np.mean(self.xy, 0)
        self.d = np.linalg.norm(self.xy - self.bs, axis=1)
        self.E_elec = 50e-9; self.eps_fs = 10e-12; self.eps_mp = 0.0013e-12
        self.d0 = 87.7; self.H_bits = 80; self.E_flop = 1e-9
        self.packet_bits = packet_bits
        self.fb_bits = fb_bits
        d = self.d
        self.per_bit = self.E_elec + np.where(d < self.d0, self.eps_fs*d*d, self.eps_mp*d**4)
        self.tx_cost = (self.H_bits + self.packet_bits)*self.per_bit

    def rx_energy(self):
        return (self.H_bits + self.packet_bits)*self.E_elec

    def listen_energy(self):
        return self.fb_bits*self.E_elec

    def compute_energy(self, flops):
        return self.E_flop*flops


class Goal:
    
    def __init__(self, kind, tol, theta=None, scale=1.0, kappa=3.0, boost=4.0,
                 v_floor=0.05, act_tol=None, w=None):
        self.kind = kind; self.tol = tol; self.theta = theta; self.scale = scale
        self.kappa = kappa; self.boost = boost; self.v_floor = v_floor
        self.act_tol = act_tol if act_tol is not None else tol
        self.w = w

    def value(self, x, last_delivered):
        if self.kind == "track":
            base = np.maximum(self.v_floor, np.abs(x - last_delivered)/self.scale)
        elif self.kind == "alert":
            v = self.kappa*np.exp(-np.abs(x - self.theta)/np.maximum(self.tol, 1e-9))
            mism = (x > self.theta) != (last_delivered > self.theta)
            v = np.where(mism, v + self.boost, v)
            base = np.maximum(self.v_floor, v)
        else:
            raise ValueError(self.kind)
        return base if self.w is None else base*self.w

    def active(self, x, last_delivered):
        if self.kind == "track":
            return np.abs(x - last_delivered) > self.act_tol
        if self.kind == "alert":
            near = np.abs(x - self.theta) < self.act_tol*self.kappa
            mism = (x > self.theta) != (last_delivered > self.theta)
            return near | mism
        raise ValueError(self.kind)

    def critical(self, x, last_delivered):
        
        if self.kind == "alert":
            return (x > self.theta) != (last_delivered > self.theta)
        return np.abs(x - last_delivered) > self.act_tol

    def error(self, x, last_delivered):
        
        if self.kind == "alert":
            return ((x > self.theta) != (last_delivered > self.theta)).astype(float)
        return np.abs(x - last_delivered)/self.scale*(np.abs(x - last_delivered) > self.act_tol)

    def collector_value(self, held):
        
        if self.kind == "alert":
            base = np.maximum(self.v_floor,
                              self.kappa*np.exp(-np.abs(held - self.theta)/np.maximum(self.tol, 1e-9)))
        else:
            base = np.full(held.shape, 1.0)
        return base if self.w is None else base*self.w


class StaticGoal(Goal):
    
    def __init__(self, c):
        Goal.__init__(self, "track", tol=0.0, scale=1.0, v_floor=1e-12, act_tol=-1.0)
        self.c = np.asarray(c, float)

    def value(self, x, last_delivered):
        return self.c

    def active(self, x, last_delivered):
        return np.ones(self.c.shape, bool)



FLOPS = {"HORAE": 6, "AGE": 4, "VOI": 4, "ALOHA": 0, "ALOHA-ADP": 3, "ALOHA-OPT": 0,
         "ATA": 3, "LTP": 8, "TDMA": 1, "MAXWEIGHT": 0, "MWQ": 0, "ACS": 0,
         "AOII": 6, "GOAL": 4, "IMP": 5, "IDX": 8, "RG": 6}

HAZARD_POLICIES = {"HORAE", "AGE", "VOI", "AOII", "IMP", "IDX"}
UNIFORM_POLICIES = {"ALOHA-ADP", "ATA", "GOAL"}
CENTRAL_POLICIES = {"TDMA", "MAXWEIGHT", "MWQ", "ACS", "RG"}


def resolve(att_idx, S, rng, powers=None, capture_db=None):
    
    if len(att_idx) == 0:
        return np.array([], int), S, 0, 0, np.array([], int)
    choice = rng.integers(0, S, size=len(att_idx))
    cnt = np.bincount(choice, minlength=S)
    single = cnt[choice] == 1
    delivered = att_idx[single]
    n_idle = int(np.sum(cnt == 0))
    n_succ = int(single.sum())
    multi = np.where(cnt >= 2)[0]
    n_coll = len(multi)
    if capture_db is not None and powers is not None and len(multi):
        extra = []
        for r in multi:
            members = att_idx[choice == r]
            p = powers[members]; top = np.argmax(p)
            rest = p.sum() - p[top]
            if rest <= 0 or 10*np.log10(p[top]/max(rest, 1e-12)) >= capture_db:
                extra.append(members[top]); n_succ += 1; n_coll -= 1
        if extra:
            delivered = np.concatenate([delivered, np.array(extra, int)])
    return delivered, n_idle, n_succ, n_coll, choice


def simulate(X, xy, policy, goal, S, seed=0, warmup=5,
             beta0=None, gain_up=0.25, capture_db=None, fading=False, ata_gamma=None,
             ltp_eta=0.10, ltp_pen=0.5, packet_bits=200, radio=None, p_loss=0.0, trace=False,
             beta_floor=BETA_FLOOR, beta_ceil=BETA_CEIL, idle_floor=None, target=1.0,
             aloha_p=None, exps=(1.0, 1.0),
             p_min=0.0, explore_every=0, hazard_cap=None,
             v_noise=0.0, v_bias=0.0, v_bits=0, v_mode="true", x_noise=0.0, theta_err=0.0,
             beacon_loss=0.0, common_loss=0.0, delay=0, idle_noise=0.0, duty=1,
             misalign=0.0, no_ack=False, freeze_idle=False, fast_down=1.0, acs_pow=3.0, acs_amin=0.2, rg_frac=0.2, rg_mini=8, beta_bcast=False, guard_t=0.0, detail=True,
             hz_thresh=0.0, hz_exp=1.0, hz_exempt=False, ceil_rel=0.0):

    T, N = X.shape
    rng = np.random.default_rng(seed)
    rng_fb = np.random.default_rng(10_000_019 + seed)
    if radio is None:
        radio = Radio(xy, packet_bits=packet_bits)
    uniform = policy in UNIFORM_POLICIES
    if idle_floor is None:
        idle_floor = min(0.5, S/4.0)
    ceil = BETA_CEIL_UNIFORM if uniform else beta_ceil
    if ceil_rel > 0 and not uniform:
        ceil = min(ceil, ceil_rel*S/max(N, 1))
    extra_ops = 1 if hz_thresh > 0 else 0
    floor = beta_floor

    last_deliv = X[0].copy()
    age = np.ones(N)
    dev_last = last_deliv.copy()
    dev_age = age.copy()
    if beta0 is None:
        beta0 = (S/max(N, 1)) if uniform else 0.02
    beta = np.full(N, float(beta0))
    beta_bs = float(beta0)
    if ata_gamma is None:
        ata_gamma = max(2.0, N/max(S, 1)*0.5)
    q = np.full(N, min(1.0, S/max(N, 1)))        
    t_err = np.zeros(N)                         
    bias = np.exp(v_bias*rng_fb.normal(size=N) - 0.5*v_bias**2) if v_bias > 0 else None
    phase = rng_fb.integers(0, duty, size=N) if duty > 1 else None
    offs = rng_fb.random(N)*misalign if misalign > 0 else None
    dev_goal = goal
    if theta_err != 0.0:
        dev_goal = Goal(goal.kind, goal.tol, theta=goal.theta + theta_err, scale=goal.scale,
                        kappa=goal.kappa, boost=goal.boost, v_floor=goal.v_floor,
                        act_tol=goal.act_tol, w=goal.w)
    separate = (x_noise > 0 or beacon_loss > 0 or common_loss > 0 or delay > 0 or duty > 1
                or misalign > 0 or no_ack or theta_err != 0.0)
    
    beta_u = np.full(N, S/max(N, 1)) if explore_every else None
    
    t_poll = np.ones(N); a_hat = np.ones(N); inno = np.ones(N)
    
    in_q = np.zeros(N, bool); rep_v = np.zeros(N); beta_r = min(1.0, S/max(N, 1))
    rr = 0

    tot_e = 0.0; comp_e = 0.0; n_tx = 0; n_succ_tot = 0; n_coll_tot = 0; n_succ_warm = 0
    vwaoi_sum = 0.0; age_sum = 0.0; maxage_sum = 0.0; vw_act_sum = 0.0
    val_succ = 0.0
    tp = fp = fn_ = tn = 0; track_num = 0.0; track_den = 0.0
    within_sum = 0.0
    tr_G = []; tr_coll = []; tr_succ = []; tr_beta = []
    tx_e = 0.0; rx_e = 0.0; listen_e = 0.0
    n_listen = 0
    age_act_sum = 0.0; age_p95_sum = 0.0; age_lo_sum = 0.0; age_hi_sum = 0.0; age_max_sum = 0.0
    n_act_slots = 0; n_ter_slots = 0
    deliv_count = np.zeros(N); attempt_count = np.zeros(N)
    ever_active = np.zeros(N, bool)
    wcls = None
    if getattr(goal, "w", None) is not None:
        _w = np.asarray(goal.w, float)
        if _w.size == N and _w.max() > _w.min():
            wcls = _w > _w.min()
    age_rt_sum = 0.0; age_cr_sum = 0.0; n_w_slots = 0
    sum_attempts = 0.0; sum_p = 0.0; at_floor = 0.0; at_ceil = 0.0; n_act_total = 0.0
    beta_sd_sum = 0.0; idle_sum = 0.0
    idle_hist = deque(maxlen=delay + 1)
    att_hist = deque(maxlen=delay + 1)
    deliv_hist = deque(maxlen=delay + 1)
    sent_hist = deque(maxlen=delay + 1)
    pending = None                               

    for t in range(1, T):
        x = X[t]
        v = goal.value(x, last_deliv)
        act = goal.active(x, last_deliv)
        comp_e += N*radio.compute_energy(FLOPS[policy] + extra_ops)
        n_idle_now = None

        if policy in CENTRAL_POLICIES:
            if policy == "TDMA":
                served = np.array([(rr + k) % N for k in range(min(S, N))], int)
                rr = (rr + min(S, N)) % N
                delivered = served[act[served]]
                n_succ = len(delivered); n_coll = 0
            elif policy in ("MAXWEIGHT", "MWQ"):
                cand = np.where(act)[0]
                if len(cand):
                    u = v[cand]*age[cand]
                    if policy == "MWQ":
                        u = u*(age[cand] + 1.5)
                    order = cand[np.argsort(-u)[:S]]
                else:
                    order = np.array([], int)
                delivered = order.astype(int)
                n_succ = len(order); n_coll = 0
            elif policy == "RG":
                
                S_req = max(1, int(round(rg_frac*S))); S_dat = max(1, S - S_req); M = rg_mini*S_req
                req = np.where((rng.random(N) < beta_r) & act & ~in_q)[0]
                if len(req):
                    ch_r = rng.integers(0, M, size=len(req))
                    cnt_r = np.bincount(ch_r, minlength=M)
                    ok = req[cnt_r[ch_r] == 1]
                    in_q[ok] = True; rep_v[ok] = v[ok]
                    idle_r = int(np.sum(cnt_r == 0))
                    e = float(radio.tx_cost[req].sum())/rg_mini
                    tot_e += e; tx_e += e; n_tx += len(req)
                    attempt_count[req] += 1
                else:
                    idle_r = M
                frac_r = min(max(idle_r, 0.5)/M, IDLE_CAP)
                beta_r = float(np.clip(beta_r*np.exp(gain_up*(np.log(M) - np.log(max(-M*np.log(frac_r), 1e-6)))), 1e-12, 1.0))
                queued = np.where(in_q)[0]
                if len(queued):
                    served = queued[np.argsort(-(rep_v[queued]*age[queued]))[:S_dat]]
                else:
                    served = np.array([], int)
                in_q[served] = False
                delivered = served[act[served]]
                n_succ = len(delivered); n_coll = 0
            else:  
                idx = goal.collector_value(last_deliv)*inno*a_hat*t_poll**acs_pow
                served = np.argpartition(-idx, min(S, N) - 1)[:min(S, N)]
                was_active = act[served]
                delivered = served[was_active]
                a_hat[served] += 0.05*(was_active.astype(float) - a_hat[served])
                a_hat = np.maximum(a_hat, acs_amin)
                if goal.kind == "track" and len(delivered):
                    inno[delivered] += 0.1*(np.abs(x[delivered] - last_deliv[delivered])/goal.scale
                                            - inno[delivered])
                    inno = np.maximum(inno, 0.05)
                t_poll += 1.0; t_poll[served] = 1.0
                n_succ = len(delivered); n_coll = 0
            if len(delivered):
                e = float(radio.tx_cost[delivered].sum()); r = radio.rx_energy()*len(delivered)
                tot_e += e + r; tx_e += e; rx_e += r; n_tx += len(delivered)
                attempt_count[delivered] += 1
            if policy in ("RG", "ACS"):
                e = N*radio.listen_energy(); tot_e += e; listen_e += e; n_listen += N
            active_idx = delivered
            n_idle_now = max(0, S - len(delivered))
            attempted = None
        else:
           
            if separate:
                x_dev = x + rng_fb.normal(0, x_noise, N) if x_noise > 0 else x
                v_dev = dev_goal.value(x_dev, dev_last)
                act_dev = dev_goal.active(x_dev, dev_last)
            else:
                x_dev = x; v_dev = v; act_dev = act
            if v_noise > 0 or bias is not None or v_bits > 0 or v_mode != "true":
                v_dev = v_dev.copy()
                if v_noise > 0:
                    v_dev = v_dev*np.exp(v_noise*rng_fb.normal(size=N) - 0.5*v_noise**2)
                if bias is not None:
                    v_dev = v_dev*bias
                if v_bits > 0:
                    lo = max(goal.v_floor, 1e-6)
                    hi = max((goal.kappa + goal.boost), 1.0)
                    if goal.w is not None:
                        hi = hi*float(np.max(goal.w))
                    lv = np.clip((np.log(np.maximum(v_dev, lo)) - np.log(lo))/(np.log(hi) - np.log(lo)), 0, 1)
                    lev = 2**v_bits - 1
                    v_dev = np.exp(np.log(lo) + np.round(lv*lev)/lev*(np.log(hi) - np.log(lo)))
                if v_mode == "invert":
                    v_dev = np.max(v_dev) + np.min(v_dev) - v_dev
                elif v_mode == "shuffle":
                    ia = np.where(act_dev)[0]
                    v_dev[ia] = v_dev[ia[rng_fb.permutation(len(ia))]]

            exploring = bool(explore_every) and (t % explore_every == 0)
            b = beta_u if exploring else beta
            if exploring:
                p = np.minimum(1.0, beta_u)*act_dev
            elif policy == "HORAE":
                h = v_dev*dev_age
                if hazard_cap is not None:
                    h = np.minimum(h, hazard_cap)
                p = (1.0 - np.exp(-b*h))*act_dev
            elif policy == "AGE":
                p = (1.0 - np.exp(-b*dev_age))*act_dev
            elif policy == "VOI":
                p = (1.0 - np.exp(-b*v_dev))*act_dev
            elif policy == "IMP":
                wi = goal.w if goal.w is not None else 1.0
                p = (1.0 - np.exp(-b*wi*dev_age))*act_dev
            elif policy == "IDX":
                p = (1.0 - np.exp(-b*np.power(v_dev, exps[0])*np.power(dev_age, exps[1])))*act_dev
            elif policy == "AOII":
                err = dev_goal.error(x_dev, dev_last)
                t_err = np.where(err > 0, t_err + 1.0, 0.0)
                wi = goal.w if goal.w is not None else 1.0
                p = (1.0 - np.exp(-b*wi*err*t_err))*act_dev
            elif policy == "ALOHA":
                p = np.full(N, min(1.0, S/max(N, 1)))*act_dev
            elif policy == "ALOHA-OPT":
                p = np.full(N, min(1.0, aloha_p))*act_dev
            elif policy == "ALOHA-ADP":
                p = np.minimum(1.0, b)*act_dev
            elif policy == "ATA":
                p = np.where(act_dev & (dev_age >= ata_gamma), np.minimum(1.0, b), 0.0)
            elif policy == "GOAL":
                p = np.where(act_dev & dev_goal.critical(x_dev, dev_last), np.minimum(1.0, b), 0.0)
            elif policy == "LTP":
                p = q*act_dev
            else:
                raise ValueError(policy)
            if hz_thresh > 0 and policy in ("HORAE", "AGE"):
                base_h = v_dev if policy == "HORAE" else 1.0
                elig = np.power(base_h, hz_exp)*dev_age >= hz_thresh*N/max(S, 1)
                if hz_exempt and goal.kind == "alert":
                    elig = elig | dev_goal.critical(x_dev, dev_last)
                p = p*elig
            if p_min > 0:
                p = np.where(act_dev, np.maximum(p, p_min), 0.0)

            attempted = rng.random(N) < p
            active_idx = np.where(attempted)[0]
            powers = None
            if capture_db is not None:
                base_pow = 1.0/np.maximum(radio.d, 1.0)**2
                powers = base_pow*rng.exponential(1.0, N) if fading else base_pow
            delivered, n_idle, n_succ, n_coll, choice = resolve(active_idx, S, rng,
                                                                powers=powers, capture_db=capture_db)
            n_idle_now = n_idle
            if misalign > 0:
                
                cur = dict(idx=active_idx, res=choice, deliv=delivered, n_idle=n_idle, x=x_dev.copy())
                if pending is None:
                    delivered = np.array([], int); n_succ = 0; n_coll = 0
                else:
                    kept = []
                    for i_ in pending["deliv"]:
                        r_ = pending["res"][pending["idx"] == i_][0]
                        hit = active_idx[choice == r_]
                        if len(hit) and np.any(offs[i_] - offs[hit] > guard_t):
                            continue
                        kept.append(i_)
                    
                    keep_now = []
                    for i_ in cur["deliv"]:
                        r_ = choice[active_idx == i_][0]
                        prev = pending["idx"][pending["res"] == r_]
                        if len(prev) and np.any(offs[prev] - offs[i_] > guard_t):
                            continue
                        keep_now.append(i_)
                    cur["deliv"] = np.array(keep_now, int)
                    delivered = np.array(kept, int)
                    n_succ = len(kept); n_coll = 0
                pending = cur
            if len(active_idx):
                e = float(radio.tx_cost[active_idx].sum())
                tot_e += e; tx_e += e; n_tx += len(active_idx)
                attempt_count[active_idx] += 1
            if len(delivered):
                r = radio.rx_energy()*len(delivered)
                tot_e += r; rx_e += r

            
            idle_hist.append(n_idle); att_hist.append(attempted.copy())
            mask_d = np.zeros(N, bool)
            if len(delivered):
                mask_d[delivered] = True
            deliv_hist.append(mask_d); sent_hist.append(x_dev if separate else None)
            listen = np.ones(N, bool)
            if duty > 1:
                listen = ((t + phase) % duty == 0)
                if len(att_hist) > delay:
                    listen = listen | att_hist[0]
            n_l = int(listen.sum())
            n_listen += n_l
            e = n_l*radio.listen_energy(); tot_e += e; listen_e += e

            if len(idle_hist) > delay:
                heard = listen
                if beacon_loss > 0:
                    heard = heard & (rng_fb.random(N) >= beacon_loss)
                if common_loss > 0 and rng_fb.random() < common_loss:
                    heard = np.zeros(N, bool)
                obs = idle_hist[0]
                if p_loss > 0:
                    obs = int(np.clip(obs + rng.normal(0, p_loss*max(S, 1)), 0, S))
                if idle_noise > 0:
                    obs = np.clip(obs + rng_fb.normal(0, idle_noise*S, N), 0, S)
                idle_frac = np.maximum(obs, idle_floor)/max(S, 1)
                G_hat = -S*np.log(np.minimum(idle_frac, IDLE_CAP))
                err_c = np.log(max(target*S, 1e-9)) - np.log(np.maximum(G_hat, 1e-6))
                if fast_down != 1.0:
                    err_c = np.where(np.asarray(obs) <= 0, err_c*fast_down, err_c)
                upd = heard & (np.asarray(obs) < S) if freeze_idle else heard
                if beta_bcast:
                    frac_b = min(max(idle_hist[0], idle_floor)/max(S, 1), IDLE_CAP)
                    err_b = np.log(max(target*S, 1e-9)) - np.log(max(-S*np.log(frac_b), 1e-6))
                    beta_bs = float(np.clip(beta_bs*np.exp(gain_up*err_b), floor, ceil))
                    beta = np.where(heard, beta_bs, beta)
                elif exploring:
                    beta_u = np.where(upd, np.clip(beta_u*np.exp(gain_up*err_c), 1e-12,
                                                   BETA_CEIL_UNIFORM), beta_u)
                else:
                    beta = np.where(upd, np.clip(beta*np.exp(gain_up*err_c), floor, ceil), beta)
                
                if separate:
                    if no_ack:
                        pass
                    else:
                        ack = heard & deliv_hist[0]
                        if ack.any():
                            dev_age = np.where(ack, float(delay), dev_age)   # +1 below
                            dev_last = np.where(ack, sent_hist[0], dev_last)
            if trace:
                tr_G.append(int(len(active_idx))); tr_coll.append(int(n_coll))
                tr_succ.append(int(n_succ)); tr_beta.append(float(np.median(beta)))
            sum_attempts += len(active_idx); sum_p += float(p.sum()); n_act_total += float(act_dev.sum())
            at_floor += float(np.mean(beta <= floor*1.0001))
            at_ceil += float(np.mean(beta >= ceil*0.9999))
            if separate or idle_noise > 0:
                beta_sd_sum += float(np.std(np.log(np.maximum(beta, 1e-300))))
            idle_sum += n_idle

        n_succ_tot += n_succ; n_coll_tot += n_coll
        succ_mask = np.zeros(N, bool)
        if len(delivered):
            succ_mask[delivered] = True
            val_succ += float(np.sum(v[delivered]))
            x_held = x
            if x_noise > 0 and policy not in CENTRAL_POLICIES:
                x_held = x_dev
            last_deliv[delivered] = x_held[delivered]
            deliv_count[delivered] += 1.0
        ever_active |= act
        if policy == "LTP" and len(active_idx):
            vsc = max(float(np.mean(v[act])) if act.any() else 1.0, 1e-6)
            r_ = np.where(succ_mask[active_idx], v[active_idx]/vsc, -ltp_pen)
            q[active_idx] = np.clip(q[active_idx]*np.exp(ltp_eta*r_), 0.01, 1.0)
        age = np.where(succ_mask, 1.0, age + 1.0)
        if separate and policy not in CENTRAL_POLICIES:
            dev_age = dev_age + 1.0
            if no_ack:
                att = attempted
                dev_age = np.where(att, 1.0, dev_age)
                dev_last = np.where(att, x_dev, dev_last)
        else:
            dev_age = age
            dev_last = last_deliv

        if t >= warmup:
            n_succ_warm += n_succ
            cost = v*age
            vwaoi_sum += float(np.sum(cost))/N
            vw_act_sum += float(np.sum(cost[act]))/N
            age_sum += float(np.mean(age)); maxage_sum += float(np.max(age))
            if act.any():
                a_act = age[act]
                age_act_sum += float(a_act.mean())
                age_max_sum += float(a_act.max())
                n_act_slots += 1
                if detail:
                    v_act = v[act]
                    age_p95_sum += float(np.percentile(a_act, 95))
                    k = len(v_act)//3
                    if k >= 1:
                        o = np.argsort(v_act, kind="stable")
                        age_lo_sum += float(a_act[o[:k]].mean())
                        age_hi_sum += float(a_act[o[-k:]].mean())
                        n_ter_slots += 1
                if wcls is not None:
                    rt = act & (~wcls); cr = act & wcls
                    if rt.any() and cr.any():
                        age_rt_sum += float(age[rt].mean())
                        age_cr_sum += float(age[cr].mean())
                        n_w_slots += 1
            within_sum += float(np.mean(np.abs(x - last_deliv) <= goal.tol))
            if goal.kind == "alert":
                td = x > goal.theta; cd = last_deliv > goal.theta
                tp += int(np.sum(td & cd)); fp += int(np.sum(~td & cd))
                fn_ += int(np.sum(td & ~cd)); tn += int(np.sum(~td & ~cd))
            else:
                track_num += float(np.sum((x - last_deliv)**2)); track_den += float(np.sum(x**2))

    tot_e += comp_e
    dur = max(T - warmup, 1)
    n_slots = max(T - 1, 1)
    assert abs((tx_e + rx_e + listen_e + comp_e) - tot_e) <= 1e-9*max(tot_e, 1.0), \
        "energy decomposition does not sum to the charged total"
    dc = deliv_count[ever_active]
    jain = float((dc.sum()**2)/(len(dc)*np.sum(dc**2))) if len(dc) and np.sum(dc**2) > 0 else float("nan")
    out = dict(policy=policy, goal=goal.kind, S=S, N=N, T=dur,
               energy_J=tot_e, comp_J=comp_e, tx=n_tx, succ=n_succ_tot, coll=n_coll_tot,
               coll_rate=n_coll_tot/max(n_tx, 1),
               vwaoi=vwaoi_sum/dur, vwaoi_active=vw_act_sum/dur,
               mean_age=age_sum/dur, max_age=maxage_sum/dur,
               val_delivered=val_succ, recon_within=within_sum/dur,
               energy_per_succ=tot_e/max(n_succ_tot, 1),
               energy_per_valbit=tot_e/max(val_succ, 1e-9),
               tx_J=tx_e, rx_J=rx_e, listen_J=listen_e, preamble_J=0.0,
               slots=n_slots, n_listen=n_listen,
               energy_per_dev_slot_uJ=1e6*tot_e/(max(N, 1)*n_slots),
               listen_frac=listen_e/max(tot_e, 1e-12),
               attempts_per_slot=sum_attempts/n_slots, G_over_S=sum_p/n_slots/max(S, 1),
               active_per_slot=n_act_total/n_slots, idle_per_slot=idle_sum/n_slots,
               beta_at_floor=at_floor/n_slots, beta_at_ceil=at_ceil/n_slots,
               beta_log_sd=beta_sd_sum/n_slots,
               age_active=age_act_sum/max(n_act_slots, 1),
               age_active_max=age_max_sum/max(n_act_slots, 1),
               age_active_p95=age_p95_sum/max(n_act_slots, 1),
               age_active_lowval=age_lo_sum/max(n_ter_slots, 1),
               age_active_highval=age_hi_sum/max(n_ter_slots, 1),
               age_active_routine=age_rt_sum/max(n_w_slots, 1) if wcls is not None else float("nan"),
               age_active_critical=age_cr_sum/max(n_w_slots, 1) if wcls is not None else float("nan"),
               jain_deliv=jain, n_ever_active=int(ever_active.sum()),
               deliv_min=float(dc.min()) if len(dc) else 0.0,
               frac_unheard=float(np.mean(dc == 0)) if len(dc) else 0.0, succ_after_warmup=n_succ_warm)
    if goal.kind == "alert":
        prec = tp/max(tp + fp, 1); rec = tp/max(tp + fn_, 1)
        out["alert_f1"] = 2*prec*rec/max(prec + rec, 1e-9)
        
        wrong_end = (X[-1] > goal.theta) != (last_deliv > goal.theta)
        out["frac_wrong_end"] = float(np.mean(wrong_end[ever_active])) if ever_active.any() else 0.0
        out["alert_acc"] = (tp + tn)/max(tp + tn + fp + fn_, 1)
    else:
        out["track_nrmse"] = float(np.sqrt(track_num/max(track_den, 1e-9)))
    if trace:
        out["trace_G"] = tr_G; out["trace_coll"] = tr_coll; out["trace_succ"] = tr_succ
        out["trace_beta"] = tr_beta
    return out
