
import os
import numpy as np
import hdata

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")


def grid_xy(N, spacing=40.0):
    side = int(np.ceil(np.sqrt(N)))
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    xy = np.stack([gx.ravel()[:N], gy.ravel()[:N]], axis=1)*spacing
    return xy.astype(float)


def sync_population(level, popseed=7, tmax=4000, reps=40):
    X0, xy0 = hdata.load_base("beijing_field.npz")
    X0 = X0[:tmax]
    std0 = float(np.std(X0))
    T0, _ = X0.shape
    rng = np.random.default_rng(popseed)
    Xs = [X0]; xys = [xy0]
    for r in range(1, reps):
        sh = 0 if rng.random() < level else int(rng.integers(1, T0))
        Xs.append(np.roll(X0, sh, axis=0) + rng.normal(0, 0.03*std0, X0.shape))
        xys.append(xy0 + np.array([(r % 6)*40.0, (r//6)*40.0]) + rng.normal(0, 2, xy0.shape))
    return np.hstack(Xs), np.vstack(xys), std0


def corr_field(phi, ell, common, popseed=1, N=1000, T=3000, M=200):
    
    rng = np.random.default_rng(popseed)
    pos = rng.random((N, 2))

    def ar(k):
        e = rng.normal(size=(T, k))
        z = np.zeros((T, k)); z[0] = rng.normal(size=k)
        s = np.sqrt(1 - phi**2)
        for t in range(1, T):
            z[t] = phi*z[t - 1] + s*e[t]
        return z
    if ell is None:
        y = ar(N)
    else:
        om = rng.normal(size=(M, 2))/ell
        ph = rng.random(M)*2*np.pi
        F = np.sqrt(2.0/M)*np.cos(pos @ om.T + ph)
        y = ar(M) @ F.T
        y = y/np.std(y, axis=0, keepdims=True)
    g = ar(1)
    X = np.sqrt(common)*g + np.sqrt(1 - common)*y
    return X, pos*1000.0, float(np.std(X))


def burst_field(cluster, popseed=1, N=1000, T=3000, event_len=40, duty=0.08):
    
    rng = np.random.default_rng(popseed)
    labels = np.arange(N)//cluster
    k = labels.max() + 1
    rate = duty/event_len
    amp = np.zeros((T, k))
    cur = np.zeros(k)
    for t in range(T):
        start = rng.random(k) < rate
        cur = np.where(start, event_len*rng.exponential(1.0, k), np.maximum(cur - 1.0, 0.0))
        amp[t] = (cur > 0)*3.0
    e = np.zeros((T, N)); e[0] = rng.normal(size=N)
    s = np.sqrt(1 - 0.95**2)
    for t in range(1, T):
        e[t] = 0.95*e[t - 1] + s*rng.normal(size=N)
    X = amp[:, labels] + 0.5*e
    return X, grid_xy(N), float(np.std(X))


def load_field(name, tmax=None):
    z = np.load(os.path.join(D, f"{name}_field.npz"), allow_pickle=True)
    X = z["X"].astype(float)
    if tmax is not None:
        X = X[:tmax]
    return X, grid_xy(X.shape[1]), float(np.std(X))


def shifted_subset(name, n_base=12, popseed=1, tmax=None):
    
    X, xy, std = load_field(name, tmax=tmax)
    T, N = X.shape
    rng = np.random.default_rng(popseed)
    idx = rng.choice(N, n_base, replace=False)
    base = X[:, idx]
    reps = int(np.ceil(N/n_base))
    Xs = [base]
    for r in range(1, reps):
        sh = int(rng.integers(1, T))
        Xs.append(np.roll(base, sh, axis=0) + rng.normal(0, 0.03*std, base.shape))
    Xp = np.hstack(Xs)[:, :N]
    return Xp, grid_xy(N), float(np.std(Xp))


def synchrony_index(X, goal, tmax=3000):
    
    ld = X[0].copy()
    A = []
    for t in range(1, min(tmax, len(X))):
        a = goal.active(X[t], ld)
        A.append(a)
        ld = np.where(a, X[t], ld)
    A = np.array(A, float)
    m = A.mean(axis=0)
    denom = float(np.sum(m*(1 - m)))
    return float(np.var(A.sum(axis=1))/max(denom, 1e-12))
