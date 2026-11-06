
import numpy as np, os

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

def _clean(X):
    
    X = X.copy()
    for j in range(X.shape[1]):
        c = X[:, j]; m = np.isnan(c)
        if m.any():
            idx = np.where(~m)[0]
            if len(idx) == 0:
                c[:] = np.nanmedian(X)
            else:
                c[:idx[0]] = c[idx[0]]
                for k in range(1, len(c)):
                    if np.isnan(c[k]):
                        c[k] = c[k-1]
        X[:, j] = c
    return X

def load_base(name):
    z = np.load(os.path.join(D, name), allow_pickle=True)
    return _clean(z["X"].astype(float)), z["xy"].astype(float)

def build_population(name, reps, seed=1, tmax=None, tstart=0):
    
    X0, xy0 = load_base(name)
    if tmax is not None:
        X0 = X0[:tmax]
    X0 = X0[tstart:]
    T, N = X0.shape
    rng = np.random.default_rng(seed)
    std = float(np.std(X0))
    Xs = [X0]; xys = [xy0]
    for r in range(1, reps):
        sh = int(rng.integers(1, T))
        Xr = np.roll(X0, sh, axis=0) + rng.normal(0, 0.03*std, X0.shape)
        Xs.append(Xr)
        off = np.array([(r % 6)*40.0, (r // 6)*40.0])
        xys.append(xy0 + off + rng.normal(0, 2, xy0.shape))
    return np.hstack(Xs), np.vstack(xys), std

def build_real_segment_population(name, L=1200, seed=2, tmax=None):
    
    X0, xy0 = load_base(name)
    if tmax is not None:
        X0 = X0[:tmax]
    T, N = X0.shape
    nwin = T // L
    std = float(np.std(X0))
    rng = np.random.default_rng(seed)
    cols = []; xys = []
    for w in range(nwin):
        seg = X0[w*L:(w+1)*L]                       
        for j in range(N):
            cols.append(seg[:, j])
            xys.append(xy0[j] + rng.normal(0, 1.0, 2) + np.array([w*3.0, 0.0]))
    X = np.stack(cols, axis=1)                      
    return X, np.array(xys, float), std

def importance_weights(N, frac_critical=0.15, w_critical=8.0, seed=3):
    
    rng = np.random.default_rng(seed)
    return np.where(rng.random(N) < frac_critical, w_critical, 1.0)

def active_fraction(X, goal, tmax=500):
    
    ld = X[0].copy(); af = []
    for t in range(1, min(tmax, len(X))):
        a = goal.active(X[t], ld); af.append(a.mean())
        ld = np.where(a, X[t], ld)
    return float(np.mean(af))
