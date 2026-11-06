
import json, os
from math import e, gamma, log, pi
import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))


def dominance(n=200000, seed=0):
    rng = np.random.default_rng(seed)
    viol = 0
    for _ in range(n):
        nn = rng.integers(2, 8); S = rng.integers(1, 6)
        h = np.sort(rng.lognormal(0, rng.uniform(0.01, 2), nn))
        kind = rng.integers(0, 3)
        if kind == 0:
            p = 1 - np.exp(-rng.lognormal(-1, 2)*h)
        elif kind == 1:
            p = np.sort(rng.random(nn))
        else:
            p = np.minimum(1, np.sort(rng.lognormal(-1, 1.5, nn)))
        u = np.full(nn, p.sum()/nn)

        def F(pp):
            one = 1 - pp/S
            return sum(h[i]*pp[i]*np.prod(np.delete(one, i)) for i in range(nn))
        if F(p) < F(u)*(1 - 1e-12):
            viol += 1
    return dict(n=n, violations=viol, pct=100.0*viol/n)


def fixed_point(Ss=(1, 2, 3, 4, 5, 8, 10, 11, 22, 44, 62)):
    out = {}
    for S in Ss:
        fl = min(0.5, S/4.0)

        def f(G):
            k = np.arange(S + 1)
            pm = binom.pmf(k, S, np.exp(-G/S))
            frac = np.minimum(np.maximum(k, fl)/S, 0.999)
            gh = -S*np.log(frac)
            return float((pm*(log(S) - np.log(np.maximum(gh, 1e-6)))).sum())
        out[str(S)] = brentq(f, 0.01*S, 30*S)/S
    return out


def burst(S=21, eta=0.25, ceil=50.0, low=1e-3):
    step = log(2*S)**(-eta)
    return dict(S=S, step=step, slots=log(ceil/low)/(eta*log(log(2*S))))


if __name__ == "__main__":
    R = dict(dominance=dominance(), fixed_point=fixed_point(), burst=burst(),
             g3=gamma(1 + 2/3)/(2*gamma(1 + 1/3)**2), g4=gamma(1.5)/(2*gamma(1.25)**2), one_const=(2/pi)*e)
    path = os.path.join(HERE, "results", "analysis_checks.json")
    json.dump(R, open(path, "w"), indent=1)
    print(json.dumps(R, indent=1))
