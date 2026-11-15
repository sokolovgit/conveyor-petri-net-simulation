import heapq
import random
import sys
from collections import deque


def run(lam, t_ob, n=5, b=None, T=60000.0, warm=10000.0, seed=1, windows=4, keep=False):
    rng = random.Random(seed)
    b = b or [0] * n
    busy = [False] * n
    buf = [deque() for _ in range(n)]
    ev = []
    seq = 0

    def push(t, prio, kind, data):
        nonlocal seq
        seq += 1
        heapq.heappush(ev, (t, prio, seq, kind, data))

    push(0.0, 1, "new", None)
    born = {}
    ret = {}
    pid = 0
    W = []
    Wt = []
    R = []
    busy_area = 0.0
    last = 0.0
    nbusy = 0
    done = 0
    while ev:
        t, prio, _, kind, data = heapq.heappop(ev)
        if t > T:
            break
        if t > warm:
            busy_area += nbusy * (t - max(last, warm))
        last = t
        if kind == "new":
            pid += 1
            born[pid] = t
            ret[pid] = 0
            push(t + 1 / lam, 1, "new", None)
            kind, data = "reach", (pid, 0)
        if kind == "reach":
            p, i = data
            if not busy[i]:
                busy[i] = True
                nbusy += 1
                push(t + rng.expovariate(1 / t_ob), 2, "done", (p, i))
            elif len(buf[i]) < b[i]:
                buf[i].append(p)
            elif i < n - 1:
                push(t + 1.0, 0, "reach", (p, i + 1))
            else:
                ret[p] += 1
                push(t + 5.0, 0, "reach", (p, 0))
        elif kind == "done":
            p, i = data
            if t > warm:
                W.append(t - born[p])
                Wt.append(t)
                R.append(ret[p])
                done += 1
            del born[p]
            del ret[p]
            if buf[i]:
                q = buf[i].popleft()
                push(t + rng.expovariate(1 / t_ob), 2, "done", (q, i))
            else:
                busy[i] = False
                nbusy -= 1
    span = T - warm
    win = []
    for k in range(windows):
        lo, hi = warm + span * k / windows, warm + span * (k + 1) / windows
        xs = [w for w, tt in zip(W, Wt, strict=True) if lo <= tt < hi]
        win.append(sum(xs) / len(xs) if xs else float("nan"))
    res = {
        "W": sum(W) / len(W),
        "R": sum(R) / len(R),
        "K": busy_area / span,
        "Q": done / span,
        "in_sys": len(born),
        "win": win,
    }
    if keep:
        res |= {"Ws": W, "Wt": Wt, "Rs": R}
    return res


if __name__ == "__main__":
    for lam, t in [(4, 1.0), (4.5, 1.0), (4, 1.1), (4.8, 1.0), (4.9, 1.0), (4, 1.2), (4, 1.24)]:
        r = run(lam, t, T=float(sys.argv[1]) if len(sys.argv) > 1 else 60000)
        print(
            f"lam={lam} t={t} rho={lam * t / 5:.2f}  W={r['W']:.2f} R={r['R']:.3f} "
            f"K={r['K']:.3f} Q={r['Q']:.3f} left={r['in_sys']} "
            f"windows={[round(x, 1) for x in r['win']]}",
            flush=True,
        )
