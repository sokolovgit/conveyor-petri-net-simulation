from conveyor import ConveyorParams
from conveyor import build as build_model
from spn import Constant, Exponential, FromPart, Transition

from petrinet import pnml

X_A, ROW, Y1 = 420, 150, 170
X_ZIG, X_ACC, X_Q, X_PROC, X_N = X_A + 145, X_A + 270, X_A + 390, X_A + 510, X_A + 630
X_RET, Y_RET = X_A - 260, Y1 + 2 * ROW


def _row(i: int) -> float:
    return Y1 + ROW * (i - 1)


def _place_xy(name: str) -> tuple[float, float]:
    kind, i = name[0], int(name[1:]) if name[1:] else 0
    match kind:
        case "G":
            return X_A - 380, Y1 - 90
        case "A":
            return (X_A - 190, Y1 - 90) if i == 0 else (X_A, _row(i))
        case "B":
            return X_Q, _row(i)
        case "E":
            return X_Q, _row(i) + 70
        case "F":
            return X_PROC + 60, _row(i) - 62
        case "N":
            return X_N, _row(i)
        case "R":
            return X_RET - 110, Y_RET
    raise ValueError(f"невідома позиція {name}")


def _transition_xy(tr: Transition) -> tuple[float, float]:
    name = tr.name
    if name == "Надходження":
        return X_A - 280, Y1 - 90
    if name == "На стрічку":
        return X_A - 90, Y1 - 90
    if name == "Повернення":
        return X_RET, Y_RET
    kind, rest = name.split(" ", 1)
    i = int(rest.split("→")[0])
    match kind:
        case "Прийняття":
            return X_ACC, _row(i)
        case "Обробка":
            return X_PROC, _row(i)
        case "Переїзд":
            return X_ZIG, _row(i) + ROW / 2
    raise ValueError(f"невідомий перехід {name}")


def _delay(tr: Transition, t_ob: float) -> dict[str, float | str]:
    match tr.delay:
        case Constant(value):
            return {"delay": value}
        case Exponential(mean):
            return {"delay": mean, "dist": "exp"}
        case FromPart():
            return {"delay": t_ob, "dist": "exp"}
    raise TypeError(f"{tr.name}: затримку {tr.delay!r} не можна записати в PNML")


def build(n: int = 5, b: list[int] | None = None, lam: float = 4.0, t_ob: float = 1.0) -> pnml.Net:
    params = ConveyorParams(buffers=tuple(b or [0] * n), lam=lam, t_ob=t_ob)
    model = build_model(params)
    out = pnml.Net(f"conveyor_n{n}")
    places = {p.name: out.place(p.name, *_place_xy(p.name), mark=p.count) for p in model.net.places}
    for tr in model.net.transitions:
        t = out.transition(tr.name, *_transition_xy(tr), priority=tr.priority, **_delay(tr, t_ob))
        for p in tr.inputs:
            out.inp(places[p.name], t)
        for p in tr.outputs:
            out.out(t, places[p.name])
    return out
