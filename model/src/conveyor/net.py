from dataclasses import dataclass
from random import Random

from spn import Constant, FromPart, Net, Part, Place, Transition, exponential

from conveyor.params import ConveyorParams

PRIORITY_ACCEPT = 1


@dataclass(frozen=True, slots=True)
class NewPart:
    t_ob: float
    deterministic: bool = False

    def __call__(self, t: float, rng: Random) -> Part:
        tau = self.t_ob if self.deterministic else exponential(self.t_ob, rng)
        return Part(t_in=t, tau=tau)


@dataclass(frozen=True, slots=True)
class ConveyorNet:
    params: ConveyorParams
    net: Net
    belt: tuple[Place, ...]
    queues: tuple[Place, ...]
    slots: tuple[Place, ...]
    free: tuple[Place, ...]
    done: tuple[Place, ...]
    returns: Place
    arrive: Transition
    to_belt: Transition
    accept: tuple[Transition, ...]
    process: tuple[Transition, ...]
    moves: tuple[Transition, ...]
    back: Transition


def build(params: ConveyorParams) -> ConveyorNet:
    n = params.n
    gen = Place("G", 1)
    belt = tuple(Place(f"A{i}", typed=True) for i in range(n + 1))
    queues = tuple(Place(f"B{i}", typed=True) for i in range(1, n + 1))
    slots = tuple(Place(f"E{i}", b + 1) for i, b in enumerate(params.buffers, start=1))
    free = tuple(Place(f"F{i}", 1) for i in range(1, n + 1))
    done = tuple(Place(f"N{i}") for i in range(1, n + 1))
    returns = Place("R")

    arrive = Transition(
        "Надходження",
        [gen],
        [gen, belt[0]],
        Constant(1 / params.lam),
        source=NewPart(params.t_ob, params.service == "det"),
    )
    to_belt = Transition("На стрічку", [belt[0]], [belt[1]], Constant(0.0))
    accept = tuple(
        Transition(
            f"Прийняття {i}",
            [belt[i], slots[i - 1]],
            [queues[i - 1]],
            Constant(0.0),
            priority=PRIORITY_ACCEPT,
        )
        for i in range(1, n + 1)
    )
    process = tuple(
        Transition(
            f"Обробка {i}",
            [queues[i - 1], free[i - 1]],
            [free[i - 1], slots[i - 1], done[i - 1]],
            FromPart(),
        )
        for i in range(1, n + 1)
    )
    moves = tuple(
        Transition(f"Переїзд {i}→{i + 1}", [belt[i]], [belt[i + 1]], Constant(params.t_move))
        for i in range(1, n)
    )
    back = Transition("Повернення", [belt[n]], [belt[1], returns], Constant(params.t_ret))

    net = Net(
        [gen, *belt, *queues, *slots, *free, *done, returns],
        [arrive, to_belt, *accept, *process, *moves, back],
    )
    return ConveyorNet(
        params=params,
        net=net,
        belt=belt,
        queues=queues,
        slots=slots,
        free=free,
        done=done,
        returns=returns,
        arrive=arrive,
        to_belt=to_belt,
        accept=accept,
        process=process,
        moves=moves,
        back=back,
    )
