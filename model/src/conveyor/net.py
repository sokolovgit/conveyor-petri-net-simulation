from spn import Constant, Exponential, Net, Place, Transition

from conveyor.params import ConveyorParams

PRIORITY_ACCEPT = 1
SERVICE_STREAM = "service"


class ConveyorNet(Net):
    def __init__(self, params: ConveyorParams) -> None:
        n = params.n
        self.params = params
        self.gen = Place("G", 1)
        self.belt = tuple(Place(f"A{i}") for i in range(1, n + 1))
        self.queues = tuple(Place(f"B{i}") for i in range(1, n + 1))
        self.slots = tuple(Place(f"E{i}", b + 1) for i, b in enumerate(params.buffers, start=1))
        self.free = tuple(Place(f"F{i}", 1) for i in range(1, n + 1))
        self.done = tuple(Place(f"N{i}") for i in range(1, n + 1))
        self.returns = Place("R")

        service = (
            Constant(params.t_ob, SERVICE_STREAM)
            if params.service == "det"
            else Exponential(params.t_ob, SERVICE_STREAM)
        )
        self.arrive = Transition(
            "Надходження", [self.gen], [self.gen, self.belt[0]], Constant(1 / params.lam)
        )
        self.accept = tuple(
            Transition(
                f"Прийняття {i}",
                [self.belt[i - 1], self.slots[i - 1]],
                [self.queues[i - 1]],
                Constant(0.0),
                priority=PRIORITY_ACCEPT,
            )
            for i in range(1, n + 1)
        )
        self.process = tuple(
            Transition(
                f"Обробка {i}",
                [self.queues[i - 1], self.free[i - 1]],
                [self.free[i - 1], self.slots[i - 1], self.done[i - 1]],
                service,
            )
            for i in range(1, n + 1)
        )
        self.moves = tuple(
            Transition(
                f"Переїзд {i}→{i + 1}", [self.belt[i - 1]], [self.belt[i]], Constant(params.t_move)
            )
            for i in range(1, n)
        )
        self.back = Transition(
            "Повернення", [self.belt[-1]], [self.belt[0], self.returns], Constant(params.t_ret)
        )
        super().__init__(
            [self.gen, *self.belt, *self.queues, *self.slots, *self.free, *self.done, self.returns],
            [self.arrive, *self.accept, *self.process, *self.moves, self.back],
        )


def build(params: ConveyorParams) -> ConveyorNet:
    return ConveyorNet(params)
