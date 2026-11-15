from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

NS = "http://www.pnml.org/version-2009/grammar/pnml"
TOOL = {"tool": "PetriObjModel", "version": "2.3.0"}


@dataclass
class Place:
    id: str
    name: str
    x: float
    y: float
    mark: int = 0


@dataclass
class Transition:
    id: str
    name: str
    x: float
    y: float
    delay: float = 0.0
    dev: float = 0.0
    dist: str | None = None
    priority: int = 0
    probability: float = 1.0


@dataclass
class Arc:
    source: str
    target: str
    k: int = 1
    info: bool = False


@dataclass
class Net:
    name: str
    places: list[Place] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    arcs: list[Arc] = field(default_factory=list)

    def place(self, name: str, x: float, y: float, mark: int = 0) -> Place:
        p = Place(f"p{len(self.places) + 1}", name, x, y, mark)
        self.places.append(p)
        return p

    def transition(self, name: str, x: float, y: float, **params) -> Transition:
        t = Transition(f"t{len(self.transitions) + 1}", name, x, y, **params)
        self.transitions.append(t)
        return t

    def inp(self, p: Place, t: Transition, k: int = 1, info: bool = False) -> None:
        self.arcs.append(Arc(p.id, t.id, k, info))

    def out(self, t: Transition, p: Place, k: int = 1) -> None:
        self.arcs.append(Arc(t.id, p.id, k))

    def loop(self, p: Place, t: Transition, k: int = 1) -> None:
        self.inp(p, t, k)
        self.out(t, p, k)

    def write(self, path: Path) -> None:
        ET.register_namespace("", NS)
        root = ET.Element(f"{{{NS}}}pnml")
        net = _sub(root, "net", id=self.name, type="http://www.pnml.org/version-2009/grammar/ptnet")
        _text(net, "name", self.name)
        page = _sub(net, "page", id="page1")
        for p in self.places:
            el = _sub(page, "place", id=p.id)
            _text(el, "name", p.name)
            _position(el, p.x, p.y)
            if p.mark:
                _text(el, "initialMarking", str(p.mark))
        for t in self.transitions:
            el = _sub(page, "transition", id=t.id)
            _text(el, "name", label(t))
            _position(el, t.x, t.y)
            tool = _sub(el, "toolspecific", **TOOL)
            _sub(tool, "timeDelay").text = str(float(t.delay))
            _sub(tool, "delayMeanValue").text = str(float(t.delay))
            if t.dev:
                _sub(tool, "standardDeviation").text = str(float(t.dev))
            if t.priority:
                _sub(tool, "priority").text = str(t.priority)
            if t.probability != 1.0:
                _sub(tool, "probability").text = str(t.probability)
            if t.dist:
                _sub(tool, "distribution").text = t.dist
        for i, a in enumerate(self.arcs, start=1):
            el = _sub(page, "arc", id=f"a{i}", source=a.source, target=a.target)
            _text(el, "inscription", str(a.k))
            if a.info:
                _sub(_sub(el, "toolspecific", **TOOL), "informational").text = "true"
        ET.indent(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)


def label(t: Transition) -> str:
    return f"{t.name} (p={t.priority})" if t.priority else t.name


def _sub(parent: ET.Element, tag: str, **attrs: str) -> ET.Element:
    return ET.SubElement(parent, f"{{{NS}}}{tag}", attrs)


def _text(parent: ET.Element, tag: str, value: str) -> None:
    _sub(_sub(parent, tag), "text").text = value


def _position(parent: ET.Element, x: float, y: float) -> None:
    _sub(_sub(parent, "graphics"), "position", x=str(round(x)), y=str(round(y)))
