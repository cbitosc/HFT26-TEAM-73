from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

@dataclass
class PartSpec:
    type_name: str
    wokwi_type: str
    allowed_pins: Set[str]
    default_value: Optional[str] = None
    default_color: Optional[str] = None
    pin_aliases: Dict[str, str] = field(default_factory=dict)
    # Wokwi pin name mapping: {visionguard_pin: wokwi_pin}
    wokwi_pin_map: Dict[str, str] = field(default_factory=dict)

@dataclass
class TemplateSlot:
    name: str
    bbox: List[float] # normalized [x, y, w, h] in board coordinates (0.0 to 1.0)
    expected_type: str

@dataclass
class PinElectricalLimit:
    pin: str
    nominal_voltage: Optional[float] = None
    min_voltage: Optional[float] = None
    max_voltage: Optional[float] = None
    max_current_ma: Optional[float] = None
    is_power: bool = False
    is_ground: bool = False
    verified: bool = False

@dataclass
class ReferenceProject:
    name: str
    required_parts: List[str]
    # Required connections as undirected pairs of (part_ref, pin)
    # E.g. [("uno", "13"), ("r1", "1")]
    canonical_loops: List[List[Tuple[str, str]]]

# ---------------------------------------------------------------------------
# Supported Parts Table (Adding a part is a data change only)
# ---------------------------------------------------------------------------
PART_SPECS: Dict[str, PartSpec] = {
    "led": PartSpec(
        type_name="led",
        wokwi_type="wokwi-led",
        allowed_pins={"A", "C"},
        default_color="red",
        pin_aliases={
            "anode": "A",
            "a": "A",
            "+": "A",
            "pos": "A",
            "cathode": "C",
            "c": "C",
            "-": "C",
            "neg": "C",
            "k": "C",
        },
        wokwi_pin_map={"A": "A", "C": "C"}
    ),
    "resistor": PartSpec(
        type_name="resistor",
        wokwi_type="wokwi-resistor",
        allowed_pins={"1", "2"},
        default_value="220",
        pin_aliases={
            "p1": "1",
            "pin1": "1",
            "left": "1",
            "p2": "2",
            "pin2": "2",
            "right": "2",
        },
        wokwi_pin_map={"1": "1", "2": "2"}
    ),
    "pushbutton": PartSpec(
        type_name="pushbutton",
        wokwi_type="wokwi-pushbutton",
        allowed_pins={"1.l", "1.r", "2.l", "2.r"},
        pin_aliases={
            "1l": "1.l",
            "1_l": "1.l",
            "1": "1.l",
            "1r": "1.r",
            "1_r": "1.r",
            "2l": "2.l",
            "2_l": "2.l",
            "2": "2.l",
            "2r": "2.r",
            "2_r": "2.r",
        },
        wokwi_pin_map={
            "1.l": "1.l",
            "1.r": "1.r",
            "2.l": "2.l",
            "2.r": "2.r",
        }
    ),
    "arduino-uno": PartSpec(
        type_name="arduino-uno",
        wokwi_type="wokwi-arduino-uno",
        allowed_pins={
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13",
            "A0", "A1", "A2", "A3", "A4", "A5",
            "5V", "3.3V", "VIN", "IOREF", "RESET", "AREF",
            "GND.1", "GND.2", "GND.3"
        },
        pin_aliases={
            "d0": "0", "d1": "1", "d2": "2", "d3": "3", "d4": "4", "d5": "5",
            "d6": "6", "d7": "7", "d8": "8", "d9": "9", "d10": "10", "d11": "11",
            "d12": "12", "d13": "13",
            "gnd": "GND.1", "gnd1": "GND.1", "gnd2": "GND.2", "gnd3": "GND.3",
            "ground": "GND.1",
            "5v": "5V",
            "3v3": "3.3V", "3.3v": "3.3V",
            "vin": "VIN",
            "reset": "RESET", "rst": "RESET",
            "aref": "AREF",
            "ioref": "IOREF",
            "a0": "A0", "a1": "A1", "a2": "A2", "a3": "A3", "a4": "A4", "a5": "A5",
        },
        wokwi_pin_map={
            # NOTE: Wokwi pin names are from memory and kept in this table for easy fixes.
            "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
            "7": "7", "8": "8", "9": "9", "10": "10", "11": "11", "12": "12", "13": "13",
            "A0": "A0", "A1": "A1", "A2": "A2", "A3": "A3", "A4": "A4", "A5": "A5",
            "5V": "5V", "3.3V": "3.3V", "VIN": "VIN", "RESET": "RESET", "AREF": "AREF", "IOREF": "IOREF",
            "GND.1": "GND.1", "GND.2": "GND.2", "GND.3": "GND.3"
        }
    )
}

# ---------------------------------------------------------------------------
# Detector Label -> VisionGuard Type -> Wokwi Type Mapping Table
# ---------------------------------------------------------------------------
DETECTOR_LABEL_MAP: Dict[str, Tuple[str, str]] = {
    "led": ("led", "wokwi-led"),
    "led_light": ("led", "wokwi-led"),
    "diode_led": ("led", "wokwi-led"),
    "resistor": ("resistor", "wokwi-resistor"),
    "r": ("resistor", "wokwi-resistor"),
    "smd_resistor": ("resistor", "wokwi-resistor"),
    "button": ("pushbutton", "wokwi-pushbutton"),
    "pushbutton": ("pushbutton", "wokwi-pushbutton"),
    "push_button": ("pushbutton", "wokwi-pushbutton"),
    "switch": ("pushbutton", "wokwi-pushbutton"),
    "tactile_switch": ("pushbutton", "wokwi-pushbutton"),
    "arduino": ("arduino-uno", "wokwi-arduino-uno"),
    "arduino_uno": ("arduino-uno", "wokwi-arduino-uno"),
    "arduino-uno": ("arduino-uno", "wokwi-arduino-uno"),
    "pcb": ("arduino-uno", "wokwi-arduino-uno"),
}

# ---------------------------------------------------------------------------
# Arduino Uno Board Template Definition
# ---------------------------------------------------------------------------
ARDUINO_UNO_PINS: Set[str] = PART_SPECS["arduino-uno"].allowed_pins

ARDUINO_UNO_LIMITS: Dict[str, PinElectricalLimit] = {
    "5V": PinElectricalLimit(pin="5V", nominal_voltage=5.0, min_voltage=4.5, max_voltage=5.5, max_current_ma=500.0, is_power=True, verified=True),
    "3.3V": PinElectricalLimit(pin="3.3V", nominal_voltage=3.3, min_voltage=3.0, max_voltage=3.6, max_current_ma=50.0, is_power=True, verified=True),
    "VIN": PinElectricalLimit(pin="VIN", nominal_voltage=9.0, min_voltage=6.0, max_voltage=12.0, max_current_ma=1000.0, is_power=True, verified=True),
    "GND.1": PinElectricalLimit(pin="GND.1", nominal_voltage=0.0, min_voltage=0.0, max_voltage=0.0, is_ground=True, verified=True),
    "GND.2": PinElectricalLimit(pin="GND.2", nominal_voltage=0.0, min_voltage=0.0, max_voltage=0.0, is_ground=True, verified=True),
    "GND.3": PinElectricalLimit(pin="GND.3", nominal_voltage=0.0, min_voltage=0.0, max_voltage=0.0, is_ground=True, verified=True),
}
# Add digital pin limits (5.0V, 20mA nominal, 40mA absolute max)
for p in range(14):
    pin_str = str(p)
    ARDUINO_UNO_LIMITS[pin_str] = PinElectricalLimit(
        pin=pin_str,
        nominal_voltage=5.0,
        min_voltage=0.0,
        max_voltage=5.0,
        max_current_ma=20.0,
        verified=True
    )

for p in range(6):
    pin_str = f"A{p}"
    ARDUINO_UNO_LIMITS[pin_str] = PinElectricalLimit(
        pin=pin_str,
        nominal_voltage=5.0,
        min_voltage=0.0,
        max_voltage=5.0,
        max_current_ma=20.0,
        verified=True
    )

# Expected inspection slots on the Arduino Uno PCB
ARDUINO_UNO_SLOTS: List[TemplateSlot] = [
    TemplateSlot(name="slot_mcu", bbox=[0.40, 0.50, 0.30, 0.15], expected_type="ic"),
    TemplateSlot(name="slot_regulator", bbox=[0.08, 0.22, 0.14, 0.12], expected_type="regulator"),
    TemplateSlot(name="slot_usb", bbox=[0.04, 0.08, 0.18, 0.16], expected_type="connector"),
    TemplateSlot(name="slot_power_jack", bbox=[0.04, 0.65, 0.18, 0.18], expected_type="connector"),
]

# ---------------------------------------------------------------------------
# Reference Projects (e.g. "led-blink-pin13")
# ---------------------------------------------------------------------------
REFERENCE_PROJECTS: Dict[str, ReferenceProject] = {
    "led-blink-pin13": ReferenceProject(
        name="led-blink-pin13",
        required_parts=["led", "resistor"],
        canonical_loops=[
            # Loop option 1: 13 -> R1:1, R1:2 -> LED:A, LED:C -> GND
            [("uno", "13"), ("r1", "1"), ("r1", "2"), ("led1", "A"), ("led1", "C"), ("uno", "GND.1")],
            # Loop option 2 (resistor on cathode): 13 -> LED:A, LED:C -> R1:1, R1:2 -> GND
            [("uno", "13"), ("led1", "A"), ("led1", "C"), ("r1", "1"), ("r1", "2"), ("uno", "GND.1")],
        ]
    )
}

# ---------------------------------------------------------------------------
# Normalization Helpers
# ---------------------------------------------------------------------------
def normalize_pin_name(part_type: str, raw_pin: str) -> str:
    """
    Normalizes a pin name according to part type rules.
    E.g. D13 -> 13, GND -> GND.1, a -> A, pin1 -> 1.
    """
    clean = str(raw_pin).strip()
    lower = clean.lower()

    spec = PART_SPECS.get(part_type)
    if spec:
        # Check direct match with case-insensitivity
        for allowed in spec.allowed_pins:
            if allowed.lower() == lower:
                return allowed
        # Check alias dictionary
        if lower in spec.pin_aliases:
            return spec.pin_aliases[lower]

    # Arduino Uno generic fallback
    if part_type == "arduino-uno" or part_type == "board":
        uno_spec = PART_SPECS["arduino-uno"]
        if lower.startswith("d") and lower[1:].isdigit():
            cand = lower[1:]
            if cand in uno_spec.allowed_pins:
                return cand
        if lower in uno_spec.pin_aliases:
            return uno_spec.pin_aliases[lower]
        for allowed in uno_spec.allowed_pins:
            if allowed.lower() == lower:
                return allowed

    return clean

def map_detector_label(label: str) -> Tuple[str, Optional[str]]:
    """
    Maps a raw detector label to (visionguard_type, wokwi_type).
    Returns ("unknown", None) if not recognized.
    """
    normalized = label.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in DETECTOR_LABEL_MAP:
        return DETECTOR_LABEL_MAP[normalized]
    # Check substring matches
    for k, v in DETECTOR_LABEL_MAP.items():
        if k in normalized or normalized in k:
            return v
    return ("unknown", None)

def get_wokwi_pin(part_type: str, pin: str) -> str:
    """
    Translates VisionGuard pin name to Wokwi pin name.
    """
    spec = PART_SPECS.get(part_type)
    if spec and pin in spec.wokwi_pin_map:
        return spec.wokwi_pin_map[pin]
    return pin

