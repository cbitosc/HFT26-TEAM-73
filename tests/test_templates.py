from visionguard.templates.boards import (
    normalize_pin_name,
    map_detector_label,
    get_wokwi_pin,
    PART_SPECS,
    ARDUINO_UNO_PINS,
    REFERENCE_PROJECTS
)

def test_normalize_pin_name():
    # Arduino Uno pins
    assert normalize_pin_name("arduino-uno", "D13") == "13"
    assert normalize_pin_name("arduino-uno", "d0") == "0"
    assert normalize_pin_name("arduino-uno", "gnd") == "GND.1"
    assert normalize_pin_name("arduino-uno", "GND.2") == "GND.2"
    assert normalize_pin_name("arduino-uno", "5v") == "5V"
    assert normalize_pin_name("arduino-uno", "3v3") == "3.3V"

    # LED pins
    assert normalize_pin_name("led", "anode") == "A"
    assert normalize_pin_name("led", "a") == "A"
    assert normalize_pin_name("led", "+") == "A"
    assert normalize_pin_name("led", "cathode") == "C"
    assert normalize_pin_name("led", "c") == "C"
    assert normalize_pin_name("led", "-") == "C"

    # Resistor pins
    assert normalize_pin_name("resistor", "pin1") == "1"
    assert normalize_pin_name("resistor", "p2") == "2"

    # Pushbutton pins
    assert normalize_pin_name("pushbutton", "1l") == "1.l"
    assert normalize_pin_name("pushbutton", "2r") == "2.r"

def test_map_detector_label():
    vg_type, wokwi_type = map_detector_label("led")
    assert vg_type == "led"
    assert wokwi_type == "wokwi-led"

    vg_type, wokwi_type = map_detector_label("smd_resistor")
    assert vg_type == "resistor"
    assert wokwi_type == "wokwi-resistor"

    vg_type, wokwi_type = map_detector_label("tactile_switch")
    assert vg_type == "pushbutton"
    assert wokwi_type == "wokwi-pushbutton"

    vg_type, wokwi_type = map_detector_label("unknown_blob")
    assert vg_type == "unknown"
    assert wokwi_type is None

def test_get_wokwi_pin():
    assert get_wokwi_pin("arduino-uno", "13") == "13"
    assert get_wokwi_pin("arduino-uno", "GND.1") == "GND.1"
    assert get_wokwi_pin("led", "A") == "A"
    assert get_wokwi_pin("led", "C") == "C"
    assert get_wokwi_pin("resistor", "1") == "1"
    assert get_wokwi_pin("pushbutton", "1.l") == "1.l"

def test_reference_projects():
    assert "led-blink-pin13" in REFERENCE_PROJECTS
    proj = REFERENCE_PROJECTS["led-blink-pin13"]
    assert "led" in proj.required_parts
    assert "resistor" in proj.required_parts
    assert len(proj.canonical_loops) >= 1

