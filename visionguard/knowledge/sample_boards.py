"""
VisionGuard Sample Boards & Component Catalog
Matches the frontend Inspection Studio (localhost:3000/studio) exactly:
- Arduino Uno R3 (Reference Standard - 25 components)
- Arduino Uno R3 (Defect Variant B - 5 Defects)
- ESP32 DevKit V1 (Dual-Core SoC)
- Solderless Breadboard Circuit (LED Blink)
"""

from typing import Any, Dict, List

SAMPLE_BOARDS = {
    "arduino-uno-r3": {
        "id": "arduino-uno-r3",
        "name": "Arduino Uno R3 (Reference Standard)",
        "badge": "REFERENCE",
        "defect_count": 0,
        "description": "Standard official Arduino Uno R3 layout with 25 detected components and verified reference routing.",
        "detections_count": 25,
        "model": "SIMULATED-v1.4",
        "components": [
            {
                "id": "J1",
                "designator": "J1",
                "name": "J1 USB Type-B Receptacle",
                "raw_name": "usb_port",
                "category": "CONNECTOR",
                "conf": 0.991,
                "bbox_norm": [0.08, 0.15, 0.12, 0.18],
                "part_number": "USB-B-TH-R/A",
                "source": "SIM",
                "specs": {
                    "package": "Through-Hole Right-Angle",
                    "rating": "5V DC / 500mA",
                    "pins": 4,
                    "shielding": "Metal Shield Grounded"
                },
                "pinout_key": "usb-type-b"
            },
            {
                "id": "J2",
                "designator": "J2",
                "name": "J2 DC 2.1mm Power Barrel Jack",
                "raw_name": "power_jack",
                "category": "CONNECTOR",
                "conf": 0.988,
                "bbox_norm": [0.07, 0.65, 0.14, 0.22],
                "part_number": "DC-005-2.1MM",
                "source": "SIM",
                "specs": {
                    "center_pin": "Positive (7-12V DC)",
                    "sleeve": "Ground",
                    "diameter": "2.1mm Center Pin / 5.5mm Outer",
                    "package": "Through-Hole PCB Mount"
                },
                "pinout_key": "dc-barrel-jack"
            },
            {
                "id": "LED4",
                "designator": "LED4",
                "name": "LED4 Power Good 'ON' LED (Green)",
                "raw_name": "led",
                "category": "LED",
                "conf": 0.985,
                "bbox_norm": [0.65, 0.40, 0.03, 0.03],
                "part_number": "SMD-0805-GREEN",
                "source": "SIM",
                "specs": {
                    "color": "Green (525nm)",
                    "package": "SMD 0805",
                    "forward_voltage": "2.1V",
                    "current": "10mA"
                },
                "pinout_key": "led"
            },
            {
                "id": "U2",
                "designator": "U2",
                "name": "U2 ATmega328P 8-Bit MCU (DIP-28)",
                "raw_name": "microcontroller",
                "category": "MCU",
                "conf": 0.984,
                "bbox_norm": [0.45, 0.60, 0.28, 0.12],
                "part_number": "ATMEGA328P-PU",
                "source": "SIM",
                "specs": {
                    "core": "8-bit AVR RISC",
                    "flash": "32 KB",
                    "sram": "2 KB",
                    "clock": "16 MHz External",
                    "voltage": "5.0V",
                    "package": "DIP-28 Socketed"
                },
                "pinout_key": "atmega328p"
            },
            {
                "id": "U1",
                "designator": "U1",
                "name": "U1 ATmega16U2 USB-Serial Interface",
                "raw_name": "microcontroller",
                "category": "MCU",
                "conf": 0.978,
                "bbox_norm": [0.28, 0.25, 0.08, 0.08],
                "part_number": "ATMEGA16U2-MU",
                "source": "SIM",
                "specs": {
                    "package": "QFN-32",
                    "flash": "16 KB",
                    "clock": "16 MHz",
                    "role": "USB to Serial Converter"
                },
                "pinout_key": "atmega16u2"
            },
            {
                "id": "Y1",
                "designator": "Y1",
                "name": "Y1 16.000 MHz Crystal Oscillator",
                "raw_name": "crystal",
                "category": "CRYSTAL",
                "conf": 0.969,
                "bbox_norm": [0.38, 0.65, 0.06, 0.04],
                "part_number": "HC-49S-16.000MHZ",
                "source": "SIM",
                "specs": {
                    "frequency": "16.000000 MHz",
                    "load_capacitance": "18pF",
                    "package": "HC-49S Low Profile SMD"
                },
                "pinout_key": "crystal"
            },
            {
                "id": "U3",
                "designator": "U3",
                "name": "U3 AMS1117-5.0V Low-Dropout Voltage Regulator",
                "raw_name": "voltage_regulator",
                "category": "REGULATOR",
                "conf": 0.965,
                "bbox_norm": [0.18, 0.70, 0.08, 0.08],
                "part_number": "AMS1117-5.0",
                "source": "SIM",
                "specs": {
                    "output_voltage": "5.0V ±1.5%",
                    "max_current": "1.0 A",
                    "dropout": "1.1V",
                    "package": "SOT-223"
                },
                "pinout_key": "voltage-regulator-ams1117"
            },
            {
                "id": "U4",
                "designator": "U4",
                "name": "U4 LP2985-3.3V Ultra-Low Dropout Voltage Regulator",
                "raw_name": "voltage_regulator",
                "category": "REGULATOR",
                "conf": 0.958,
                "bbox_norm": [0.32, 0.75, 0.05, 0.04],
                "part_number": "LP2985-33DBVR",
                "source": "SIM",
                "specs": {
                    "output_voltage": "3.3V",
                    "max_current": "150 mA",
                    "package": "SOT-23-5"
                },
                "pinout_key": "voltage-regulator-lp2985"
            },
            {
                "id": "J3",
                "designator": "J3",
                "name": "J3 Female Header 10-Pin (Digital / PWM / AREF)",
                "raw_name": "gpio_header",
                "category": "CONNECTOR",
                "conf": 0.952,
                "bbox_norm": [0.48, 0.08, 0.22, 0.06],
                "part_number": "HDR-1X10-2.54MM",
                "source": "SIM",
                "specs": {
                    "pins": 10,
                    "pitch": "2.54mm",
                    "signals": "D8, D9, D10, D11, D12, D13, GND, AREF, SDA, SCL"
                },
                "pinout_key": "arduino-uno"
            },
            {
                "id": "J4",
                "designator": "J4",
                "name": "J4 Female Header 8-Pin (Digital PWM 0-7)",
                "raw_name": "gpio_header",
                "category": "CONNECTOR",
                "conf": 0.948,
                "bbox_norm": [0.72, 0.08, 0.18, 0.06],
                "part_number": "HDR-1X8-2.54MM",
                "source": "SIM",
                "specs": {
                    "pins": 8,
                    "pitch": "2.54mm",
                    "signals": "D0(RX), D1(TX), D2, D3, D4, D5, D6, D7"
                },
                "pinout_key": "arduino-uno"
            },
            {
                "id": "J5",
                "designator": "J5",
                "name": "J5 Female Header 8-Pin (Power Rail / Reset / Vin)",
                "raw_name": "gpio_header",
                "category": "CONNECTOR",
                "conf": 0.945,
                "bbox_norm": [0.45, 0.88, 0.18, 0.06],
                "part_number": "HDR-1X8-PWR",
                "source": "SIM",
                "specs": {
                    "pins": 8,
                    "pitch": "2.54mm",
                    "signals": "NC, IOREF, RESET, 3.3V, 5V, GND, GND, VIN"
                },
                "pinout_key": "arduino-uno"
            },
            {
                "id": "J6",
                "designator": "J6",
                "name": "J6 Female Header 6-Pin (Analog In A0-A5)",
                "raw_name": "gpio_header",
                "category": "CONNECTOR",
                "conf": 0.942,
                "bbox_norm": [0.65, 0.88, 0.14, 0.06],
                "part_number": "HDR-1X6-ANALOG",
                "source": "SIM",
                "specs": {
                    "pins": 6,
                    "pitch": "2.54mm",
                    "signals": "A0, A1, A2, A3, A4, A5"
                },
                "pinout_key": "arduino-uno"
            },
            {
                "id": "SW1",
                "designator": "SW1",
                "name": "SW1 Tactile Momentary Pushbutton (Reset)",
                "raw_name": "push_button",
                "category": "SWITCH",
                "conf": 0.938,
                "bbox_norm": [0.08, 0.38, 0.06, 0.06],
                "part_number": "TACT-SW-6X6MM",
                "source": "SIM",
                "specs": {
                    "function": "Microcontroller Hardware Reset (Active Low)",
                    "type": "SPST Momentary NO",
                    "package": "Through-Hole 6x6mm"
                },
                "pinout_key": "pushbutton"
            },
            {
                "id": "C1",
                "designator": "C1",
                "name": "C1 47uF / 25V Aluminum Electrolytic Capacitor",
                "raw_name": "capacitor",
                "category": "PASSIVE",
                "conf": 0.932,
                "bbox_norm": [0.15, 0.80, 0.06, 0.06],
                "part_number": "ELEC-47UF-25V-SMD",
                "source": "SIM",
                "specs": {
                    "capacitance": "47 µF",
                    "voltage_rating": "25V DC",
                    "type": "Polarized Aluminum Electrolytic",
                    "package": "SMD Can 6.3mm"
                },
                "pinout_key": "capacitor"
            },
            {
                "id": "C2",
                "designator": "C2",
                "name": "C2 47uF / 25V Aluminum Electrolytic Capacitor",
                "raw_name": "capacitor",
                "category": "PASSIVE",
                "conf": 0.930,
                "bbox_norm": [0.24, 0.72, 0.06, 0.06],
                "part_number": "ELEC-47UF-25V-SMD",
                "source": "SIM",
                "specs": {
                    "capacitance": "47 µF",
                    "voltage_rating": "25V DC",
                    "type": "Polarized Aluminum Electrolytic",
                    "package": "SMD Can 6.3mm"
                },
                "pinout_key": "capacitor"
            },
            {
                "id": "R1",
                "designator": "R1",
                "name": "R1 10k Ohm 0805 Pullup Resistor",
                "raw_name": "resistor",
                "category": "PASSIVE",
                "conf": 0.925,
                "bbox_norm": [0.12, 0.46, 0.03, 0.02],
                "part_number": "RES-10K-0805",
                "source": "SIM",
                "specs": {
                    "resistance": "10,000 Ω (10k)",
                    "tolerance": "±1%",
                    "power_rating": "125 mW",
                    "package": "SMD 0805"
                },
                "pinout_key": "resistor"
            },
            {
                "id": "R2",
                "designator": "R2",
                "name": "R2 1k Ohm 0805 Current Limiting Resistor",
                "raw_name": "resistor",
                "category": "PASSIVE",
                "conf": 0.921,
                "bbox_norm": [0.63, 0.42, 0.03, 0.02],
                "part_number": "RES-1K-0805",
                "source": "SIM",
                "specs": {
                    "resistance": "1,000 Ω (1k)",
                    "tolerance": "±1%",
                    "package": "SMD 0805"
                },
                "pinout_key": "resistor"
            },
            {
                "id": "R3",
                "designator": "R3",
                "name": "R3 1k Ohm 0805 Current Limiting Resistor",
                "raw_name": "resistor",
                "category": "PASSIVE",
                "conf": 0.918,
                "bbox_norm": [0.63, 0.46, 0.03, 0.02],
                "part_number": "RES-1K-0805",
                "source": "SIM",
                "specs": {
                    "resistance": "1,000 Ω (1k)",
                    "tolerance": "±1%",
                    "package": "SMD 0805"
                },
                "pinout_key": "resistor"
            },
            {
                "id": "D1",
                "designator": "D1",
                "name": "D1 M7 SMD General Purpose Rectifier Diode",
                "raw_name": "diode",
                "category": "DIODE",
                "conf": 0.915,
                "bbox_norm": [0.10, 0.58, 0.04, 0.03],
                "part_number": "M7-SMA-DIODE",
                "source": "SIM",
                "specs": {
                    "forward_current": "1.0 A",
                    "reverse_voltage": "1000V",
                    "package": "DO-214AC (SMA)",
                    "function": "Reverse polarity protection for Vin"
                },
                "pinout_key": "diode"
            },
            {
                "id": "F1",
                "designator": "F1",
                "name": "F1 Polyfuse 500mA Resettable PTC Fuse",
                "raw_name": "fuse",
                "category": "PASSIVE",
                "conf": 0.912,
                "bbox_norm": [0.15, 0.22, 0.04, 0.03],
                "part_number": "PTC-0805-500MA",
                "source": "SIM",
                "specs": {
                    "hold_current": "500 mA",
                    "trip_current": "1000 mA",
                    "voltage_rating": "6V DC",
                    "package": "SMD 1812"
                },
                "pinout_key": "fuse"
            },
            {
                "id": "J7",
                "designator": "J7",
                "name": "J7 ICSP 6-Pin Header for ATmega328P",
                "raw_name": "gpio_header",
                "category": "CONNECTOR",
                "conf": 0.908,
                "bbox_norm": [0.72, 0.62, 0.06, 0.08],
                "part_number": "HDR-2X3-2.54MM",
                "source": "SIM",
                "specs": {
                    "pins": 6,
                    "pitch": "2.54mm (2x3)",
                    "signals": "MISO, VCC, SCK, MOSI, RESET, GND"
                },
                "pinout_key": "icsp"
            },
            {
                "id": "J8",
                "designator": "J8",
                "name": "J8 ICSP 6-Pin Header for ATmega16U2",
                "raw_name": "gpio_header",
                "category": "CONNECTOR",
                "conf": 0.905,
                "bbox_norm": [0.32, 0.12, 0.06, 0.08],
                "part_number": "HDR-2X3-2.54MM",
                "source": "SIM",
                "specs": {
                    "pins": 6,
                    "pitch": "2.54mm (2x3)",
                    "signals": "MISO, VCC, SCK, MOSI, RESET, GND"
                },
                "pinout_key": "icsp"
            },
            {
                "id": "LED1",
                "designator": "LED1",
                "name": "LED1 Built-in User LED (Pin 13 - Amber)",
                "raw_name": "led",
                "category": "LED",
                "conf": 0.902,
                "bbox_norm": [0.55, 0.38, 0.03, 0.03],
                "part_number": "SMD-0805-AMBER",
                "source": "SIM",
                "specs": {
                    "color": "Amber / Yellow-Orange (590nm)",
                    "package": "SMD 0805",
                    "driver": "Op-Amp LMV358 Buffer via Pin 13"
                },
                "pinout_key": "led"
            },
            {
                "id": "LED2",
                "designator": "LED2",
                "name": "LED2 RX Status LED (Yellow)",
                "raw_name": "led",
                "category": "LED",
                "conf": 0.898,
                "bbox_norm": [0.42, 0.38, 0.03, 0.03],
                "part_number": "SMD-0805-YELLOW",
                "source": "SIM",
                "specs": {
                    "color": "Yellow",
                    "package": "SMD 0805",
                    "function": "Serial Data Receive Indicator"
                },
                "pinout_key": "led"
            },
            {
                "id": "LED3",
                "designator": "LED3",
                "name": "LED3 TX Status LED (Yellow)",
                "raw_name": "led",
                "category": "LED",
                "conf": 0.895,
                "bbox_norm": [0.42, 0.43, 0.03, 0.03],
                "part_number": "SMD-0805-YELLOW",
                "source": "SIM",
                "specs": {
                    "color": "Yellow",
                    "package": "SMD 0805",
                    "function": "Serial Data Transmit Indicator"
                },
                "pinout_key": "led"
            }
        ],
        "defects": []
    },
    "arduino-uno-defect": {
        "id": "arduino-uno-defect",
        "name": "Arduino Uno R3 (Defect Variant B)",
        "badge": "5 DEFECTS",
        "defect_count": 5,
        "description": "Defective manufacturing test sample containing 5 deliberate hardware flaws: missing filter capacitor, solder bridge, rotated diode, cold solder joint, and lifted trace.",
        "detections_count": 24,
        "model": "SIMULATED-v1.4",
        "defects": [
            {
                "id": "def_missing_c1",
                "type": "missing_component",
                "severity": "likely",
                "conf": 0.96,
                "component_ref": "C1",
                "bbox_norm": [0.15, 0.80, 0.06, 0.06],
                "note": "Filter capacitor C1 missing from footprint. High 5V voltage ripple expected under heavy USB/Vin load.",
                "remediation": "Solder 47µF 25V SMD electrolytic capacitor to footprint C1."
            },
            {
                "id": "def_solder_bridge_u2",
                "type": "solder_bridge",
                "severity": "likely",
                "conf": 0.98,
                "component_ref": "U2",
                "bbox_norm": [0.47, 0.65, 0.05, 0.04],
                "note": "Visible excess solder bridge shorting Pin 7 (VCC) and Pin 8 (GND) on U2 ATmega328P. Will cause power supply rail short!",
                "remediation": "Apply flux and clean bridge with desoldering wick before applying power."
            },
            {
                "id": "def_reverse_diode_d1",
                "type": "reversed_polarity",
                "severity": "likely",
                "conf": 0.92,
                "component_ref": "D1",
                "bbox_norm": [0.10, 0.58, 0.04, 0.03],
                "note": "Protection diode D1 cathode line marker rotated 180 degrees. External DC barrel power will not pass to regulator.",
                "remediation": "Rework diode D1 with cathode pointing toward AMS1117 regulator input."
            },
            {
                "id": "def_cold_joint_j2",
                "type": "cold_solder_joint",
                "severity": "possible",
                "conf": 0.88,
                "component_ref": "J2",
                "bbox_norm": [0.07, 0.75, 0.06, 0.06],
                "note": "Dull, non-wetting solder fillet on primary ground pin of Barrel Jack J2. Risk of intermittent DC connection.",
                "remediation": "Reflow pin with soldering iron at 350°C and fresh 63/37 solder."
            },
            {
                "id": "def_lifted_trace_sw1",
                "type": "lifted_trace",
                "severity": "possible",
                "conf": 0.84,
                "component_ref": "SW1",
                "bbox_norm": [0.09, 0.42, 0.04, 0.03],
                "note": "Micro-fracture / copper peeling on reset signal trace leading to pin 1 of SW1. Reset button may fail to trigger.",
                "remediation": "Bridge hairline crack with 30 AWG kynar jumper wire."
            }
        ]
    },
    "esp32-devkit": {
        "id": "esp32-devkit",
        "name": "ESP32 DevKit V1 (Dual-Core SoC)",
        "badge": "REFERENCE",
        "defect_count": 0,
        "description": "Espressif ESP32 dual-core Xtensa 240MHz development board with Wi-Fi/Bluetooth, CP2102 USB-UART, and dual 15-pin headers.",
        "detections_count": 18,
        "model": "SIMULATED-v1.4",
        "components": [
            {
                "id": "U1",
                "designator": "U1",
                "name": "U1 ESP-WROOM-32 Wi-Fi & BLE Module",
                "raw_name": "microcontroller",
                "category": "MCU",
                "conf": 0.995,
                "bbox_norm": [0.25, 0.15, 0.50, 0.45],
                "part_number": "ESP32-D0WDQ6",
                "source": "SIM",
                "specs": {"cores": 2, "freq": "240MHz", "ram": "520KB", "flash": "4MB", "wireless": "Wi-Fi 802.11 b/g/n + BT 4.2"},
                "pinout_key": "esp32-devkit"
            },
            {
                "id": "J1",
                "designator": "J1",
                "name": "J1 Micro-USB Type-B Receptacle",
                "raw_name": "usb_port",
                "category": "CONNECTOR",
                "conf": 0.989,
                "bbox_norm": [0.42, 0.85, 0.16, 0.12],
                "part_number": "MICRO-USB-SMD",
                "source": "SIM",
                "specs": {"type": "Micro-B 5-pin", "rating": "5V / 1A"},
                "pinout_key": "usb-type-b"
            },
            {
                "id": "U2",
                "designator": "U2",
                "name": "U2 CP2102 USB-to-UART Bridge Controller",
                "raw_name": "ic",
                "category": "IC",
                "conf": 0.982,
                "bbox_norm": [0.42, 0.68, 0.16, 0.12],
                "part_number": "CP2102-GMR",
                "source": "SIM",
                "specs": {"package": "QFN-28", "max_baud": "921600 bps"},
                "pinout_key": "cp2102"
            },
            {
                "id": "U3",
                "designator": "U3",
                "name": "U3 AMS1117-3.3V Low-Dropout Voltage Regulator",
                "raw_name": "voltage_regulator",
                "category": "REGULATOR",
                "conf": 0.975,
                "bbox_norm": [0.15, 0.70, 0.12, 0.10],
                "part_number": "AMS1117-3.3",
                "source": "SIM",
                "specs": {"output": "3.3V", "current": "800mA", "package": "SOT-223"},
                "pinout_key": "voltage-regulator-ams1117"
            },
            {
                "id": "SW1",
                "designator": "SW1",
                "name": "SW1 EN (Reset) Tactile Button",
                "raw_name": "push_button",
                "category": "SWITCH",
                "conf": 0.965,
                "bbox_norm": [0.22, 0.84, 0.08, 0.08],
                "part_number": "SMD-TACT-3X4",
                "source": "SIM",
                "specs": {"type": "Momentary NO", "pin": "EN / Chip Power Down"},
                "pinout_key": "pushbutton"
            },
            {
                "id": "SW2",
                "designator": "SW2",
                "name": "SW2 BOOT (GPIO0) Tactile Button",
                "raw_name": "push_button",
                "category": "SWITCH",
                "conf": 0.962,
                "bbox_norm": [0.70, 0.84, 0.08, 0.08],
                "part_number": "SMD-TACT-3X4",
                "source": "SIM",
                "specs": {"type": "Momentary NO", "pin": "GPIO0 Bootloader Trigger"},
                "pinout_key": "pushbutton"
            },
            {
                "id": "J2",
                "designator": "J2",
                "name": "J2 Left Pin Header 15-Pin (3V3, EN, ADC, Touch)",
                "raw_name": "gpio_header",
                "category": "CONNECTOR",
                "conf": 0.955,
                "bbox_norm": [0.05, 0.15, 0.08, 0.70],
                "part_number": "HDR-1X15-2.54MM",
                "source": "SIM",
                "specs": {"pins": 15, "pitch": "2.54mm"},
                "pinout_key": "esp32-devkit"
            },
            {
                "id": "J3",
                "designator": "J3",
                "name": "J3 Right Pin Header 15-Pin (GPIO, SPI, UART)",
                "raw_name": "gpio_header",
                "category": "CONNECTOR",
                "conf": 0.952,
                "bbox_norm": [0.87, 0.15, 0.08, 0.70],
                "part_number": "HDR-1X15-2.54MM",
                "source": "SIM",
                "specs": {"pins": 15, "pitch": "2.54mm"},
                "pinout_key": "esp32-devkit"
            }
        ],
        "defects": []
    },
    "breadboard-led": {
        "id": "breadboard-led",
        "name": "Solderless Breadboard Circuit (LED Blink)",
        "badge": "PROTOTYPE",
        "defect_count": 0,
        "description": "Standard Arduino Pin 13 LED blink circuit on half-size solderless breadboard with current-limiting series resistor.",
        "detections_count": 8,
        "model": "SIMULATED-v1.4",
        "components": [
            {
                "id": "U1",
                "designator": "U1",
                "name": "U1 Arduino Uno R3 Board",
                "raw_name": "arduino_board",
                "category": "BOARD",
                "conf": 0.990,
                "bbox_norm": [0.10, 0.15, 0.40, 0.65],
                "part_number": "ARDUINO-UNO-R3",
                "source": "SIM",
                "specs": {"core": "ATmega328P", "voltage": "5V"},
                "pinout_key": "arduino-uno"
            },
            {
                "id": "BB1",
                "designator": "BB1",
                "name": "BB1 Half-Size Solderless Breadboard (400 Tie-Points)",
                "raw_name": "breadboard",
                "category": "BOARD",
                "conf": 0.985,
                "bbox_norm": [0.55, 0.15, 0.38, 0.65],
                "part_number": "BB-400-TIE",
                "source": "SIM",
                "specs": {"tie_points": 400, "rails": "Dual Power Rails +/-"},
                "pinout_key": "breadboard"
            },
            {
                "id": "LED1",
                "designator": "LED1",
                "name": "LED1 5mm Red Diffused Through-Hole LED",
                "raw_name": "led",
                "category": "LED",
                "conf": 0.978,
                "bbox_norm": [0.65, 0.35, 0.08, 0.08],
                "part_number": "LED-5MM-RED",
                "source": "SIM",
                "specs": {"wavelength": "630nm", "forward_voltage": "2.0V", "current": "20mA", "anode": "Long Lead (+)", "cathode": "Flat Rim (-)"},
                "pinout_key": "led"
            },
            {
                "id": "R1",
                "designator": "R1",
                "name": "R1 220 Ohm 1/4W Carbon Film Resistor",
                "raw_name": "resistor",
                "category": "PASSIVE",
                "conf": 0.972,
                "bbox_norm": [0.72, 0.45, 0.08, 0.04],
                "part_number": "CFR-25-220R",
                "source": "SIM",
                "specs": {"resistance": "220 Ω", "tolerance": "±5% (Gold Band)", "color_code": "Red-Red-Brown-Gold", "power": "0.25 W"},
                "pinout_key": "resistor"
            },
            {
                "id": "W1",
                "designator": "W1",
                "name": "W1 Jumper Wire: Uno Pin 13 to Resistor Lead 1",
                "raw_name": "wire",
                "category": "WIRE",
                "conf": 0.950,
                "bbox_norm": [0.45, 0.20, 0.30, 0.25],
                "part_number": "JUMP-MM-BLUE",
                "source": "SIM",
                "specs": {"color": "Blue", "from": "uno:13", "to": "r1:1", "type": "Male-to-Male 24 AWG"},
                "pinout_key": "wire"
            },
            {
                "id": "W2",
                "designator": "W2",
                "name": "W2 Jumper Wire: LED Cathode to Uno GND",
                "raw_name": "wire",
                "category": "WIRE",
                "conf": 0.945,
                "bbox_norm": [0.45, 0.50, 0.25, 0.20],
                "part_number": "JUMP-MM-BLACK",
                "source": "SIM",
                "specs": {"color": "Black", "from": "led1:C", "to": "uno:GND.1", "type": "Male-to-Male 24 AWG"},
                "pinout_key": "wire"
            }
        ],
        "defects": []
    }
}

