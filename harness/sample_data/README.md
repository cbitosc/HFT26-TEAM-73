# VisionGuard Harness Sample Data

This directory contains test datasets, reference circuit documents, and assets for verifying the backend via Streamlit.

## Included Sample Documents
- `sample_led_blink.json`: A verified canonical LED blink circuit (Arduino Uno pin 13 -> 220 ohm resistor -> Red LED -> GND). All connections confirmed.
- `sample_bad_circuit.json`: A test circuit deliberately engineered to trigger deterministic validation failures:
  - `R001 UNSUPPORTED_PIN`: Invalid pin `uno:99`.
  - `R003 DUPLICATE_CONNECTION`: Identical connection `uno:13 <-> led1:A` specified multiple times.
  - `R009 LED_NEEDS_SERIES_RESISTOR`: LED wired directly between pin 13 and GND without a current-limiting resistor.
  - `R010 UNCONFIRMED_WIRES`: Wire proposal with `confirmed: false`.

## Adding Test Assets
1. **PCB Photos**: Place sample `.jpg` or `.png` images of hardware/breadboards in this directory. They can be selected from the "Analyze an image" tab.
2. **Video Clips**: Place short `.mp4` or `.avi` camera recordings (5-10 seconds) for testing the live tracking frame stream in Tab 8.
3. **Slot Inspector Component Crops**: Place $224 \times 224$ component crops here for testing the NVIDIA TAO classifier in Tab 7. Include `present` or `missing` in the filename (e.g. `capacitor_present_01.jpg`, `mcu_missing_02.jpg`) so accuracy can be automatically calculated.

