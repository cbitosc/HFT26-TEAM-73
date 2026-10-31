from typing import Dict, Tuple
from visionguard.models.circuit import Circuit
from visionguard.templates.boards import PART_SPECS

def generate_circuit_svg(circuit: Circuit) -> str:
    """
    Renders a clean, readable schematic-style SVG layout of the circuit.
    - Confirmed wires are solid
    - Unconfirmed wires are dashed
    - Labels every part and used pin
    - Pure Python string generation (no browser or headless dependencies)
    """
    width = 1100
    height = 700

    # Board layout coordinates
    board_x, board_y = 60, 100
    board_w, board_h = 240, 480

    # Calculate pin coordinates on board block
    board_pins: Dict[str, Tuple[int, int]] = {}

    # Digital pins (top to middle along right edge of board)
    digital_pins = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "AREF", "GND.1"]
    for i, p in enumerate(digital_pins):
        py = board_y + 40 + i * 22
        board_pins[p] = (board_x + board_w, py)

    # Power & Analog pins (lower along right edge of board)
    pwr_analog_pins = ["5V", "3.3V", "VIN", "RESET", "IOREF", "GND.2", "GND.3", "A0", "A1", "A2", "A3", "A4", "A5"]
    for j, p in enumerate(pwr_analog_pins):
        py = board_y + 40 + (len(digital_pins) + j) * 15
        board_pins[p] = (board_x + board_w, py)

    # Component layout: spaced columns to the right
    part_pin_coords: Dict[str, Dict[str, Tuple[int, int]]] = {}
    part_boxes: Dict[str, Tuple[int, int, int, int]] = {}

    col_start_x = 450
    col_spacing = 180
    row_start_y = 120
    row_spacing = 160

    # Separate non-board parts
    discrete_parts = [p for p in circuit.parts if p.type not in ("arduino-uno", "board")]

    for idx, part in enumerate(discrete_parts):
        col = idx % 3
        row = idx // 3
        px = col_start_x + col * col_spacing
        py = row_start_y + row * row_spacing
        pw, ph = 120, 90
        part_boxes[part.id] = (px, py, pw, ph)

        # Assign terminal pin coordinates
        pin_coords = {}
        if part.type == "resistor":
            # Resistor has terminals 1 (left) and 2 (right)
            pin_coords["1"] = (px, py + ph // 2)
            pin_coords["2"] = (px + pw, py + ph // 2)
        elif part.type == "led":
            # LED has Anode A (left) and Cathode C (right)
            pin_coords["A"] = (px, py + ph // 2)
            pin_coords["C"] = (px + pw, py + ph // 2)
        elif part.type == "pushbutton":
            # Pushbutton 1.l, 1.r, 2.l, 2.r
            pin_coords["1.l"] = (px, py + 25)
            pin_coords["1.r"] = (px + pw, py + 25)
            pin_coords["2.l"] = (px, py + ph - 25)
            pin_coords["2.r"] = (px + pw, py + ph - 25)
        else:
            # Generic 2-pin
            pin_coords["1"] = (px, py + ph // 2)
            pin_coords["2"] = (px + pw, py + ph // 2)

        part_pin_coords[part.id] = pin_coords

    # Resolve used endpoints
    used_pins = set()
    for conn in circuit.connections:
        used_pins.add(conn.from_)
        used_pins.add(conn.to)

    # Start SVG
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background-color: #1e1e24; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">',
        '<!-- Defs for markers and styles -->',
        '<defs>',
        '  <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%"><feDropShadow dx="2" dy="2" stdDeviation="3" flood-opacity="0.4"/></filter>',
        '</defs>',
        '<!-- Title & Header -->',
        f'<text x="50" y="45" font-size="22" font-weight="bold" fill="#f8f9fa">VisionGuard Schematic: {circuit.id[:18]}</text>',
        f'<text x="50" y="70" font-size="13" fill="#adb5bd">Board: {circuit.board.template} | Parts: {len(circuit.parts)} | Connections: {len(circuit.connections)}</text>',
        '<!-- Legend -->',
        '<g transform="translate(780, 25)">',
        '  <rect x="0" y="0" width="280" height="50" rx="6" fill="#2b2d42" stroke="#4a4e69" />',
        '  <line x1="20" y1="20" x2="60" y2="20" stroke="#4ade80" stroke-width="3" />',
        '  <text x="70" y="24" font-size="12" fill="#f8f9fa">Confirmed Wire</text>',
        '  <line x1="20" y1="36" x2="60" y2="36" stroke="#facc15" stroke-width="2.5" stroke-dasharray="5,4" />',
        '  <text x="70" y="40" font-size="12" fill="#f8f9fa">Proposed Wire</text>',
        '</g>',
        '<!-- Arduino Board -->',
        f'<rect x="{board_x}" y="{board_y}" width="{board_w}" height="{board_h}" rx="12" fill="#006699" stroke="#004d73" stroke-width="3" filter="url(#shadow)"/>',
        f'<text x="{board_x + 20}" y="{board_y + 35}" font-size="16" font-weight="bold" fill="#ffffff">ARDUINO UNO</text>',
    ]

    # Draw Board Pins
    for pin_name, (px, py) in board_pins.items():
        is_used = any(f"uno:{pin_name}" in ep or f"board:{pin_name}" in ep for ep in used_pins)
        pin_color = "#38bdf8" if is_used else "#94a3b8"
        font_weight = "bold" if is_used else "normal"
        # Pin terminal circle
        svg_lines.append(f'<circle cx="{px}" cy="{py}" r="4" fill="{pin_color}" stroke="#004d73" stroke-width="1.5" />')
        # Pin label
        svg_lines.append(f'<text x="{px - 8}" y="{py + 4}" font-size="11" font-weight="{font_weight}" fill="#ffffff" text-anchor="end">{pin_name}</text>')

    # Draw Discrete Components
    for part in discrete_parts:
        if part.id not in part_boxes:
            continue
        bx, by, bw, bh = part_boxes[part.id]
        box_bg = "#27272a"
        border_col = "#3b82f6" if part.confirmed else "#eab308"
        status_label = "CONFIRMED" if part.confirmed else f"PROPOSAL ({part.conf:.2f})"

        svg_lines.append(f'<g id="part-{part.id}">')
        svg_lines.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="8" fill="{box_bg}" stroke="{border_col}" stroke-width="2" filter="url(#shadow)"/>')
        svg_lines.append(f'<text x="{bx + 10}" y="{by + 20}" font-size="13" font-weight="bold" fill="#f8f9fa">{part.id}</text>')
        svg_lines.append(f'<text x="{bx + 10}" y="{by + 36}" font-size="11" fill="#9ca3af">{part.type.upper()}</text>')

        # Value or color subtitle
        val_str = f"val: {part.value}" if part.value else (f"color: {part.color}" if part.color else "")
        if val_str:
            svg_lines.append(f'<text x="{bx + 10}" y="{by + 52}" font-size="10" fill="#a78bfa">{val_str}</text>')

        svg_lines.append(f'<text x="{bx + 10}" y="{by + 76}" font-size="9" fill="{border_col}">{status_label}</text>')

        # Terminals and labels
        p_coords = part_pin_coords.get(part.id, {})
        for pin_name, (tx, ty) in p_coords.items():
            svg_lines.append(f'<circle cx="{tx}" cy="{ty}" r="4" fill="{border_col}" stroke="#1e1e24" stroke-width="1.5"/>')
            anchor = "start" if tx <= bx + 10 else "end"
            label_offset = 8 if anchor == "start" else -8
            svg_lines.append(f'<text x="{tx + label_offset}" y="{ty + 4}" font-size="10" fill="#e2e8f0" text-anchor="{anchor}">{pin_name}</text>')
        svg_lines.append('</g>')

    # Helper function to get coordinates for any endpoint string "partId:pin"
    def resolve_endpoint_coords(endpoint: str) -> Tuple[int, int]:
        if ":" not in endpoint:
            return (board_x + board_w, board_y + 100)
        pid, pin = endpoint.split(":", 1)
        pid_clean = pid.strip()
        pin_clean = pin.strip()

        if pid_clean.lower() in ("uno", "board", "arduino", "arduino-uno"):
            return board_pins.get(pin_clean, (board_x + board_w, board_y + 100))

        if pid_clean in part_pin_coords:
            p_map = part_pin_coords[pid_clean]
            if pin_clean in p_map:
                return p_map[pin_clean]
            # Default to center of component box
            if pid_clean in part_boxes:
                bx, by, bw, bh = part_boxes[pid_clean]
                return (bx, by + bh // 2)

        return (board_x + board_w + 50, board_y + 100)

    # Draw Connections
    svg_lines.append('<!-- Circuit Connections / Wires -->')
    for idx, conn in enumerate(circuit.connections):
        x1, y1 = resolve_endpoint_coords(conn.from_)
        x2, y2 = resolve_endpoint_coords(conn.to)

        is_confirmed = conn.confirmed
        # Wire color determination
        if conn.wire_color:
            stroke_color = conn.wire_color
            if stroke_color == "black":
                stroke_color = "#334155" # Dark slate visible against dark background
            elif stroke_color == "red":
                stroke_color = "#ef4444"
            elif stroke_color == "green":
                stroke_color = "#22c55e"
        elif "GND" in conn.from_ or "GND" in conn.to:
            stroke_color = "#64748b" # Ground gray/slate
        elif any(pw in conn.from_ or pw in conn.to for pw in ("5V", "3.3V", "VIN")):
            stroke_color = "#ef4444" # Power red
        else:
            stroke_color = "#22c55e" # Signal green

        if not is_confirmed:
            stroke_dash = 'stroke-dasharray="6,4"'
            stroke_width = "2.0"
            stroke_color = "#facc15" # Yellow for proposed
        else:
            stroke_dash = 'stroke-dasharray="none"'
            stroke_width = "2.8"

        # Smooth cubic Bezier routing curve
        dx = abs(x2 - x1)
        ctrl_offset = max(40, int(dx * 0.45))
        cx1 = x1 + ctrl_offset if x1 < x2 else x1 - ctrl_offset
        cy1 = y1
        cx2 = x2 - ctrl_offset if x1 < x2 else x2 + ctrl_offset
        cy2 = y2

        path_d = f"M {x1} {y1} C {cx1} {cy1}, {cx2} {cy2}, {x2} {y2}"
        svg_lines.append(f'<path d="{path_d}" fill="none" stroke="{stroke_color}" stroke-width="{stroke_width}" {stroke_dash} opacity="0.9"/>')

    svg_lines.append('</svg>')
    return "\n".join(svg_lines)
