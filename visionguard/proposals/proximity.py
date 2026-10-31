from typing import List, Tuple
from visionguard.models.circuit import Circuit, Connection

# Approximate header pin locations on Arduino Uno in normalized board coordinates (x, y)
# Digital header row along top edge (y ~ 0.05)
HEADER_PINS = [
    ("13", 0.38, 0.05),
    ("12", 0.42, 0.05),
    ("11", 0.46, 0.05),
    ("10", 0.50, 0.05),
    ("9", 0.54, 0.05),
    ("8", 0.58, 0.05),
    ("7", 0.65, 0.05),
    ("6", 0.69, 0.05),
    ("5", 0.73, 0.05),
    ("4", 0.77, 0.05),
    ("3", 0.81, 0.05),
    ("2", 0.85, 0.05),
    ("1", 0.89, 0.05),
    ("0", 0.93, 0.05),
    # Power header row along bottom edge (y ~ 0.95)
    ("5V", 0.48, 0.95),
    ("3.3V", 0.44, 0.95),
    ("GND.1", 0.53, 0.95),
    ("GND.2", 0.57, 0.95),
    ("VIN", 0.62, 0.95),
]

def propose_from_proximity(circuit: Circuit) -> List[Connection]:
    """
    Proposes wire connections when a detected part's bounding box sits next to a template pin/header row.
    Always confirmed=False with low-to-medium confidence (e.g. 0.40 - 0.55).
    """
    proposals: List[Connection] = []
    existing = {(c.from_, c.to) for c in circuit.connections}
    existing.update({(c.to, c.from_) for c in circuit.connections})

    for part in circuit.parts:
        if not part.bbox or len(part.bbox) < 4:
            continue

        # bbox is [x, y, w, h]
        bx, by, bw, bh = part.bbox
        cx = bx + bw / 2.0
        cy = by + bh / 2.0

        # Find closest header pin
        closest_pin = None
        min_dist = float("inf")

        for pin_name, hx, hy in HEADER_PINS:
            dist = ((cx - hx) ** 2 + (cy - hy) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_pin = pin_name

        # If part is within proximity threshold (e.g. 0.35 in normalized coords)
        if min_dist < 0.35 and closest_pin:
            part_pin = "1" if part.type == "resistor" else ("A" if part.type == "led" else "1.l")
            ep_part = f"{part.id}:{part_pin}"
            ep_uno = f"uno:{closest_pin}"

            if (ep_uno, ep_part) not in existing and (ep_part, ep_uno) not in existing:
                # Calculate confidence inversely proportional to distance (0.35 to 0.55)
                conf = max(0.35, min(0.55, 0.60 - min_dist * 0.7))
                wire_col = "black" if "GND" in closest_pin else ("red" if closest_pin in ("5V", "3.3V", "VIN") else "green")
                proposals.append(Connection(
                    **{
                        "from": ep_uno,
                        "to": ep_part,
                        "source": "vision",
                        "conf": round(conf, 2),
                        "confirmed": False,
                        "wire_color": wire_col,
                        "note": f"Proximity heuristic: {part.id} near {closest_pin} (dist: {min_dist:.2f})"
                    }
                ))
                existing.add((ep_uno, ep_part))

    return proposals

