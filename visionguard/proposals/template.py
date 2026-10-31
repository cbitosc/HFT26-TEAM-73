from typing import List
from visionguard.models.circuit import Circuit, Connection

def propose_from_template(circuit: Circuit) -> List[Connection]:
    """
    Proposes connections based on trusted board template rules (e.g. standard ground rails).
    source="template", confirmed=False.
    """
    proposals: List[Connection] = []
    board_template = circuit.board.template.lower()

    if "arduino" in board_template or "uno" in board_template:
        # Check if we have an LED or pushbutton without a ground connection
        existing_connections = {(c.from_, c.to) for c in circuit.connections}
        existing_connections.update({(c.to, c.from_) for c in circuit.connections})

        for part in circuit.parts:
            if part.type == "led":
                # Template rule: LED cathode connects to ground rail
                pair1 = (f"{part.id}:C", "uno:GND.1")
                pair2 = ("uno:GND.1", f"{part.id}:C")
                if pair1 not in existing_connections and pair2 not in existing_connections:
                    proposals.append(Connection(
                        **{
                            "from": f"{part.id}:C",
                            "to": "uno:GND.1",
                            "source": "template",
                            "conf": 0.80,
                            "confirmed": False,
                            "wire_color": "black",
                            "note": "Template rule: LED cathode to ground rail"
                        }
                    ))
            elif part.type == "pushbutton":
                # Template rule: Pushbutton ground reference
                pair1 = (f"{part.id}:2.r", "uno:GND.1")
                pair2 = ("uno:GND.1", f"{part.id}:2.r")
                if pair1 not in existing_connections and pair2 not in existing_connections:
                    proposals.append(Connection(
                        **{
                            "from": f"{part.id}:2.r",
                            "to": "uno:GND.1",
                            "source": "template",
                            "conf": 0.75,
                            "confirmed": False,
                            "wire_color": "black",
                            "note": "Template rule: Pushbutton reference to ground"
                        }
                    ))
    return proposals

