import uuid
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from visionguard.templates.boards import normalize_pin_name, PART_SPECS

class BoardConfig(BaseModel):
    template: str = "arduino-uno"

    model_config = {
        "extra": "ignore"
    }

class Part(BaseModel):
    id: str
    type: str
    value: Optional[str] = None
    color: Optional[str] = None
    bbox: Optional[List[float]] = None # [x, y, w, h]
    source: Literal["vision", "live", "user", "template", "stub"] = "vision"
    conf: float = Field(default=1.0, ge=0.0, le=1.0)
    confirmed: bool = False
    track_id: Optional[int] = None
    wokwi_type: Optional[str] = None

    model_config = {
        "extra": "ignore"
    }

    def confirm_user(self) -> None:
        """Sets source to user, confidence to 1.0, confirmed to True."""
        self.source = "user"
        self.conf = 1.0
        self.confirmed = True

class Connection(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    source: Literal["vision", "user", "template"] = "vision"
    conf: float = Field(default=1.0, ge=0.0, le=1.0)
    confirmed: bool = False
    wire_color: Optional[str] = None
    note: Optional[str] = None

    model_config = {
        "populate_by_name": True,
        "extra": "ignore"
    }

    @property
    def to_(self) -> str:
        """Convenience property for symmetry with from_."""
        return self.to

    def confirm_user(self) -> None:
        """Sets source to user, confidence to 1.0, confirmed to True."""
        self.source = "user"
        self.conf = 1.0
        self.confirmed = True

class InspectionSlot(BaseModel):
    slot: str
    state: Literal["present", "missing", "skipped"]
    conf: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = {
        "extra": "ignore"
    }

class Detection(BaseModel):
    """
    Top-level recognized hardware component detection.
    Coordinate Convention:
      - Origin (0, 0) is top-left of the ORIGINAL image.
      - bbox_px: [x, y, w, h] in absolute original image pixels.
      - bbox_norm: [x, y, w, h] normalized in [0.0, 1.0] relative to the ORIGINAL image.
    """
    id: str
    name: str
    category: str
    bbox_px: List[float]
    bbox_norm: List[float]
    conf: float = Field(default=1.0, ge=0.0, le=1.0)
    source: Literal["pcb_model", "open_vocab", "llm", "stub"] = "open_vocab"
    ocr_text: Optional[str] = None
    part_number: Optional[str] = None
    mapped_part_id: Optional[str] = None
    confirmed: bool = False
    weak: bool = False
    layers_matched: List[str] = Field(default_factory=list)

    model_config = {
        "extra": "ignore"
    }

class Defect(BaseModel):
    """
    Potential physical, optical, or assembly defect on the hardware.
    Severity is strictly 'possible' or 'likely'—never claiming certainty or hardware safety.
    """
    id: str
    type: str
    bbox_px: Optional[List[float]] = None # [x, y, w, h] or None for global defects
    conf: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "heuristic" # "nv_slot_inspector", "pcb_defect", "heuristic", "llm"
    severity: Literal["possible", "likely"] = "possible"
    note: Optional[str] = None

    model_config = {
        "extra": "ignore"
    }

class HealthSummary(BaseModel):
    """
    Deterministic hardware health summary. Never claims the hardware is 'safe'.
    """
    counts_by_category: dict = Field(default_factory=dict)
    weak_detections_count: int = 0
    defects_by_severity: dict = Field(default_factory=dict)
    skipped_checks: List[str] = Field(default_factory=list)
    overall_status: Literal["ok", "review", "issues"] = "review"
    reasons: List[str] = Field(default_factory=list)
    checks_not_run: List[str] = Field(default_factory=list)

    model_config = {
        "extra": "ignore"
    }

class Circuit(BaseModel):
    version: int = 1
    id: str
    image_id: Optional[str] = None
    board: BoardConfig = Field(default_factory=BoardConfig)
    parts: List[Part] = Field(default_factory=list)
    connections: List[Connection] = Field(default_factory=list)
    inspection: List[InspectionSlot] = Field(default_factory=list)
    detections: List[Detection] = Field(default_factory=list)
    defects: List[Defect] = Field(default_factory=list)
    health: Optional[HealthSummary] = None

    model_config = {
        "populate_by_name": True,
        "extra": "ignore"
    }

    @field_validator("parts")
    @classmethod
    def validate_unique_part_ids(cls, parts: List[Part]) -> List[Part]:
        ids = set()
        for part in parts:
            if part.id in ids:
                raise ValueError(f"Duplicate part id found: '{part.id}'")
            ids.add(part.id)
        return parts

    @model_validator(mode="after")
    def normalize_all_pins(self) -> "Circuit":
        """
        Normalizes pin names in all connections based on part types and board template.
        """
        part_type_map = {p.id: p.type for p in self.parts}
        # Board template itself acts as 'uno' or 'board'
        board_template = self.board.template

        normalized_connections: List[Connection] = []
        for conn in self.connections:
            # Parse 'from' and 'to'
            from_str = conn.from_
            to_str = conn.to

            norm_from = cls_normalize_endpoint(from_str, part_type_map, board_template)
            norm_to = cls_normalize_endpoint(to_str, part_type_map, board_template)

            # Reconstruct Connection with normalized endpoints
            norm_conn = Connection(
                **{
                    "from": norm_from,
                    "to": norm_to,
                    "source": conn.source,
                    "conf": conn.conf,
                    "confirmed": conn.confirmed,
                    "wire_color": conn.wire_color,
                    "note": conn.note,
                }
            )
            normalized_connections.append(norm_conn)

        self.connections = normalized_connections
        return self

def cls_normalize_endpoint(endpoint: str, part_type_map: dict, board_template: str) -> str:
    """Normalizes an endpoint formatted as 'partId:pin'."""
    if ":" not in endpoint:
        return endpoint
    part_id, pin = endpoint.split(":", 1)
    part_id_strip = part_id.strip()
    pin_strip = pin.strip()

    if part_id_strip.lower() in ("uno", "board", "arduino", "arduino-uno"):
        norm_pin = normalize_pin_name("arduino-uno", pin_strip)
        return f"{part_id_strip}:{norm_pin}"

    part_type = part_type_map.get(part_id_strip)
    if part_type:
        norm_pin = normalize_pin_name(part_type, pin_strip)
        return f"{part_id_strip}:{norm_pin}"

    return f"{part_id_strip}:{pin_strip}"

def create_empty_circuit(board_template: str = "arduino-uno", circuit_id: Optional[str] = None) -> Circuit:
    """
    Factory for an empty-but-valid Circuit document.
    Used whenever vision fails or to start a fresh circuit.
    """
    cid = circuit_id or str(uuid.uuid4())
    return Circuit(
        version=1,
        id=cid,
        board=BoardConfig(template=board_template),
        parts=[],
        connections=[],
        inspection=[],
        detections=[],
        defects=[],
        health=None
    )
