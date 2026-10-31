import io
import math
from typing import Any, Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont

from visionguard.models.circuit import Circuit

CATEGORY_COLORS = {
    "board": (0, 102, 204),          # Blue
    "ic": (153, 51, 204),            # Purple
    "mcu": (153, 51, 204),           # Purple
    "passive": (255, 153, 0),        # Orange
    "semiconductor": (0, 204, 102),  # Green
    "led": (0, 230, 115),            # Bright Green
    "connector": (0, 204, 204),      # Cyan
    "switch": (255, 204, 0),         # Yellow
    "regulator": (255, 128, 0),      # Dark Orange
    "crystal": (178, 102, 255),      # Light Purple
    "diode": (255, 77, 77),          # Coral
    "module": (255, 102, 153),       # Pink
    "defect": (255, 51, 51),         # Red
    "default": (180, 180, 180)       # Light Gray
}

def draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    coords: List[float],
    outline: tuple,
    width: int = 2,
    dash_length: int = 6
):
    """Draws a dashed rectangle outline."""
    x1, y1, x2, y2 = coords
    # Top and bottom
    for x in range(int(x1), int(x2), dash_length * 2):
        draw.line([(x, y1), (min(x + dash_length, int(x2)), y1)], fill=outline, width=width)
        draw.line([(x, y2), (min(x + dash_length, int(x2)), y2)], fill=outline, width=width)
    # Left and right
    for y in range(int(y1), int(y2), dash_length * 2):
        draw.line([(x1, y), (x1, min(y + dash_length, int(y2)))], fill=outline, width=width)
        draw.line([(x2, y), (x2, min(y + dash_length, int(y2)))], fill=outline, width=width)

def generate_circuit_overlay(circuit: Circuit, original_image_bytes: Optional[bytes] = None) -> bytes:
    """
    Renders the visual diagnostic overlay onto the original uploaded image.
    Draws all component detection boxes (colored by category, dashed for weak/unconfirmed),
    labels with confidence and OCR part numbers, and flags potential defects in crimson.
    Returns PNG bytes.
    """
    if original_image_bytes:
        try:
            base_img = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
        except Exception:
            base_img = Image.new("RGB", (800, 600), color=(30, 35, 45))
    else:
        # Determine canvas size from detections if possible
        max_x = max([d.bbox_px[0] + d.bbox_px[2] for d in circuit.detections if d.bbox_px] or [800.0])
        max_y = max([d.bbox_px[1] + d.bbox_px[3] for d in circuit.detections if d.bbox_px] or [600.0])
        cw = max(800, int(max_x + 50))
        ch = max(600, int(max_y + 50))
        base_img = Image.new("RGB", (cw, ch), color=(25, 30, 40))

    img_w, img_h = base_img.size
    overlay = base_img.copy()
    draw = ImageDraw.Draw(overlay)

    # 1. Draw Detections
    for det in circuit.detections:
        bbox = det.bbox_px
        if not bbox or len(bbox) != 4:
            continue

        x, y, w, h = bbox
        x2, y2 = x + w, y + h

        # Clamp to canvas boundaries
        x1_c, y1_c = max(0, x), max(0, y)
        x2_c, y2_c = min(img_w - 1, x2), min(img_h - 1, y2)

        color = CATEGORY_COLORS.get(det.category.lower(), CATEGORY_COLORS["default"])
        is_weak = det.weak or (det.conf < 0.45)
        is_dashed = is_weak or (not det.confirmed)

        # Draw box
        if is_dashed:
            draw_dashed_rectangle(draw, [x1_c, y1_c, x2_c, y2_c], outline=color, width=2)
        else:
            draw.rectangle([x1_c, y1_c, x2_c, y2_c], outline=color, width=3)

        # Build label text
        weak_tag = " (weak)" if is_weak else ""
        part_tag = f" [{det.part_number}]" if det.part_number else ""
        label = f"{det.name} {det.conf:.2f}{weak_tag}{part_tag}"

        # Draw text background banner
        text_y0 = max(0.0, y1_c - 16.0)
        draw.rectangle([x1_c, text_y0, x1_c + (len(label) * 7.5), text_y0 + 15.0], fill=color)
        draw.text((x1_c + 3, text_y0 + 1), label, fill=(255, 255, 255))

    # 2. Draw Defects in crimson red
    for def_item in circuit.defects:
        d_bbox = def_item.bbox_px
        if not d_bbox or len(d_bbox) != 4:
            continue

        dx, dy, dw, dh = d_bbox
        dx2, dy2 = dx + dw, dy + dh
        dx1_c, dy1_c = max(0, dx), max(0, dy)
        dx2_c, dy2_c = min(img_w - 1, dx2), min(img_h - 1, dy2)

        d_color = CATEGORY_COLORS["defect"]
        draw.rectangle([dx1_c, dy1_c, dx2_c, dy2_c], outline=d_color, width=3)

        d_label = f"DEFECT: {def_item.type} ({def_item.severity})"
        d_text_y0 = max(0.0, dy1_c - 16.0)
        draw.rectangle([dx1_c, d_text_y0, dx1_c + (len(d_label) * 7.5), d_text_y0 + 15.0], fill=d_color)
        draw.text((dx1_c + 3, d_text_y0 + 1), d_label, fill=(255, 255, 255))

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return buf.getvalue()

