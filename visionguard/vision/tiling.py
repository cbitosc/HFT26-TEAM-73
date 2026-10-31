from typing import Callable, Dict, List, Tuple

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Computes Intersection-over-Union (IoU) for two bounding boxes [x, y, w, h].
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

def nms(detections: List[Dict], iou_threshold: float = 0.45) -> List[Dict]:
    """
    Applies Non-Maximum Suppression (NMS) to merged tile detections.
    Each detection dict contains: {'label': str, 'bbox': [x, y, w, h], 'conf': float}
    """
    if not detections:
        return []

    # Sort descending by confidence
    sorted_dets = sorted(detections, key=lambda d: d.get("conf", 0.0), reverse=True)
    kept: List[Dict] = []

    for det in sorted_dets:
        box = det["bbox"]
        label = det.get("label")
        should_keep = True

        for k in kept:
            # Suppress if same label and IoU exceeds threshold
            if k.get("label") == label:
                if calculate_iou(box, k["bbox"]) > iou_threshold:
                    should_keep = False
                    break

        if should_keep:
            kept.append(det)

    return kept

def tile_and_detect(
    image_w: int,
    image_h: int,
    detect_fn: Callable[[int, int, int, int], List[Dict]],
    tile_size: int = 1024,
    overlap: float = 0.20,
    nms_iou_threshold: float = 0.45
) -> List[Dict]:
    """
    Tiles an image if larger than tile_size, runs detect_fn on each tile,
    shifts bounding boxes back to full image coordinates, and runs NMS.
    detect_fn signature: (tile_x, tile_y, tile_w, tile_h) -> list of detections in tile coords
    """
    if image_w <= tile_size and image_h <= tile_size:
        return detect_fn(0, 0, image_w, image_h)

    stride = int(tile_size * (1.0 - overlap))
    if stride <= 0:
        stride = tile_size

    all_detections: List[Dict] = []

    y = 0
    while y < image_h:
        cur_h = min(tile_size, image_h - y)
        x = 0
        while x < image_w:
            cur_w = min(tile_size, image_w - x)
            tile_dets = detect_fn(x, y, cur_w, cur_h)

            for d in tile_dets:
                tb = d["bbox"]
                # Shift box to full board coordinates
                global_box = [x + tb[0], y + tb[1], tb[2], tb[3]]
                shifted_det = dict(d)
                shifted_det["bbox"] = global_box
                all_detections.append(shifted_det)

            if x + cur_w >= image_w:
                break
            x += stride

        if y + cur_h >= image_h:
            break
        y += stride

    return nms(all_detections, iou_threshold=nms_iou_threshold)

