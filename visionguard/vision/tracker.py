from collections import deque
from typing import Dict, List, Optional, Tuple
from visionguard.vision.tiling import calculate_iou
from visionguard.config import settings

class Track:
    def __init__(self, track_id: int, label: str, bbox: List[float], conf: float, history_len: int = 8):
        self.track_id: int = track_id
        self.label: str = label
        self.bbox: List[float] = bbox
        self.conf: float = conf
        self.history: deque = deque(maxlen=history_len)
        self.history.append(1)
        self.promoted: bool = False
        self.misses: int = 0
        self.age: int = 1

    def record_hit(self, bbox: List[float], conf: float) -> None:
        self.bbox = bbox
        self.conf = conf
        self.history.append(1)
        self.misses = 0
        self.age += 1

    def record_miss(self) -> None:
        self.history.append(0)
        self.misses += 1
        self.age += 1

    @property
    def hit_count(self) -> int:
        return sum(self.history)

class IoUTracker:
    """
    Simple IoU tracker for live camera streams.
    Assigns stable track_id and promotes parts into the circuit only after
    appearing in N of the last M frames.
    """
    def __init__(
        self,
        promote_n: Optional[int] = None,
        promote_m: Optional[int] = None,
        iou_match_threshold: float = 0.30,
        max_misses: int = 10
    ):
        self.promote_n: int = promote_n or settings.LIVE_PROMOTE_N
        self.promote_m: int = promote_m or settings.LIVE_PROMOTE_M
        self.iou_match_threshold: float = iou_match_threshold
        self.max_misses: int = max_misses
        self.next_track_id: int = 1
        self.active_tracks: Dict[int, Track] = {}

    def update(self, detections: List[Dict]) -> Tuple[List[Track], List[Track]]:
        """
        Updates tracks with detections from current frame.
        Returns: (all_active_tracks, newly_promoted_tracks)
        """
        matched_tracks = set()
        matched_dets = set()
        newly_promoted: List[Track] = []

        # Find best IoU matches between active tracks and detections
        candidates = []
        for det_idx, det in enumerate(detections):
            det_box = det["bbox"]
            det_label = det["label"]
            for track_id, track in self.active_tracks.items():
                if track.label == det_label:
                    iou = calculate_iou(track.bbox, det_box)
                    if iou >= self.iou_match_threshold:
                        candidates.append((iou, track_id, det_idx))

        # Sort candidates by IoU descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        for iou, track_id, det_idx in candidates:
            if track_id not in matched_tracks and det_idx not in matched_dets:
                track = self.active_tracks[track_id]
                det = detections[det_idx]
                track.record_hit(det["bbox"], det["conf"])
                matched_tracks.add(track_id)
                matched_dets.add(det_idx)

        # Unmatched active tracks get a miss
        for track_id, track in list(self.active_tracks.items()):
            if track_id not in matched_tracks:
                track.record_miss()
                if track.misses > self.max_misses:
                    del self.active_tracks[track_id]

        # Unmatched detections become new tracks
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                tid = self.next_track_id
                self.next_track_id += 1
                new_track = Track(
                    track_id=tid,
                    label=det["label"],
                    bbox=det["bbox"],
                    conf=det["conf"],
                    history_len=self.promote_m
                )
                self.active_tracks[tid] = new_track

        # Check promotion condition
        for track in self.active_tracks.values():
            if not track.promoted and track.hit_count >= self.promote_n:
                track.promoted = True
                newly_promoted.append(track)

        return list(self.active_tracks.values()), newly_promoted

