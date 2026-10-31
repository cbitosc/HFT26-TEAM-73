from visionguard.vision.tracker import IoUTracker

def test_iou_tracker_promotion():
    # Configure tracker to promote when detected in 3 of 5 frames
    tracker = IoUTracker(promote_n=3, promote_m=5, iou_match_threshold=0.3)

    det_led = [{"label": "led", "bbox": [100.0, 100.0, 50.0, 50.0], "conf": 0.85}]

    # Frame 1: first detection
    active, promoted = tracker.update(det_led)
    assert len(active) == 1
    assert len(promoted) == 0
    t_id = active[0].track_id

    # Frame 2: second detection
    active, promoted = tracker.update(det_led)
    assert len(active) == 1
    assert len(promoted) == 0
    assert active[0].track_id == t_id

    # Frame 3: third detection -> should trigger promotion!
    active, promoted = tracker.update(det_led)
    assert len(active) == 1
    assert len(promoted) == 1
    assert promoted[0].track_id == t_id
    assert promoted[0].label == "led"

    # Frame 4: fourth detection -> already promoted, so newly promoted is empty
    active, promoted = tracker.update(det_led)
    assert len(promoted) == 0

def test_iou_tracker_drop_inactive():
    tracker = IoUTracker(promote_n=3, promote_m=5, max_misses=2)
    det = [{"label": "resistor", "bbox": [50.0, 50.0, 30.0, 30.0], "conf": 0.8}]

    # Frame 1
    active, _ = tracker.update(det)
    assert len(active) == 1

    # Frame 2: missed
    active, _ = tracker.update([])
    assert len(active) == 1

    # Frame 3: missed
    active, _ = tracker.update([])
    assert len(active) == 1

    # Frame 4: missed > max_misses (2) -> dropped
    active, _ = tracker.update([])
    assert len(active) == 0

