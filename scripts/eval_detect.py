import argparse
import csv
import io
import os
import sys
import time
from typing import Dict, List, Set
from PIL import Image

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.api_client import VisionGuardAPIClient

def run_evaluation(images_dir: str, annotations_csv: str, backend_url: str):
    """
    Evaluates VisionGuard component recognition against ground-truth CSV annotations.
    Computes per-image recall and identifies false positive detections.
    """
    if not os.path.isdir(images_dir):
        print(f"Error: Images directory '{images_dir}' not found.")
        sys.exit(1)

    if not os.path.isfile(annotations_csv):
        print(f"Error: Annotations CSV '{annotations_csv}' not found.")
        sys.exit(1)

    # Read ground truth annotations
    ground_truth: Dict[str, Set[str]] = {}
    with open(annotations_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("image_filename", "").strip()
            comps_raw = row.get("expected_components", "").strip()
            if fname and comps_raw:
                expected = {c.strip().lower() for c in comps_raw.replace(";", ",").split(",") if c.strip()}
                ground_truth[fname] = expected

    print("=" * 75)
    print("VISIONGUARD HARDWARE DETECTION RECALL EVALUATOR")
    print("=" * 75)
    print(f"Target Images Directory : {images_dir}")
    print(f"Annotations File        : {annotations_csv}")
    print(f"Backend URL             : {backend_url}")
    print(f"Annotated Images Count  : {len(ground_truth)}")
    print("-" * 75)

    client = VisionGuardAPIClient(base_url=backend_url)

    per_image_results = []
    total_expected = 0
    total_matched = 0
    total_fp = 0

    for fname, expected_set in ground_truth.items():
        img_path = os.path.join(images_dir, fname)
        if not os.path.exists(img_path):
            print(f"[-] Warning: Image file '{fname}' in CSV was not found in directory.")
            continue

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        t0 = time.perf_counter()
        res = client.analyze_image(img_bytes, filename=fname)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        detected_names: Set[str] = set()
        if res.ok and isinstance(res.data, dict):
            circuit = res.data.get("circuit", {})
            dets = circuit.get("detections", [])
            for d in dets:
                name = d.get("name", "").lower()
                cat = d.get("category", "").lower()
                detected_names.add(name)
                detected_names.add(cat)
            for p in circuit.get("parts", []):
                detected_names.add(p.get("type", "").lower())

        # Calculate True Positives and False Positives
        matched = expected_set.intersection(detected_names)
        missing = expected_set - detected_names
        false_positives = {d for d in detected_names if d not in expected_set}

        recall = (len(matched) / len(expected_set)) if expected_set else 1.0

        total_expected += len(expected_set)
        total_matched += len(matched)
        total_fp += len(false_positives)

        per_image_results.append({
            "filename": fname,
            "expected_count": len(expected_set),
            "matched_count": len(matched),
            "recall": recall,
            "missing": sorted(list(missing)),
            "false_positives": sorted(list(false_positives)),
            "elapsed_ms": elapsed_ms
        })

        rec_pct = recall * 100.0
        print(f"[{fname[:24].ljust(25)}] Recall: {rec_pct:5.1f}% ({len(matched)}/{len(expected_set)}) | Missing: {sorted(list(missing))} | FP: {len(false_positives)} ({elapsed_ms:.1f}ms)")

    # Overall Summary
    print("-" * 75)
    mean_recall = (total_matched / total_expected) if total_expected > 0 else 0.0
    print(f"OVERALL SUMMARY:")
    print(f"  * Mean Recall          : {mean_recall * 100.0:.2f}% ({total_matched}/{total_expected})")
    print(f"  * Total False Positives: {total_fp}")
    print(f"  * Images Evaluated     : {len(per_image_results)}")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VisionGuard detection recall on hardware dataset")
    parser.add_argument("--images-dir", default="data/eval_images", help="Path to directory containing images")
    parser.add_argument("--annotations-csv", default="data/eval_annotations.csv", help="Path to ground truth annotations CSV")
    parser.add_argument("--backend-url", default="http://localhost:8000", help="VisionGuard backend base URL")
    args = parser.parse_args()

    run_evaluation(args.images_dir, args.annotations_csv, args.backend_url)
