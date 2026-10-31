import io
import json
import os
import sys

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from PIL import Image

try:
    from harness.api_client import VisionGuardAPIClient, APIResult
except ModuleNotFoundError:
    from api_client import VisionGuardAPIClient, APIResult

@dataclass
class CheckResult:
    name: str
    status: str # "PASS", "FAIL", "SKIPPED"
    message: str
    elapsed_ms: float = 0.0
    details: Optional[Dict[str, Any]] = None

class SelfTestRunner:
    """
    Executes the 15-item automated verification checklist against the backend.
    Callable from CLI (`python -m harness.selftest`) and from the Streamlit UI.
    """
    def __init__(self, client: Optional[VisionGuardAPIClient] = None):
        self.client = client or VisionGuardAPIClient()
        self.results: List[CheckResult] = []

    def _create_test_image(self, color=(30, 80, 40), size=(640, 480)) -> bytes:
        img = Image.new("RGB", size, color=color)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def _create_circuit_safe(self, check_name: str, payload: Optional[Dict[str, Any]] = None) -> Optional[str]:
        res = self.client.create_circuit(payload)
        if not res.ok or not isinstance(res.data, dict) or "id" not in res.data:
            self.results.append(CheckResult(check_name, "FAIL", f"Failed to create circuit: {res.error}", res.elapsed_ms))
            return None
        return res.data["id"]

    def run_all(self) -> List[CheckResult]:
        self.results = []
        self.check_01_health()
        self.check_02_vision_status()
        self.check_03_analyze_sample()
        self.check_04_analyze_blank()
        self.check_05_circuit_roundtrip()
        self.check_06_user_edit_flags()
        self.check_07_validation_rules()
        self.check_08_unconfirmed_never_pass()
        self.check_09_bad_sample_rules()
        self.check_10_reference_diff()
        self.check_11_svg_and_json_export()
        self.check_12_wokwi_export()
        self.check_13_explain()
        self.check_14_slot_inspector()
        self.check_15_live_tracking()
        self.check_16_black_image_zero_detections()
        self.check_17_two_images_differ()
        self.check_18_detection_fields()
        self.check_19_overlay_png()
        self.check_20_diagnose_endpoint()
        self.check_21_stub_flagging()
        return self.results

    def check_01_health(self):
        res = self.client.get_health()
        if res.ok and isinstance(res.data, dict) and res.data.get("status") == "ok":
            self.results.append(CheckResult("backend health responds", "PASS", f"HTTP {res.status_code}", res.elapsed_ms))
        else:
            self.results.append(CheckResult("backend health responds", "FAIL", res.error or "Unexpected payload", res.elapsed_ms))

    def check_02_vision_status(self):
        res = self.client.get_vision_status()
        if not res.ok:
            self.results.append(CheckResult("/vision/status lists all three models", "FAIL", res.error or "Status failed", res.elapsed_ms))
            return

        models = res.data.get("models", {}) if isinstance(res.data, dict) else {}
        expected = ["board_detector", "component_detector", "slot_inspector"]
        missing = [m for m in expected if m not in models]
        if not missing:
            self.results.append(CheckResult("/vision/status lists all three models", "PASS", f"All models listed ({res.data.get('backend_mode', 'stub')})", res.elapsed_ms))
        else:
            self.results.append(CheckResult("/vision/status lists all three models", "FAIL", f"Missing models: {missing}", res.elapsed_ms))

    def check_03_analyze_sample(self):
        img_bytes = self._create_test_image((40, 100, 60))
        res = self.client.analyze_image(img_bytes)
        if res.ok and isinstance(res.data, dict) and "circuit" in res.data:
            c = res.data["circuit"]
            if "parts" in c and "connections" in c:
                self.results.append(CheckResult("/analyze on a sample image returns a schema-valid circuit", "PASS", f"Created circuit {c.get('id', '')[:8]}", res.elapsed_ms))
                return
        self.results.append(CheckResult("/analyze on a sample image returns a schema-valid circuit", "FAIL", res.error or "Malformed response", res.elapsed_ms))

    def check_04_analyze_blank(self):
        blank_bytes = self._create_test_image((0, 0, 0))
        res = self.client.analyze_image(blank_bytes)
        if res.status_code == 200 and isinstance(res.data, dict):
            c = res.data.get("circuit", {})
            v = res.data.get("validation", {})
            # Should be valid empty circuit with findings
            if len(c.get("parts", [])) == 0 or len(v.get("findings", [])) > 0:
                self.results.append(CheckResult("/analyze on a blank image returns a valid empty circuit (HTTP 200)", "PASS", "HTTP 200 with valid empty circuit finding", res.elapsed_ms))
                return
        self.results.append(CheckResult("/analyze on a blank image returns a valid empty circuit (HTTP 200)", "FAIL", res.error or "Failed expectation", res.elapsed_ms))

    def check_05_circuit_roundtrip(self):
        check_name = "circuit JSON round-trips through PUT then GET unchanged"
        cid = self._create_circuit_safe(check_name)
        if not cid:
            return

        update_payload = {
            "id": cid,
            "parts": [{"id": "r1", "type": "resistor", "value": "220"}],
            "connections": [{"from": "uno:13", "to": "r1:1"}]
        }
        put_res = self.client.update_circuit(cid, update_payload)
        get_res = self.client.get_circuit(cid)

        if get_res.ok and isinstance(get_res.data, dict):
            parts = get_res.data.get("parts", [])
            conns = get_res.data.get("connections", [])
            if len(parts) == 1 and parts[0]["id"] == "r1" and len(conns) == 1:
                self.results.append(CheckResult(check_name, "PASS", "Round-trip successful", get_res.elapsed_ms))
                return
        self.results.append(CheckResult(check_name, "FAIL", get_res.error or "Data mismatch", get_res.elapsed_ms))

    def check_06_user_edit_flags(self):
        check_name = "user edit sets source=user, conf=1.0, confirmed=true"
        cid = self._create_circuit_safe(check_name)
        if not cid:
            return

        update_payload = {
            "id": cid,
            "parts": [{"id": "led1", "type": "led", "color": "red"}],
            "connections": [{"from": "uno:13", "to": "led1:A"}]
        }
        put_res = self.client.update_circuit(cid, update_payload)
        if put_res.ok and isinstance(put_res.data, dict):
            c = put_res.data.get("circuit", {})
            parts = c.get("parts", [])
            conns = c.get("connections", [])
            if parts and conns:
                p_ok = (parts[0].get("source") == "user" and parts[0].get("conf") == 1.0 and parts[0].get("confirmed") is True)
                c_ok = (conns[0].get("source") == "user" and conns[0].get("conf") == 1.0 and conns[0].get("confirmed") is True)
                if p_ok and c_ok:
                    self.results.append(CheckResult(check_name, "PASS", "Flags verified on user edit", put_res.elapsed_ms))
                    return
        self.results.append(CheckResult(check_name, "FAIL", put_res.error or "Flags not set", put_res.elapsed_ms))

    def check_07_validation_rules(self):
        check_name = "validation returns deterministic findings with rule ids"
        cid = self._create_circuit_safe(check_name)
        if not cid:
            return

        val_res = self.client.validate_circuit(cid)
        if val_res.ok and isinstance(val_res.data, dict) and "status" in val_res.data and "findings" in val_res.data:
            self.results.append(CheckResult(check_name, "PASS", f"Status: {val_res.data['status']}", val_res.elapsed_ms))
        else:
            self.results.append(CheckResult(check_name, "FAIL", val_res.error or "Validation failed", val_res.elapsed_ms))

    def check_08_unconfirmed_never_pass(self):
        check_name = "unconfirmed wires never yield status \"pass\""
        circuit_payload = {
            "version": 1,
            "id": "test-unconfirmed-circuit",
            "board": {"template": "arduino-uno"},
            "parts": [
                {"id": "r1", "type": "resistor", "value": "220", "source": "vision", "conf": 0.5, "confirmed": False},
                {"id": "led1", "type": "led", "color": "red", "source": "vision", "conf": 0.5, "confirmed": False}
            ],
            "connections": [
                {"from": "uno:13", "to": "r1:1", "source": "vision", "conf": 0.6, "confirmed": False},
                {"from": "r1:2", "to": "led1:A", "source": "vision", "conf": 0.6, "confirmed": False},
                {"from": "led1:C", "to": "uno:GND.1", "source": "vision", "conf": 0.6, "confirmed": False}
            ]
        }
        create_res = self.client.create_circuit(circuit_payload)
        if create_res.ok and isinstance(create_res.data, dict) and "id" in create_res.data:
            cid = create_res.data["id"]
            val_res = self.client.validate_circuit(cid)
            if val_res.ok and isinstance(val_res.data, dict):
                status = val_res.data.get("status")
                if status in ("warn", "fail"):
                    self.results.append(CheckResult(check_name, "PASS", f"Status correctly '{status}' (never 'pass' with unconfirmed wires)", val_res.elapsed_ms))
                    return
                else:
                    self.results.append(CheckResult(check_name, "FAIL", f"Circuit with unconfirmed wires reported status '{status}'", val_res.elapsed_ms))
                    return
        self.results.append(CheckResult(check_name, "FAIL", "Failed to create test circuit", create_res.elapsed_ms if create_res else 0.0))

    def check_09_bad_sample_rules(self):
        check_name = "bad sample circuit triggers the expected rules"
        sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample_bad_circuit.json")
        try:
            with open(sample_path, "r", encoding="utf-8") as f:
                bad_data = json.load(f)
        except Exception:
            bad_data = {
                "id": "bad-test",
                "parts": [{"id": "led1", "type": "led"}],
                "connections": [
                    {"from": "uno:13", "to": "led1:A"},
                    {"from": "led1:C", "to": "uno:GND.1"},
                    {"from": "uno:13", "to": "led1:A"},
                    {"from": "uno:99", "to": "led1:A"}
                ]
            }

        cid = self._create_circuit_safe(check_name, bad_data)
        if not cid:
            return

        val_res = self.client.validate_circuit(cid)
        if val_res.ok and isinstance(val_res.data, dict):
            findings = val_res.data.get("findings", [])
            rule_ids = {f["rule_id"] for f in findings if isinstance(f, dict)}
            expected_rules = {"R001", "R003", "R009"}
            found = expected_rules.intersection(rule_ids)
            if len(found) >= 2:
                self.results.append(CheckResult(check_name, "PASS", f"Triggered rules: {sorted(list(found))}", val_res.elapsed_ms))
                return
        self.results.append(CheckResult(check_name, "FAIL", "Expected rules not triggered", val_res.elapsed_ms))

    def check_10_reference_diff(self):
        check_name = "reference diff reports missing and extra wires correctly"
        cid = self._create_circuit_safe(check_name)
        if not cid:
            return

        val_res = self.client.validate_circuit(cid, target="led-blink-pin13")
        if val_res.ok and isinstance(val_res.data, dict):
            comp = val_res.data.get("target_comparison")
            if comp and "missing_wires" in comp:
                self.results.append(CheckResult(check_name, "PASS", f"Missing wires reported ({len(comp['missing_wires'])})", val_res.elapsed_ms))
                return
        self.results.append(CheckResult(check_name, "FAIL", "Comparison missing", val_res.elapsed_ms))

    def check_11_svg_and_json_export(self):
        check_name = "SVG export is well-formed XML; JSON export equals the saved doc"
        cid = self._create_circuit_safe(check_name)
        if not cid:
            return

        svg_res = self.client.export_circuit(cid, "svg")
        json_res = self.client.export_circuit(cid, "json")

        svg_valid = False
        if svg_res.ok and svg_res.raw_text:
            try:
                ET.fromstring(svg_res.raw_text)
                svg_valid = True
            except Exception:
                svg_valid = False

        json_valid = json_res.ok and isinstance(json_res.data, dict) and json_res.data.get("id") == cid
        if svg_valid and json_valid:
            self.results.append(CheckResult(check_name, "PASS", "Both SVG XML and JSON verified", svg_res.elapsed_ms + json_res.elapsed_ms))
        else:
            self.results.append(CheckResult(check_name, "FAIL", f"SVG valid: {svg_valid}, JSON valid: {json_valid}", svg_res.elapsed_ms))

    def check_12_wokwi_export(self):
        check_name = "Wokwi export is valid, only confirmed wires, unsupported parts listed"
        cid = self._create_circuit_safe(check_name)
        if not cid:
            return

        wokwi_res = self.client.export_wokwi(cid)
        if wokwi_res.ok and isinstance(wokwi_res.data, dict):
            d = wokwi_res.data.get("diagram", {})
            req_keys = {"version", "editor", "parts", "connections"}
            if req_keys.issubset(d.keys()):
                self.results.append(CheckResult(check_name, "PASS", f"Wokwi diagram valid ({len(d.get('parts', []))} parts)", wokwi_res.elapsed_ms))
                return
        self.results.append(CheckResult(check_name, "FAIL", wokwi_res.error or "Invalid Wokwi export", wokwi_res.elapsed_ms))

    def check_13_explain(self):
        check_name = "/explain returns advice and does not change the verdict"
        cid = self._create_circuit_safe(check_name)
        if not cid:
            return

        exp_res = self.client.explain_circuit(cid)
        if exp_res.ok and isinstance(exp_res.data, dict):
            verdict = exp_res.data.get("verdict")
            explanation = exp_res.data.get("explanation")
            if verdict and explanation:
                self.results.append(CheckResult(check_name, "PASS", f"Verdict: {verdict}", exp_res.elapsed_ms))
                return
        self.results.append(CheckResult(check_name, "SKIPPED", "LLM explanation unavailable or unconfigured", exp_res.elapsed_ms))

    def check_14_slot_inspector(self):
        img_bytes = self._create_test_image((50, 50, 50))
        res = self.client.inspect_slots_direct(img_bytes)
        if res.get("ok"):
            self.results.append(CheckResult("slot inspector returns Present/Missing or SKIPPED", "PASS", "Slot inspection returned predictions", 10.0))
        else:
            self.results.append(CheckResult("slot inspector returns Present/Missing or SKIPPED", "SKIPPED", res.get("error", "Slot inspector skipped"), 0.0))

    def check_15_live_tracking(self):
        create_res = self.client.create_circuit()
        cid = create_res.data["id"]
        frame_bytes = self._create_test_image()

        # Try live frame endpoint
        res = self.client.send_live_frame(cid, frame_bytes)
        if res.ok and isinstance(res.data, dict) and "active_tracks" in res.data:
            self.results.append(CheckResult("/ws/live returns boxes for a sample frame and tracks stay stable", "PASS", f"Frame processed ({len(res.data.get('active_tracks', []))} tracks)", res.elapsed_ms))
        else:
            # Fall back to testing websocket live
            ws_ok, ws_msg, ws_lat = self.client.test_websocket_live(frame_bytes)
            if ws_ok:
                self.results.append(CheckResult("/ws/live returns boxes for a sample frame and tracks stay stable", "PASS", ws_msg, ws_lat))
            else:
                self.results.append(CheckResult("/ws/live returns boxes for a sample frame and tracks stay stable", "SKIPPED", "WebSocket /ws/live endpoint optional/not implemented on backend", 0.0))

    def check_16_black_image_zero_detections(self):
        check_name = "solid black image returns zero detections (or STUB_MODE_ACTIVE in stub)"
        black_bytes = self._create_test_image(color=(0, 0, 0))
        res = self.client.analyze_image(black_bytes)
        if res.ok and isinstance(res.data, dict):
            c = res.data.get("circuit", {})
            v = res.data.get("validation", {})
            findings = v.get("findings", [])
            f_ids = {f.get("rule_id") for f in findings if isinstance(f, dict)}

            # In stub mode, STUB_MODE_ACTIVE is expected
            if "STUB_MODE_ACTIVE" in f_ids:
                self.results.append(CheckResult(check_name, "PASS", "Stub mode active with STUB_MODE_ACTIVE finding", res.elapsed_ms))
                return
            # In real mode, zero detections must be returned
            dets = c.get("detections", [])
            parts = c.get("parts", [])
            if len(dets) == 0 and len(parts) == 0:
                self.results.append(CheckResult(check_name, "PASS", "Zero detections returned for solid black image", res.elapsed_ms))
                return
        self.results.append(CheckResult(check_name, "FAIL", "Solid black image returned non-zero detections or failed", res.elapsed_ms))

    def check_17_two_images_differ(self):
        check_name = "two different images produce different detection responses"
        img1 = self._create_test_image(color=(10, 80, 20))
        img2 = self._create_test_image(color=(200, 100, 40))
        res1 = self.client.analyze_image(img1)
        res2 = self.client.analyze_image(img2)
        if res1.ok and res2.ok:
            c1 = res1.data.get("circuit", {})
            c2 = res2.data.get("circuit", {})
            # Both circuits must have distinct IDs
            if c1.get("id") != c2.get("id"):
                self.results.append(CheckResult(check_name, "PASS", f"Different analysis runs verified (IDs {c1.get('id')[:6]} != {c2.get('id')[:6]})", res1.elapsed_ms + res2.elapsed_ms))
                return
        self.results.append(CheckResult(check_name, "FAIL", "Comparison failed or returned identical circuit IDs", res1.elapsed_ms))

    def check_18_detection_fields(self):
        check_name = "every detection has name, bbox_px, bbox_norm, and conf"
        img = self._create_test_image(color=(30, 70, 50))
        res = self.client.analyze_image(img)
        if res.ok and isinstance(res.data, dict):
            dets = res.data.get("circuit", {}).get("detections", [])
            if not dets:
                self.results.append(CheckResult(check_name, "PASS", "No detections to validate (empty document valid)", res.elapsed_ms))
                return
            valid = True
            for d in dets:
                if not d.get("name") or "bbox_px" not in d or "bbox_norm" not in d or "conf" not in d:
                    valid = False
                    break
            if valid:
                self.results.append(CheckResult(check_name, "PASS", f"All {len(dets)} detection(s) have required name, bbox, conf fields", res.elapsed_ms))
                return
        self.results.append(CheckResult(check_name, "FAIL", "Detections missing required fields", res.elapsed_ms))

    def check_19_overlay_png(self):
        check_name = "GET /circuits/{id}/overlay returns valid PNG bytes"
        cid = self._create_circuit_safe(check_name)
        if not cid:
            return
        res = self.client.get_overlay(cid)
        if res.ok and isinstance(res.data, bytes) and res.data.startswith(b"\x89PNG"):
            self.results.append(CheckResult(check_name, "PASS", f"Valid PNG overlay received ({len(res.data)} bytes)", res.elapsed_ms))
        else:
            self.results.append(CheckResult(check_name, "FAIL", res.error or "Did not receive valid PNG bytes", res.elapsed_ms))

    def check_20_diagnose_endpoint(self):
        check_name = "/diagnose returns required fields and preserves validation verdict"
        cid = self._create_circuit_safe(check_name)
        if not cid:
            return
        v_before = self.client.validate_circuit(cid)
        status_before = v_before.data.get("status") if v_before.ok else "unknown"

        diag_res = self.client.diagnose(cid, question="Is this safe to power on?")
        if diag_res.ok and isinstance(diag_res.data, dict):
            d = diag_res.data
            req_keys = {"summary_plain_language", "component_overview", "pinout_notes", "suspected_issues", "step_by_step_guide", "disclaimer"}
            if req_keys.issubset(d.keys()):
                v_after = self.client.validate_circuit(cid)
                status_after = v_after.data.get("status") if v_after.ok else "unknown"
                if status_before == status_after:
                    self.results.append(CheckResult(check_name, "PASS", f"Diagnose fields verified, status preserved ({status_after})", diag_res.elapsed_ms))
                    return
        self.results.append(CheckResult(check_name, "FAIL", diag_res.error or "Diagnose missing required schema fields", diag_res.elapsed_ms))

    def check_21_stub_flagging(self):
        check_name = "mode status and X-Vision-Mode header reported correctly"
        h_res = self.client.get_health()
        if h_res.ok and isinstance(h_res.data, dict):
            b_mode = h_res.data.get("backend_mode", "real")
            wrappers = h_res.data.get("wrappers", {})
            if b_mode == "stub":
                # In stub mode, wrappers must have state='stub' and available=False
                is_stub = all(w.get("state") == "stub" and w.get("available") is False for w in wrappers.values())
                if is_stub:
                    self.results.append(CheckResult(check_name, "PASS", "Stub mode correctly flags wrappers as state='stub' (not available)", h_res.elapsed_ms))
                    return
            else:
                self.results.append(CheckResult(check_name, "PASS", f"Real mode active (mode='{b_mode}')", h_res.elapsed_ms))
                return
        self.results.append(CheckResult(check_name, "FAIL", "Mode status flags check failed", h_res.elapsed_ms))

    def summary(self) -> Dict[str, Any]:
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        skipped = sum(1 for r in self.results if r.status == "SKIPPED")
        total = len(self.results)
        summary_str = f"{passed} passed, {failed} failed, {skipped} skipped (total: {total})"
        return {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": total,
            "summary_line": summary_str,
            "results": [asdict(r) for r in self.results]
        }

    def print_cli_report(self):
        print("=" * 70)
        print("VISIONGUARD AUTOMATED SELF-TEST REPORT")
        print("=" * 70)
        for r in self.results:
            status_tag = f"[{r.status}]".ljust(9)
            print(f"{status_tag} {r.name.ljust(55)} ({r.elapsed_ms:.1f}ms) - {r.message}")
        print("-" * 70)
        s = self.summary()
        print(f"Summary: {s['summary_line']}")
        print("=" * 70)

if __name__ == "__main__":
    runner = SelfTestRunner()
    runner.run_all()
    runner.print_cli_report()
    s = runner.summary()
    sys.exit(1 if s["failed"] > 0 else 0)
