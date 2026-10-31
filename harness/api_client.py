import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import requests

DEFAULT_BACKEND_URL = os.getenv("VG_BACKEND_URL", "http://localhost:8000")

@dataclass
class APIResult:
    """
    Standardized result object returned by all api_client methods.
    Ensures Streamlit never crashes on HTTP or network errors.
    """
    ok: bool
    status_code: int
    data: Optional[Any] = None
    error: Optional[str] = None
    raw_text: str = ""
    elapsed_ms: float = 0.0
    endpoint: str = ""
    method: str = "GET"

class VisionGuardAPIClient:
    """
    Client for the VisionGuard HTTP and WebSocket API.
    All network calls have explicit timeouts and retry-free execution.
    """
    def __init__(self, base_url: str = DEFAULT_BACKEND_URL, default_timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.default_timeout = default_timeout
        self.session = requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> APIResult:
        """Central helper with timing and error trapping."""
        url = f"{self.base_url}{path}"
        req_timeout = timeout or self.default_timeout
        start_time = time.perf_counter()

        try:
            resp = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                files=files,
                timeout=req_timeout,
                headers=headers
            )
            elapsed = (time.perf_counter() - start_time) * 1000.0

            # Try parsing JSON response
            data = None
            try:
                data = resp.json()
            except Exception:
                data = None

            is_ok = 200 <= resp.status_code < 300
            error_msg = None if is_ok else f"HTTP {resp.status_code}: {resp.text[:300]}"

            return APIResult(
                ok=is_ok,
                status_code=resp.status_code,
                data=data if data is not None else resp.content,
                error=error_msg,
                raw_text=resp.text,
                elapsed_ms=round(elapsed, 2),
                endpoint=path,
                method=method
            )
        except (requests.exceptions.RequestException, Exception) as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return APIResult(
                ok=False,
                status_code=0,
                data=None,
                error=f"Network error ({type(exc).__name__}): {str(exc)}",
                raw_text=str(exc),
                elapsed_ms=round(elapsed, 2),
                endpoint=path,
                method=method
            )

    # -----------------------------------------------------------------------
    # System & Health
    # -----------------------------------------------------------------------

    def get_health(self) -> APIResult:
        """Calls GET /health."""
        return self._request("GET", "/health")

    def get_vision_status(self) -> APIResult:
        """
        Calls GET /vision/status with fallback to GET /health.
        Returns model availability info.
        """
        res = self._request("GET", "/vision/status")
        if res.ok and res.data:
            return res

        # Fallback to /health which includes vision wrappers information
        health_res = self.get_health()
        if health_res.ok and isinstance(health_res.data, dict):
            wrappers = health_res.data.get("wrappers", {})
            normalized_status = {
                "backend_mode": health_res.data.get("backend_mode", "stub"),
                "models": wrappers,
                "llm": health_res.data.get("llm_multimodal", {})
            }
            return APIResult(
                ok=True,
                status_code=health_res.status_code,
                data=normalized_status,
                raw_text=health_res.raw_text,
                elapsed_ms=health_res.elapsed_ms,
                endpoint="/vision/status [derived]",
                method="GET"
            )
        return res

    def get_schema(self) -> APIResult:
        """Calls GET /circuits/schema."""
        return self._request("GET", "/circuits/schema")

    def get_templates(self) -> APIResult:
        """Calls GET /templates."""
        return self._request("GET", "/templates")

    # -----------------------------------------------------------------------
    # Circuit Analysis
    # -----------------------------------------------------------------------

    def analyze_image(
        self,
        image_bytes: bytes,
        filename: str = "image.jpg",
        connections: str = "proximity,template",
        board_template: str = "arduino-uno",
        inspect: bool = True,
        target: Optional[str] = None
    ) -> APIResult:
        """
        Calls POST /analyze (with fallback to POST /circuits/analyze).
        Timeout: 60s.
        """
        params: Dict[str, Any] = {
            "connections": connections,
            "board_template": board_template,
            "inspect": str(inspect).lower(),
        }
        if target:
            params["target"] = target

        files = {"file": (filename, image_bytes, "image/jpeg")}

        # Try /analyze first
        res = self._request("POST", "/analyze", params=params, files=files, timeout=60.0)
        if res.status_code == 404:
            # Fall back to /circuits/analyze
            files = {"file": (filename, image_bytes, "image/jpeg")}
            res = self._request("POST", "/circuits/analyze", params=params, files=files, timeout=60.0)
        return res

    # -----------------------------------------------------------------------
    # Circuit CRUD
    # -----------------------------------------------------------------------

    def create_circuit(self, payload: Optional[Dict[str, Any]] = None) -> APIResult:
        """Calls POST /circuits."""
        return self._request("POST", "/circuits", json_data=payload or {})

    def get_circuit(self, circuit_id: str) -> APIResult:
        """Calls GET /circuits/{id}."""
        return self._request("GET", f"/circuits/{circuit_id}")

    def update_circuit(self, circuit_id: str, payload: Dict[str, Any]) -> APIResult:
        """
        Calls PUT /circuits/{id}.
        User edits set source='user', conf=1.0, confirmed=True.
        """
        return self._request("PUT", f"/circuits/{circuit_id}", json_data=payload)

    # -----------------------------------------------------------------------
    # Validation & Explanation
    # -----------------------------------------------------------------------

    def validate_circuit(self, circuit_id: str, target: Optional[str] = None) -> APIResult:
        """
        Runs validation against circuit ID.
        Tries POST /circuits/{id}/validate, then GET /circuits/{id}/validate.
        """
        params = {}
        if target:
            params["target"] = target
            params["reference"] = target

        # Try POST first
        res = self._request("POST", f"/circuits/{circuit_id}/validate", params=params)
        if res.status_code in (404, 405):
            res = self._request("GET", f"/circuits/{circuit_id}/validate", params=params)
        return res

    def explain_circuit(
        self,
        circuit_id: str,
        target: Optional[str] = None,
        findings: Optional[List[Dict[str, Any]]] = None
    ) -> APIResult:
        """
        Calls POST /explain (with fallback to POST /circuits/{id}/explain).
        """
        params = {"target": target} if target else {}
        payload = {"circuit_id": circuit_id, "findings": findings or []}

        # Try /explain first
        res = self._request("POST", "/explain", params=params, json_data=payload)
        if res.status_code in (404, 405):
            res = self._request("POST", f"/circuits/{circuit_id}/explain", params=params, json_data=payload)
        return res

    # -----------------------------------------------------------------------
    # Exports
    # -----------------------------------------------------------------------

    def export_circuit(self, circuit_id: str, format_name: str) -> APIResult:
        """
        Exports circuit in given format: svg, png, or json.
        Tries POST /circuits/{id}/export?format=..., then GET /circuits/{id}/export/{format}.
        """
        fmt = format_name.lower().strip()
        res = self._request("POST", f"/circuits/{circuit_id}/export", params={"format": fmt})
        if res.status_code in (404, 405):
            res = self._request("GET", f"/circuits/{circuit_id}/export/{fmt}")
        return res

    def export_wokwi(self, circuit_id: str) -> APIResult:
        """Calls Wokwi diagram export."""
        res = self._request("POST", f"/circuits/{circuit_id}/export/wokwi")
        if res.status_code in (404, 405):
            res = self._request("GET", f"/circuits/{circuit_id}/export/wokwi")
        return res

    def get_overlay(self, circuit_id: str, format_name: str = "png") -> APIResult:
        """Calls GET /circuits/{id}/overlay?format=png to retrieve annotated image bytes."""
        return self._request("GET", f"/circuits/{circuit_id}/overlay", params={"format": format_name})

    def diagnose(
        self,
        circuit_id: str,
        question: Optional[str] = None,
        mode: str = "troubleshoot"
    ) -> APIResult:
        """Calls POST /diagnose with Featherless diagnostic assistant."""
        payload: Dict[str, Any] = {"circuit_id": circuit_id, "mode": mode}
        if question:
            payload["question"] = question
        return self._request("POST", "/diagnose", json_data=payload, timeout=25.0)

    # -----------------------------------------------------------------------
    # Live Camera & Frames
    # -----------------------------------------------------------------------

    def send_live_frame(self, circuit_id: str, frame_bytes: bytes) -> APIResult:
        """Sends a single live video frame to POST /circuits/{id}/live-frame."""
        files = {"file": ("frame.jpg", frame_bytes, "image/jpeg")}
        return self._request("POST", f"/circuits/{circuit_id}/live-frame", files=files)

    def test_websocket_live(self, sample_frame_bytes: bytes, timeout: float = 3.0) -> Tuple[bool, str, float]:
        """
        Attempts testing WebSocket live frame interface at WS /ws/live.
        Returns: (success: bool, message: str, latency_ms: float)
        """
        try:
            import websocket
            ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/live"
            t0 = time.perf_counter()

            ws = websocket.create_connection(ws_url, timeout=timeout)
            ws.send_binary(sample_frame_bytes)
            reply = ws.recv()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            ws.close()
            return True, f"WS connected, response: {str(reply)[:100]}", round(elapsed_ms, 2)
        except Exception as exc:
            return False, f"WebSocket /ws/live unavailable: {str(exc)}", 0.0

    # -----------------------------------------------------------------------
    # Slot Inspection Subprocess / API
    # -----------------------------------------------------------------------

    def inspect_slots_direct(self, crop_bytes: bytes, slot_name: str = "slot_test") -> Dict[str, Any]:
        """
        Calls local slot inspector wrapper if available, or backend route.
        """
        try:
            from visionguard.vision.slot_inspector import SlotInspector
            inspector = SlotInspector()
            slots = [{"name": slot_name, "bbox": [0.0, 0.0, 1.0, 1.0]}]
            res = inspector.inspect_slots(crop_bytes, template_slots=slots)
            if res:
                return {"ok": True, "results": res, "error": None}
            return {"ok": False, "results": [], "error": "No results"}
        except Exception as e:
            return {"ok": False, "results": [], "error": str(e)}
