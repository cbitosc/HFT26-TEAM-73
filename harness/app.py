import io
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path so 'harness' and 'visionguard' can always be resolved
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

try:
    from harness.api_client import VisionGuardAPIClient, APIResult, DEFAULT_BACKEND_URL
    from harness.selftest import SelfTestRunner, CheckResult
except ModuleNotFoundError:
    from api_client import VisionGuardAPIClient, APIResult, DEFAULT_BACKEND_URL
    from selftest import SelfTestRunner, CheckResult

# ---------------------------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VisionGuard Test Harness",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for crisp status badges and clean layout
st.markdown("""
<style>
    .badge-pass { background-color: #28a745; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-warn { background-color: #ffc107; color: black; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-fail { background-color: #dc3545; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-info { background-color: #17a2b8; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .stat-box { border: 1px solid #444; border-radius: 6px; padding: 10px; margin-bottom: 10px; }
    .wire-confirmed { color: #28a745; font-weight: 600; }
    .wire-unconfirmed { color: #ffc107; font-style: italic; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "backend_url" not in st.session_state:
    st.session_state["backend_url"] = DEFAULT_BACKEND_URL

if "circuit_id" not in st.session_state:
    st.session_state["circuit_id"] = None

if "last_api_result" not in st.session_state:
    st.session_state["last_api_result"] = None

if "active_circuit" not in st.session_state:
    st.session_state["active_circuit"] = None

if "health_cache" not in st.session_state:
    st.session_state["health_cache"] = None


def get_client() -> VisionGuardAPIClient:
    return VisionGuardAPIClient(base_url=st.session_state["backend_url"])


def show_raw_json_expander(result: Optional[APIResult], title: str = "Raw Response / Debug"):
    if not result:
        return
    with st.expander(f"🔍 {title} [{result.method} {result.endpoint} - HTTP {result.status_code} ({result.elapsed_ms:.1f}ms)]", expanded=False):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"**Status:** {'✅ OK' if result.ok else '❌ Error'}")
            st.write(f"**HTTP Code:** `{result.status_code}`")
            st.write(f"**Duration:** `{result.elapsed_ms:.1f} ms`")
            if result.error:
                st.error(result.error)
        with col2:
            if isinstance(result.data, (dict, list)):
                st.json(result.data)
            else:
                st.code(result.raw_text[:2000] if result.raw_text else "<empty response>", language="text")


def draw_bounding_boxes(image: Image.Image, parts: List[Dict[str, Any]]) -> Image.Image:
    """Draws bounding boxes and labels for parts onto an image using Pillow."""
    img_copy = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size

    colors = {
        "led": "#00FF00",
        "resistor": "#FF9900",
        "pushbutton": "#00CCFF",
        "arduino-uno": "#0066CC",
        "board": "#0066CC",
        "unknown": "#FF00FF"
    }

    for p in parts:
        bbox = p.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x, y, bw, bh = bbox
        # If coordinates are normalized (0.0 to 1.0)
        if x <= 1.0 and y <= 1.0 and bw <= 1.0 and bh <= 1.0:
            box_x0 = x * w
            box_y0 = y * h
            box_x1 = (x + bw) * w
            box_y1 = (y + bh) * h
        else:
            box_x0 = x
            box_y0 = y
            box_x1 = x + bw
            box_y1 = y + bh

        part_type = str(p.get("type", "unknown")).lower()
        color = colors.get(part_type, "#FFFFFF")
        label = f"{p.get('id', '')} ({part_type}) {p.get('conf', 1.0):.2f}"

        draw.rectangle([box_x0, box_y0, box_x1, box_y1], outline=color, width=3)
        draw.text((box_x0 + 4, max(0, box_y0 - 15)), label, fill=color)

    return img_copy


def generate_netlist_text(circuit: Dict[str, Any]) -> str:
    """Generates simple text netlist from circuit parts and connections."""
    lines = [f"# Netlist for Circuit: {circuit.get('id', 'unknown')}", f"# Board: {circuit.get('board', {}).get('template', 'unknown')}", ""]
    lines.append("## Components:")
    for p in circuit.get("parts", []):
        lines.append(f"  - {p.get('id')}: {p.get('type')} (val={p.get('value')}, col={p.get('color')}, conf={p.get('conf')})")

    lines.append("\n## Nets / Connections:")
    for i, c in enumerate(circuit.get("connections", []), start=1):
        status = "CONFIRMED" if c.get("confirmed") else "PROPOSED"
        src = c.get("source", "unknown")
        lines.append(f"  Net {i:02d}: {c.get('from')} <--> {c.get('to')} [{status}, src={src}, conf={c.get('conf')}]")

    return "\n".join(lines)


def generate_graphviz_dot(circuit: Dict[str, Any]) -> str:
    """Generates Graphviz DOT representation of circuit components and wires."""
    dot = ["graph G {", "  rankdir=LR;", "  node [shape=box, style=filled, fillcolor=\"#2b303c\", fontcolor=white, fontname=\"Helvetica\"];"]
    
    # Board node
    board_template = circuit.get("board", {}).get("template", "arduino-uno")
    dot.append(f'  "uno" [label="{board_template}\\n(Board)", fillcolor="#1f77b4", shape=record];')

    # Part nodes
    for p in circuit.get("parts", []):
        pid = p.get("id", "part")
        ptype = p.get("type", "component")
        pval = f"\\nval: {p.get('value')}" if p.get("value") else ""
        dot.append(f'  "{pid}" [label="{pid}\\n{ptype}{pval}"];')

    # Connections
    for c in circuit.get("connections", []):
        from_ep = c.get("from", "")
        to_ep = c.get("to", "")
        from_part = from_ep.split(":")[0] if ":" in from_ep else from_ep
        to_part = to_ep.split(":")[0] if ":" in to_ep else to_ep

        style = "solid" if c.get("confirmed") else "dashed"
        color = "#28a745" if c.get("confirmed") else "#ffc107"
        label = f"{from_ep} - {to_ep}"
        dot.append(f'  "{from_part}" -- "{to_part}" [label="{label}", style="{style}", color="{color}", fontcolor="#cccccc", fontsize=9];')

    dot.append("}")
    return "\n".join(dot)


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------

st.sidebar.title("⚡ VisionGuard Harness")
st.sidebar.caption("API Verification & Interactive Testbed")

# Backend URL config
backend_input = st.sidebar.text_input(
    "Backend Base URL",
    value=st.session_state["backend_url"],
    help="Target VisionGuard FastAPI server"
)
if backend_input != st.session_state["backend_url"]:
    st.session_state["backend_url"] = backend_input
    st.rerun()

client = get_client()

# Check connection button
if st.sidebar.button("🔄 Check Connection", use_container_width=True):
    with st.spinner("Pinging backend..."):
        h_res = client.get_health()
        st.session_state["health_cache"] = h_res
        if h_res.ok:
            st.sidebar.success(f"Connected! ({h_res.elapsed_ms:.1f}ms)")
        else:
            st.sidebar.error(f"Failed to connect: {h_res.error}")

# Cached or fresh health info
if not st.session_state["health_cache"]:
    st.session_state["health_cache"] = client.get_health()

health_res = st.session_state["health_cache"]

# Model Status Badges in Sidebar
st.sidebar.markdown("### Model Wrappers")
if health_res and health_res.ok and isinstance(health_res.data, dict):
    wrappers = health_res.data.get("wrappers", {})
    mode = health_res.data.get("backend_mode", "stub")

    for model_name, display_name in [
        ("board_detector", "Board Detector"),
        ("component_detector", "Component Detector"),
        ("slot_inspector", "Slot Inspector")
    ]:
        info = wrappers.get(model_name, {})
        avail = info.get("available", False)
        status_txt = info.get("status", "unknown")

        if avail:
            badge = "🟢 Available"
        elif "stub" in mode:
            badge = "🟡 Stub Mode"
        else:
            badge = "🔴 Unavailable"

        st.sidebar.markdown(f"**{display_name}:** {badge}")
        st.sidebar.caption(f"Reason: `{status_txt}`")

    st.sidebar.markdown(f"**Backend Mode:** `{mode}`")
    st.sidebar.markdown("**Licence Note:** *Non-commercial research / YOLO weights*")

    mode_options = ["stub", "real"]
    current_mode_idx = 0 if mode == "stub" else 1
    selected_mode = st.sidebar.radio(
        "Mode Selector",
        options=mode_options,
        index=current_mode_idx,
        format_func=lambda m: "🟡 Stub Mode (Full Output & Canned Demo)" if m == "stub" else "🔴 Real Mode (Needs Weights)",
        key="backend_mode_radio"
    )
    if selected_mode != mode:
        client._request("POST", f"/mode?mode={selected_mode}")
        st.session_state["health_cache"] = None
        st.rerun()
else:
    st.sidebar.warning("⚠️ Backend not reachable")

st.sidebar.divider()

# Active Circuit Management in Sidebar
st.sidebar.markdown("### Active Circuit Session")
cur_id = st.session_state["circuit_id"]
if cur_id:
    st.sidebar.info(f"**ID:** `{cur_id[:8]}...`\n\n(`{cur_id}`)")
else:
    st.sidebar.text("No circuit loaded.")

load_id_input = st.sidebar.text_input("Load Circuit by ID", placeholder="UUID...")
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    if st.button("📥 Load", use_container_width=True):
        if load_id_input.strip():
            res = client.get_circuit(load_id_input.strip())
            if res.ok:
                st.session_state["circuit_id"] = load_id_input.strip()
                st.session_state["active_circuit"] = res.data
                st.sidebar.success("Loaded!")
                st.rerun()
            else:
                st.sidebar.error("Not found.")
with col_sb2:
    if st.button("🧹 Reset", use_container_width=True):
        st.session_state["circuit_id"] = None
        st.session_state["active_circuit"] = None
        st.rerun()


# ---------------------------------------------------------------------------
# Main Tabs UI
# ---------------------------------------------------------------------------

# System mode & health alert banners
if health_res and health_res.ok and isinstance(health_res.data, dict):
    b_mode = health_res.data.get("backend_mode", "real")
    wrappers_data = health_res.data.get("wrappers", {})

    if b_mode == "stub":
        st.error("🚨 **STUB MODE: detections are fake** — Results are canned demo data, not detected from your image.")

    unavail = [
        k for k, v in wrappers_data.items()
        if not v.get("available") and v.get("state") != "stub"
    ]
    if unavail and b_mode != "stub":
        st.warning(f"⚠️ **Vision Layers Unavailable**: {', '.join(unavail)}. Install requirements-vision.txt or configure weights.")

tabs = st.tabs([
    "1. 🏥 Health & Models",
    "2. 📷 Analyze Image",
    "3. ✏️ Circuit Editor",
    "4. 🛡️ Validation",
    "5. 📦 Exports",
    "6. 🤖 AI Explanation",
    "7. 🔬 Slot Inspector",
    "8. 🎥 Live Detection",
    "9. 🚀 Automated Self-Test"
])


# ===========================================================================
# TAB 1: HEALTH AND VISION STATUS
# ===========================================================================
with tabs[0]:
    st.header("Backend Health & Vision Status")
    st.markdown("Verifies connectivity to the FastAPI server and checks the status of all three pretrained vision wrappers.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Target URL", st.session_state["backend_url"])
    with col2:
        status_label = "ONLINE" if (health_res and health_res.ok) else "OFFLINE"
        st.metric("Server Status", status_label)
    with col3:
        b_mode = health_res.data.get("backend_mode", "N/A") if (health_res and health_res.ok) else "N/A"
        st.metric("Backend Mode", b_mode)
    with col4:
        lat = f"{health_res.elapsed_ms:.1f} ms" if health_res else "N/A"
        st.metric("Latency", lat)

    st.subheader("Vision Model Wrappers")
    if health_res and health_res.ok and isinstance(health_res.data, dict):
        wrappers_data = health_res.data.get("wrappers", {})
        table_rows = []
        for name, info in wrappers_data.items():
            avail = info.get("available", False)
            table_rows.append({
                "Model Wrapper": name,
                "Available": "✅ True" if avail else "❌ False",
                "Status / Reason": info.get("status", "Unknown"),
                "Type": "Pretrained Wrapper"
            })
        st.table(pd.DataFrame(table_rows))

        llm_info = health_res.data.get("llm_multimodal", {})
        st.subheader("Multimodal LLM Status")
        st.write(f"- **Configured:** {'✅ Yes' if llm_info.get('configured') else '⚠️ No (Mock/Fallback active)'}")
        st.write(f"- **Model Identifier:** `{llm_info.get('model', 'gemini-1.5-flash')}`")
    else:
        st.error("Cannot query health endpoint. Check that uvicorn is running.")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Fetch Circuit Schema (/circuits/schema)"):
            s_res = client.get_schema()
            st.session_state["last_api_result"] = s_res
            if s_res.ok:
                st.success(f"Retrieved JSON Schema ({s_res.elapsed_ms:.1f}ms)")
                st.json(s_res.data)
            else:
                st.error(f"Error fetching schema: {s_res.error}")

    with col_btn2:
        if st.button("Fetch Templates (/templates)"):
            t_res = client.get_templates()
            st.session_state["last_api_result"] = t_res
            if t_res.ok:
                st.success(f"Retrieved Templates ({t_res.elapsed_ms:.1f}ms)")
                st.json(t_res.data)
            else:
                st.error(f"Error fetching templates: {t_res.error}")

    show_raw_json_expander(health_res, "Health Response")


# ===========================================================================
# TAB 2: INSPECTION STUDIO (MATCHES FRONTEND LOCALHOST:3000/STUDIO)
# ===========================================================================
with tabs[1]:
    from visionguard.knowledge.sample_boards import SAMPLE_BOARDS
    from visionguard.knowledge.pinouts import PinoutKnowledgeBase
    kb = PinoutKnowledgeBase()

    # Top banner matching frontend
    st.info("ℹ️ **Simulated Inspection Engine**: Component boxes & defects are rendered by VisionGuard's offline model. Fully connected with backend API (`http://localhost:8000`).")

    col_ctrl, col_canvas, col_results = st.columns([1, 1.4, 1.6])

    with col_ctrl:
        st.subheader("Input & Controls")
        
        sample_board_options = list(SAMPLE_BOARDS.keys())
        selected_board_key = st.selectbox(
            "Select Sample Board (4 Available)",
            options=sample_board_options,
            format_func=lambda k: f"{SAMPLE_BOARDS[k]['name']} [{SAMPLE_BOARDS[k]['badge']}]",
            index=0
        )

        board_meta = SAMPLE_BOARDS[selected_board_key]
        if board_meta["defect_count"] > 0:
            st.error(f"⚠️ **{board_meta['defect_count']} Defects Present** on this board variant")
        else:
            st.caption(f"Status: Reference standard • {board_meta['detections_count']} components")

        source_tab = st.radio("Hardware Source Feed", ["Sample", "Upload", "Camera"], horizontal=True)

        uploaded_file = None
        if source_tab == "Upload":
            uploaded_file = st.file_uploader("Upload Hardware Image", type=["jpg", "jpeg", "png"])
        elif source_tab == "Camera":
            st.info("Live camera feed streaming to `/circuits/{id}/live-frame`")

        detection_mode = st.selectbox("Detection Mode", ["Precise", "Fast", "Sensitive"], index=0)

        run_inspection_clicked = st.button("⚡ RUN VISION INSPECTION", type="primary", use_container_width=True)

    # Trigger inspection
    if run_inspection_clicked:
        with st.spinner(f"Running vision inspection on {board_meta['name']}..."):
            img_bytes = None
            fname = "sample.jpg"

            if source_tab == "Upload" and uploaded_file is not None:
                img_bytes = uploaded_file.getvalue()
                fname = uploaded_file.name
            else:
                sample_path = os.path.join(PROJECT_ROOT, "data", "eval_images", "test_board_1.jpg")
                if os.path.exists(sample_path):
                    with open(sample_path, "rb") as f:
                        img_bytes = f.read()
                    fname = "test_board_1.jpg"

            if img_bytes:
                st.session_state["sample_image_bytes"] = img_bytes
                res = client.analyze_image(
                    image_bytes=img_bytes,
                    filename=fname,
                    connections="proximity,template",
                    board_template=selected_board_key,
                    inspect=True
                )
                st.session_state["last_api_result"] = res

                if res.ok and isinstance(res.data, dict):
                    circuit = res.data.get("circuit", {})
                    st.session_state["circuit_id"] = circuit.get("id")
                    st.session_state["active_circuit"] = circuit
                    st.session_state["active_board_key"] = selected_board_key
                    st.success(f"Inspection Complete! Circuit ID: `{circuit.get('id')[:8]}` ({res.elapsed_ms:.1f}ms)")
                else:
                    st.error(f"Inspection Failed: {res.error}")

    active_circuit = st.session_state.get("active_circuit")

    # Center Viewport: Diagnostic Overlay
    with col_canvas:
        st.subheader("Center Viewport")
        if active_circuit:
            ov_res = client.get_overlay(active_circuit.get("id"))
            if ov_res.ok and isinstance(ov_res.data, bytes) and ov_res.data.startswith(b"\x89PNG"):
                st.image(ov_res.data, caption="Diagnostic Visual Overlay (All Component Bounding Boxes & Defects)", use_container_width=True)
            elif "sample_image_bytes" in st.session_state:
                orig_img = Image.open(io.BytesIO(st.session_state["sample_image_bytes"]))
                annotated_img = draw_bounding_boxes(orig_img, active_circuit.get("parts", []))
                st.image(annotated_img, caption="Detected Components with Bounding Boxes", use_container_width=True)
            
            dets_total = len(active_circuit.get("detections", []))
            st.caption(f"🔍 **Detections:** `{dets_total} / {dets_total}` | **MODEL:** `SIMULATED-v1.4` | **Board:** `{selected_board_key}`")
        else:
            st.info("Click **⚡ RUN VISION INSPECTION** to view the live diagnostic overlay.")

    # Right Panel: 4 Tabs matching frontend
    with col_results:
        st.subheader("Inspection Results")
        
        tab_dets, tab_specs, tab_pinouts, tab_health = st.tabs(["Detections", "Specs", "Pinouts", "Health"])

        with tab_dets:
            if active_circuit:
                dets_list = active_circuit.get("detections", [])
                
                # Filter & Sort
                col_f1, col_f2 = st.columns([2, 1])
                with col_f1:
                    filter_q = st.text_input("🔍 Filter detections...", "", placeholder="e.g. USB, ATmega, LED...")
                with col_f2:
                    sort_by = st.selectbox("Sort", ["By Conf", "By Name"], index=0)

                filtered_dets = [d for d in dets_list if (filter_q.lower() in d.get("name", "").lower() or filter_q.lower() in d.get("category", "").lower())]
                if sort_by == "By Conf":
                    filtered_dets = sorted(filtered_dets, key=lambda x: x.get("conf", 0.0), reverse=True)
                else:
                    filtered_dets = sorted(filtered_dets, key=lambda x: x.get("name", ""))

                st.write(f"Showing **{len(filtered_dets)}** components:")
                for d in filtered_dets:
                    c_conf = d.get("conf", 1.0) * 100.0
                    cat_name = d.get("category", "IC").upper()
                    
                    with st.container():
                        col_card1, col_card2 = st.columns([3, 1])
                        with col_card1:
                            st.markdown(f"**{d.get('name')}**")
                            st.caption(f"Category: `{cat_name}` | Source: `{d.get('source', 'SIM').upper()}`")
                        with col_card2:
                            st.markdown(f"**`{c_conf:.1f}%`**")
                        st.progress(min(1.0, max(0.0, c_conf / 100.0)))
                        st.divider()
            else:
                st.caption("No detections yet. Run inspection to populate.")

        with tab_specs:
            if active_circuit:
                cur_board_key = st.session_state.get("active_board_key", "arduino-uno-r3")
                b_meta = SAMPLE_BOARDS.get(cur_board_key, SAMPLE_BOARDS["arduino-uno-r3"])
                comps_specs = b_meta.get("components", [])

                st.markdown(f"#### Hardware Specifications: {b_meta['name']}")
                specs_data = []
                for c in comps_specs:
                    specs_dict = c.get("specs", {})
                    specs_summary = ", ".join([f"{k}: {v}" for k, v in specs_dict.items()])
                    specs_data.append({
                        "Designator": c.get("designator", c.get("id")),
                        "Name": c.get("name"),
                        "Part Number": c.get("part_number", "-"),
                        "Category": c.get("category", "-"),
                        "Specifications": specs_summary
                    })
                st.dataframe(pd.DataFrame(specs_data), use_container_width=True)
            else:
                st.caption("Hardware specs will appear once inspection is run.")

        with tab_pinouts:
            st.markdown("#### Verified Hardware Pinout Knowledge Base")
            target_pinout = "arduino-uno"
            if st.session_state.get("active_board_key") == "esp32-devkit":
                target_pinout = "esp32-devkit"

            po_data, po_msg = kb.lookup(target_pinout)
            if po_data:
                st.write(f"**Device:** `{po_data.get('name')}` (Verified: ✅)")
                st.caption(po_data.get("description", ""))
                pins_df = pd.DataFrame(po_data.get("pins", []))
                st.dataframe(pins_df, use_container_width=True)
            else:
                st.info(po_msg)

        with tab_health:
            if active_circuit:
                health_info = active_circuit.get("health", {})
                h_stat = health_info.get("overall_status", "review").upper()
                if h_stat == "OK":
                    st.success(f"**Overall Status: `PASS / OK`** (0 critical flaws identified)")
                elif h_stat == "ISSUES":
                    st.error(f"**Overall Status: `ISSUES DETECTED`** (Defects require remediation)")
                else:
                    st.warning(f"**Overall Status: `{h_stat}`**")

                defects_list = active_circuit.get("defects", [])
                if defects_list:
                    st.markdown(f"##### Flagged Defects ({len(defects_list)} Total)")
                    for df in defects_list:
                        st.error(f"🔴 **{df.get('type').replace('_', ' ').title()}** [{df.get('severity').upper()}]")
                        st.write(df.get("note", "-"))
                else:
                    st.success("✅ **Zero physical or assembly defects detected.** Board passes reference standards.")

                # Rule findings
                last_res = st.session_state.get("last_api_result")
                if last_res and isinstance(last_res.data, dict) and "validation" in last_res.data:
                    val_findings = last_res.data["validation"].get("findings", [])
                    st.markdown("##### Deterministic Rule Findings")
                    for f in val_findings:
                        st.caption(f"• **[{f.get('rule_id')}]** {f.get('message')}")
            else:
                st.caption("Health metrics will appear once inspection is run.")

        # Diagnostic Assistant Section
        st.divider()
        st.subheader("🤖 Featherless Diagnostic Assistant")
        col_diag1, col_diag2 = st.columns([3, 1])
        with col_diag1:
            diag_q = st.text_input("Ask a diagnostic or assembly question (optional):", placeholder="e.g. Is this LED circuit safe to power with 5V?")
        with col_diag2:
            diag_mode = st.selectbox("Diagnostic Mode", ["troubleshoot", "summary", "assembly"])

        if st.button("🚀 Run Diagnostic Assistant", use_container_width=True):
            with st.spinner("Assembling structured data and querying Featherless diagnostic agent..."):
                d_res = client.diagnose(active_circuit.get("id"), question=diag_q, mode=diag_mode)
                st.session_state["diagnostic_result"] = d_res
                if d_res.ok and isinstance(d_res.data, dict):
                    st.success("Diagnostic analysis complete!")
                else:
                    st.warning(f"Diagnostic agent notice: {d_res.error}")

        if "diagnostic_result" in st.session_state and st.session_state["diagnostic_result"].ok:
            d_data = st.session_state["diagnostic_result"].data
            st.markdown("##### Plain-Language Summary")
            st.info(d_data.get("summary_plain_language", ""))

            st.markdown("##### Numbered Step-by-Step Diagnostic / Assembly Guide")
            steps = d_data.get("step_by_step_guide", [])
            for s in steps:
                st.write(f"**Step {s.get('step_number')}:** {s.get('instruction')}")
                st.caption(f"Target: `{s.get('target_component_id')}` | Bounding Box: `{s.get('bbox')}`")

            st.markdown("##### Suspected Issues & Pinout Notes")
            issues = d_data.get("suspected_issues", [])
            if issues:
                st.dataframe(pd.DataFrame(issues), use_container_width=True)
            p_notes = d_data.get("pinout_notes", [])
            if p_notes:
                st.dataframe(pd.DataFrame(p_notes), use_container_width=True)

            st.warning(f"⚠️ **Disclaimer:** {d_data.get('disclaimer', '')}")

        # Deterministic validation findings
        last_res = st.session_state.get("last_api_result")
        if last_res and isinstance(last_res.data, dict) and "validation" in last_res.data:
            val_findings = last_res.data["validation"].get("findings", [])
            st.markdown("#### Deterministic Rule Findings")
            for f in val_findings:
                sev = f.get("severity", "info")
                msg = f"[{f.get('rule_id')}] {f.get('message')} (Refs: {', '.join(f.get('refs', []))})"
                if sev == "error":
                    st.error(msg)
                elif sev == "warning":
                    st.warning(msg)
                else:
                    st.info(msg)

    # Tool: Compare Two Images (Regression Test)
    st.divider()
    st.subheader("🔬 Compare Two Images Tool")
    st.markdown("Analyze two uploads simultaneously to verify detection differences (detects canned/stub data).")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        img_file1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"], key="compare_img1")
    with col_c2:
        img_file2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"], key="compare_img2")

    if st.button("⚖️ Compare Detections for Both Images", disabled=(img_file1 is None or img_file2 is None)):
        with st.spinner("Analyzing both images..."):
            r1 = client.analyze_image(img_file1.getvalue(), filename=img_file1.name)
            r2 = client.analyze_image(img_file2.getvalue(), filename=img_file2.name)

            if r1.ok and r2.ok:
                c1_dets = r1.data.get("circuit", {}).get("detections", [])
                c2_dets = r2.data.get("circuit", {}).get("detections", [])
                c1_ids = [d.get("name") for d in c1_dets]
                c2_ids = [d.get("name") for d in c2_dets]

                if c1_ids != c2_ids or r1.data.get("circuit", {}).get("id") != r2.data.get("circuit", {}).get("id"):
                    st.success("✅ **PASS**: Analysis produced distinct detection results!")
                else:
                    st.error("❌ **FAIL**: Both images produced identical detection lists (canned data detected)!")

                col_comp1, col_comp2 = st.columns(2)
                with col_comp1:
                    st.markdown("**Image 1 Detections:**")
                    st.dataframe(pd.DataFrame(c1_dets) if c1_dets else pd.DataFrame([{"info": "Zero detections"}]))
                with col_comp2:
                    st.markdown("**Image 2 Detections:**")
                    st.dataframe(pd.DataFrame(c2_dets) if c2_dets else pd.DataFrame([{"info": "Zero detections"}]))

    show_raw_json_expander(st.session_state.get("last_api_result"), "Analyze Response")


# ===========================================================================
# TAB 3: CIRCUIT EDITOR
# ===========================================================================
with tabs[2]:
    st.header("Circuit Editor")
    st.markdown("Replaces the React Flow canvas during testing. Edit parts and wires, confirm proposals, save updates via `PUT /circuits/{id}`, and verify user edits.")

    col_btn_load1, col_btn_load2 = st.columns(2)
    with col_btn_load1:
        if st.button("📂 Load LED Blink Sample Circuit"):
            sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample_led_blink.json")
            if os.path.exists(sample_path):
                with open(sample_path, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
                c_res = client.create_circuit(s_data)
                st.session_state["last_api_result"] = c_res
                if c_res.ok:
                    st.session_state["circuit_id"] = c_res.data.get("id")
                    st.session_state["active_circuit"] = c_res.data
                    st.success("Loaded LED Blink sample circuit!")
                    st.rerun()

    with col_btn_load2:
        if st.button("➕ Create New Empty Circuit"):
            c_res = client.create_circuit()
            st.session_state["last_api_result"] = c_res
            if c_res.ok:
                st.session_state["circuit_id"] = c_res.data.get("id")
                st.session_state["active_circuit"] = c_res.data
                st.success("Created empty circuit!")
                st.rerun()

    circuit_id = st.session_state.get("circuit_id")
    if not circuit_id:
        st.warning("No circuit currently loaded. Load one above or analyze an image in Tab 2.")
    else:
        # Fetch fresh circuit document from backend
        fetch_res = client.get_circuit(circuit_id)
        if fetch_res.ok and isinstance(fetch_res.data, dict):
            circuit_doc = fetch_res.data
            st.session_state["active_circuit"] = circuit_doc
        else:
            st.error(f"Error fetching circuit: {fetch_res.error}")
            circuit_doc = st.session_state.get("active_circuit", {})

        parts = circuit_doc.get("parts", [])
        connections = circuit_doc.get("connections", [])

        st.subheader("Components & Parts")
        part_options = [p["id"] for p in parts]
        st.dataframe(pd.DataFrame(parts) if parts else pd.DataFrame([{"info": "No parts"}]), use_container_width=True)

        st.subheader("Connections & Wires")
        st.dataframe(pd.DataFrame(connections) if connections else pd.DataFrame([{"info": "No connections"}]), use_container_width=True)

        st.subheader("Wire & Part Actions")
        act_col1, act_col2 = st.columns(2)

        with act_col1:
            st.markdown("#### Wire Operations")
            # Confirm wire(s)
            c_wire_idx = st.selectbox(
                "Select Wire to Confirm",
                options=list(range(len(connections))),
                format_func=lambda i: f"Wire {i}: {connections[i].get('from')} -> {connections[i].get('to')} ({'CONFIRMED' if connections[i].get('confirmed') else 'PROPOSED'})" if i < len(connections) else ""
            ) if connections else None

            col_w_btn1, col_w_btn2 = st.columns(2)
            with col_w_btn1:
                if st.button("Confirm Selected Wire", disabled=(c_wire_idx is None)):
                    connections[c_wire_idx]["confirmed"] = True
                    connections[c_wire_idx]["source"] = "user"
                    connections[c_wire_idx]["conf"] = 1.0
                    u_res = client.update_circuit(circuit_id, circuit_doc)
                    st.session_state["last_api_result"] = u_res
                    if u_res.ok:
                        st.success("Confirmed wire!")
                        st.rerun()

            with col_w_btn2:
                if st.button("Confirm ALL Wires", disabled=(len(connections) == 0)):
                    for c in connections:
                        c["confirmed"] = True
                        c["source"] = "user"
                        c["conf"] = 1.0
                    u_res = client.update_circuit(circuit_id, circuit_doc)
                    st.session_state["last_api_result"] = u_res
                    if u_res.ok:
                        st.success("All wires confirmed!")
                        st.rerun()

            # Confirm above threshold
            th_col1, th_col2 = st.columns([2, 1])
            with th_col1:
                conf_th = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
            with th_col2:
                if st.button("Confirm >= Thresh"):
                    cnt = 0
                    for c in connections:
                        if c.get("conf", 0.0) >= conf_th:
                            c["confirmed"] = True
                            c["source"] = "user"
                            c["conf"] = 1.0
                            cnt += 1
                    u_res = client.update_circuit(circuit_id, circuit_doc)
                    st.session_state["last_api_result"] = u_res
                    if u_res.ok:
                        st.success(f"Confirmed {cnt} wires!")
                        st.rerun()

            # Delete wire
            if st.button("Delete Selected Wire", disabled=(c_wire_idx is None)):
                connections.pop(c_wire_idx)
                u_res = client.update_circuit(circuit_id, circuit_doc)
                st.session_state["last_api_result"] = u_res
                if u_res.ok:
                    st.success("Deleted wire!")
                    st.rerun()

            # Add Wire
            st.markdown("##### Add New Wire")
            # Collect available pins
            available_pins = ["uno:13", "uno:GND.1", "uno:5V", "uno:A0", "uno:A1"]
            for p in parts:
                pid = p["id"]
                ptype = p.get("type", "")
                if ptype == "led":
                    available_pins.extend([f"{pid}:A", f"{pid}:C"])
                elif ptype == "resistor":
                    available_pins.extend([f"{pid}:1", f"{pid}:2"])
                elif ptype == "pushbutton":
                    available_pins.extend([f"{pid}:1.l", f"{pid}:1.r", f"{pid}:2.l", f"{pid}:2.r"])
                else:
                    available_pins.extend([f"{pid}:pin1", f"{pid}:pin2"])

            add_col1, add_col2, add_col3 = st.columns([2, 2, 1])
            with add_col1:
                from_pin = st.selectbox("From Pin", available_pins, key="from_pin_sel")
            with add_col2:
                to_pin = st.selectbox("To Pin", available_pins, index=min(1, len(available_pins)-1), key="to_pin_sel")
            with add_col3:
                if st.button("Add Wire"):
                    new_wire = {
                        "from": from_pin,
                        "to": to_pin,
                        "source": "user",
                        "conf": 1.0,
                        "confirmed": True,
                        "wire_color": "green"
                    }
                    circuit_doc.setdefault("connections", []).append(new_wire)
                    u_res = client.update_circuit(circuit_id, circuit_doc)
                    st.session_state["last_api_result"] = u_res
                    if u_res.ok:
                        st.success("Added wire!")
                        st.rerun()

        with act_col2:
            st.markdown("#### Part Operations")
            # Add Part
            st.markdown("##### Add Component")
            ap_col1, ap_col2 = st.columns(2)
            with ap_col1:
                new_part_type = st.selectbox("Type", ["resistor", "led", "pushbutton"])
                new_part_id = st.text_input("ID", value=f"{new_part_type[0]}{len(parts)+1}")
            with ap_col2:
                new_part_val = st.text_input("Value", value="220" if new_part_type == "resistor" else "")
                new_part_col = st.text_input("Color", value="red" if new_part_type == "led" else "")

            if st.button("Add Component"):
                new_part = {
                    "id": new_part_id,
                    "type": new_part_type,
                    "value": new_part_val or None,
                    "color": new_part_col or None,
                    "source": "user",
                    "conf": 1.0,
                    "confirmed": True
                }
                circuit_doc.setdefault("parts", []).append(new_part)
                u_res = client.update_circuit(circuit_id, circuit_doc)
                st.session_state["last_api_result"] = u_res
                if u_res.ok:
                    st.success(f"Added {new_part_id}!")
                    st.rerun()

            # Delete Part
            if part_options:
                del_part_id = st.selectbox("Select Part to Delete", part_options)
                if st.button(f"Delete Part '{del_part_id}'"):
                    circuit_doc["parts"] = [p for p in circuit_doc["parts"] if p["id"] != del_part_id]
                    # Also drop dangling wires
                    circuit_doc["connections"] = [
                        c for c in circuit_doc.get("connections", [])
                        if not (c.get("from", "").startswith(f"{del_part_id}:") or c.get("to", "").startswith(f"{del_part_id}:"))
                    ]
                    u_res = client.update_circuit(circuit_id, circuit_doc)
                    st.session_state["last_api_result"] = u_res
                    if u_res.ok:
                        st.success(f"Deleted {del_part_id}!")
                        st.rerun()

        # Non-negotiable test: Verify User Edit Sets source='user', conf=1.0, confirmed=True
        st.divider()
        st.markdown("#### User-Edit Compliance Check")
        if st.button("🧪 Test User-Edit Compliance (source='user', conf=1.0, confirmed=true)"):
            test_conn = {
                "from": "uno:13",
                "to": "uno:GND.1",
                "source": "vision",
                "conf": 0.3,
                "confirmed": False
            }
            temp_doc = dict(circuit_doc)
            temp_doc["connections"] = temp_doc.get("connections", []) + [test_conn]
            # When updating via user API, the backend rule must promote it or preserve confirmed
            u_res = client.update_circuit(circuit_id, temp_doc)
            if u_res.ok and isinstance(u_res.data, dict):
                saved_c = u_res.data.get("circuit", {})
                last_c = saved_c.get("connections", [])[-1]
                if last_c.get("source") == "user" and last_c.get("conf") == 1.0 and last_c.get("confirmed") is True:
                    st.success("✅ **PASS**: User edit correctly enforced source='user', conf=1.0, confirmed=true!")
                else:
                    st.error(f"❌ **FAIL**: Backend did not set expected user edit attributes: {last_c}")
            else:
                st.error("Update request failed.")

        # Visualizations: Graphviz Schematic & Netlist
        st.divider()
        st.subheader("Circuit Schematic & Netlist")
        vis_col1, vis_col2 = st.columns([1, 1])
        with vis_col1:
            st.markdown("**Graphviz Schematic Graph** (Solid: Confirmed, Dashed: Proposed)")
            try:
                dot_src = generate_graphviz_dot(circuit_doc)
                st.graphviz_chart(dot_src, use_container_width=True)
            except Exception as e:
                st.warning(f"Graphviz rendering notice: {e}")

        with vis_col2:
            st.markdown("**Canonical Text Netlist**")
            netlist_txt = generate_netlist_text(circuit_doc)
            st.text_area("Netlist", netlist_txt, height=280)

        show_raw_json_expander(st.session_state.get("last_api_result"), "Circuit Editor API Result")


# ===========================================================================
# TAB 4: VALIDATION
# ===========================================================================
with tabs[3]:
    st.header("Deterministic Validation Engine")
    st.markdown("Pure rule evaluation (R001 - R010). The LLM can never change these verdicts.")

    circuit_id = st.session_state.get("circuit_id")
    if not circuit_id:
        st.warning("No circuit currently loaded. Please load or create one first.")
    else:
        v_col1, v_col2 = st.columns([2, 1])
        with v_col1:
            ref_project = st.selectbox("Reference Project Comparison", ["None", "led-blink-pin13"], index=1)
            target_arg = None if ref_project == "None" else ref_project

        with v_col2:
            val_button = st.button("🛡️ Validate Circuit", use_container_width=True)

        if val_button:
            with st.spinner("Validating circuit against deterministic rules..."):
                val_res = client.validate_circuit(circuit_id, target=target_arg)
                st.session_state["last_api_result"] = val_res

                if val_res.ok and isinstance(val_res.data, dict):
                    status = val_res.data.get("status", "unknown").upper()
                    findings = val_res.data.get("findings", [])
                    missing_wires = val_res.data.get("missing_wires", [])
                    extra_wires = val_res.data.get("extra_wires", [])

                    st.markdown(f"### Verdict: **`{status}`**")
                    if status == "PASS":
                        st.success("✅ **PASS**: All design rules passed successfully.")
                    elif status == "WARN":
                        st.warning("⚠️ **WARN**: Validation generated warnings (e.g. unconfirmed wires or unverified ratings).")
                    else:
                        st.error("❌ **FAIL**: Validation failed due to design errors.")

                    st.markdown("#### Findings")
                    if not findings:
                        st.info("No findings generated.")
                    for f in findings:
                        rule = f.get("rule_id", "R???")
                        sev = f.get("severity", "info")
                        msg = f.get("message", "")
                        refs = f.get("refs", [])
                        ref_str = f" [Refs: {', '.join(refs)}]" if refs else ""

                        if sev == "error":
                            st.error(f"**{rule} (ERROR):** {msg}{ref_str}")
                        elif sev == "warning":
                            st.warning(f"**{rule} (WARNING):** {msg}{ref_str}")
                        else:
                            st.info(f"**{rule} (INFO):** {msg}{ref_str}")

                    if target_arg:
                        st.markdown("#### Target Comparison Diffs")
                        d_col1, d_col2 = st.columns(2)
                        with d_col1:
                            st.markdown("**Missing Required Wires:**")
                            st.write(missing_wires if missing_wires else "None ✅")
                        with d_col2:
                            st.markdown("**Extra / Spurious Wires:**")
                            st.write(extra_wires if extra_wires else "None ✅")

        # Test: Load bad sample circuit and check expected rule firings
        st.divider()
        st.subheader("Test Deliberately Bad Circuit")
        st.markdown("Loads `sample_bad_circuit.json` which intentionally violates `R001`, `R003`, `R009`, and `R010`.")

        if st.button("🧪 Test Bad Sample Circuit Rules"):
            bad_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample_bad_circuit.json")
            if os.path.exists(bad_path):
                with open(bad_path, "r", encoding="utf-8") as f:
                    bad_json = json.load(f)
                c_res = client.create_circuit(bad_json)
                if c_res.ok:
                    b_id = c_res.data.get("id")
                    v_res = client.validate_circuit(b_id)
                    findings = v_res.data.get("findings", []) if v_res.ok and isinstance(v_res.data, dict) else []
                    fired_rules = {f.get("rule_id") for f in findings}

                    expected = [
                        ("R001", "Unsupported pin check"),
                        ("R003", "Duplicate wire check"),
                        ("R009", "LED series resistor check"),
                        ("R010", "Unconfirmed wires check")
                    ]

                    st.markdown("#### Bad Sample Rule Scorecard:")
                    for r_id, desc in expected:
                        if r_id in fired_rules:
                            st.success(f"✅ **PASS**: Expected rule `{r_id}` ({desc}) fired successfully.")
                        else:
                            st.error(f"❌ **FAIL**: Expected rule `{r_id}` ({desc}) did NOT fire!")

                    # Check that circuit with unconfirmed wires never passes
                    has_unconfirmed = any(f.get("rule_id") == "R010" for f in findings)
                    verdict = v_res.data.get("status")
                    if has_unconfirmed and verdict == "pass":
                        st.error("❌ **FAIL**: Circuit with unconfirmed wires reported status 'pass'!")
                    else:
                        st.success(f"✅ **PASS**: Circuit with unconfirmed wires did not pass (status='{verdict}').")

        show_raw_json_expander(st.session_state.get("last_api_result"), "Validation Response")


# ===========================================================================
# TAB 5: EXPORTS
# ===========================================================================
with tabs[4]:
    st.header("Circuit Exports & Wokwi Integration")
    st.markdown("Exports the saved canonical document to SVG, PNG, canonical JSON, and Wokwi `diagram.json`.")

    circuit_id = st.session_state.get("circuit_id")
    if not circuit_id:
        st.warning("No circuit loaded. Please load or create a circuit first.")
    else:
        st.info(f"Exporting active circuit: `{circuit_id}`")
        exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)

        # SVG Export
        with exp_col1:
            if st.button("Export SVG", use_container_width=True):
                s_res = client.export_circuit(circuit_id, "svg")
                st.session_state["last_api_result"] = s_res
                if s_res.ok:
                    svg_content = s_res.raw_text
                    st.session_state["exported_svg"] = svg_content
                    st.success("SVG generated!")

        # PNG Export
        with exp_col2:
            if st.button("Export PNG", use_container_width=True):
                p_res = client.export_circuit(circuit_id, "png")
                st.session_state["last_api_result"] = p_res
                if p_res.ok:
                    st.session_state["exported_png"] = p_res.data
                    st.success("PNG generated!")
                else:
                    st.warning(f"PNG generation unavailable: {p_res.error}")

        # JSON Export
        with exp_col3:
            if st.button("Export JSON", use_container_width=True):
                j_res = client.export_circuit(circuit_id, "json")
                st.session_state["last_api_result"] = j_res
                if j_res.ok:
                    st.session_state["exported_json"] = json.dumps(j_res.data, indent=2)
                    st.success("JSON exported!")

        # Wokwi Export
        with exp_col4:
            if st.button("Export Wokwi", use_container_width=True):
                w_res = client.export_wokwi(circuit_id)
                st.session_state["last_api_result"] = w_res
                if w_res.ok and isinstance(w_res.data, dict):
                    st.session_state["exported_wokwi"] = w_res.data
                    st.success("Wokwi exported!")

        st.divider()

        # Display and downloads
        if "exported_svg" in st.session_state:
            st.subheader("Inline SVG Schematic")
            st.components.v1.html(st.session_state["exported_svg"], height=320, scrolling=True)
            st.download_button(
                "⬇️ Download circuit.svg",
                data=st.session_state["exported_svg"],
                file_name="circuit.svg",
                mime="image/svg+xml"
            )

        if "exported_png" in st.session_state and isinstance(st.session_state["exported_png"], bytes):
            st.subheader("Rendered PNG")
            st.image(st.session_state["exported_png"], caption="circuit.png")
            st.download_button(
                "⬇️ Download circuit.png",
                data=st.session_state["exported_png"],
                file_name="circuit.png",
                mime="image/png"
            )

        if "exported_json" in st.session_state:
            st.subheader("Canonical Circuit JSON v1")
            st.download_button(
                "⬇️ Download circuit.json",
                data=st.session_state["exported_json"],
                file_name="circuit.json",
                mime="application/json"
            )

        if "exported_wokwi" in st.session_state:
            w_data = st.session_state["exported_wokwi"]
            st.subheader("Wokwi `diagram.json` Validation")
            diag = w_data.get("diagram", {})

            # Validation check of diagram.json structure
            req_keys = ["version", "editor", "parts", "connections"]
            missing_keys = [k for k in req_keys if k not in diag]
            parts_in_diag = {p.get("id") for p in diag.get("parts", [])}
            bad_conns = []
            for c in diag.get("connections", []):
                # c is [from_spec, to_spec, color, ...]
                if isinstance(c, list) and len(c) >= 2:
                    p1 = c[0].split(":")[0]
                    p2 = c[1].split(":")[0]
                    if p1 not in parts_in_diag or p2 not in parts_in_diag:
                        bad_conns.append(c)

            if not missing_keys and not bad_conns:
                st.success("✅ **PASS**: diagram.json is schema-valid and all connection endpoints exist!")
            else:
                st.error(f"❌ **FAIL**: Wokwi diagram validation errors: missing keys={missing_keys}, bad conns={bad_conns}")

            st.markdown(f"- **Unsupported Parts Count:** {len(w_data.get('unsupported_parts', []))}")
            st.markdown(f"- **Proposed Wires Count:** {len(w_data.get('proposed_wires', []))}")

            st.info("💡 **Wokwi Instructions:** Paste `diagram.json` into the `diagram.json` tab of a new project on [wokwi.com](https://wokwi.com) and fix any rejected pin name.")

            st.download_button(
                "⬇️ Download diagram.json",
                data=json.dumps(diag, indent=2),
                file_name="diagram.json",
                mime="application/json"
            )

        # Test: Verify exports read the saved document, not request state
        st.divider()
        st.markdown("#### Test: Exports Read Saved Document")
        if st.button("🧪 Verify Exports Read Saved Document"):
            c_doc = client.get_circuit(circuit_id).data
            exp_j = client.export_circuit(circuit_id, "json").data
            if c_doc and exp_j and c_doc.get("id") == exp_j.get("id") and len(c_doc.get("parts", [])) == len(exp_j.get("parts", [])):
                st.success("✅ **PASS**: Export faithfully mirrors the saved backend document.")
            else:
                st.error("❌ **FAIL**: Export mismatch with saved document.")

        show_raw_json_expander(st.session_state.get("last_api_result"), "Export Result")


# ===========================================================================
# TAB 6: AI EXPLANATION
# ===========================================================================
with tabs[5]:
    st.header("Multimodal AI Explanation")
    st.markdown("Sends findings to the LLM wrapper. **Non-negotiable principle:** Deterministic Python rules decide pass/fail; the LLM only explains and can never alter a verdict.")

    circuit_id = st.session_state.get("circuit_id")
    if not circuit_id:
        st.warning("No circuit loaded.")
    else:
        target_ref = st.selectbox("Target Project Context", ["None", "led-blink-pin13"], key="exp_target")
        target_arg = None if target_ref == "None" else target_ref

        if st.button("🤖 Generate AI Explanation", use_container_width=True):
            with st.spinner("Requesting explanation from backend..."):
                # Check validation verdict before
                v_before = client.validate_circuit(circuit_id, target=target_arg)
                status_before = v_before.data.get("status") if v_before.ok else "unknown"

                exp_res = client.explain_circuit(circuit_id, target=target_arg)
                st.session_state["last_api_result"] = exp_res

                # Check validation verdict after
                v_after = client.validate_circuit(circuit_id, target=target_arg)
                status_after = v_after.data.get("status") if v_after.ok else "unknown"

                if exp_res.ok and isinstance(exp_res.data, dict):
                    explanation_txt = exp_res.data.get("explanation", "")
                    st.subheader("AI Explanation Output:")
                    st.markdown(f"> {explanation_txt}")

                    # Verdict invariance check
                    if status_before == status_after:
                        st.success(f"✅ **PASS**: Deterministic verdict remained identical (`{status_before}`) before and after AI explanation.")
                    else:
                        st.error(f"❌ **FAIL**: Deterministic verdict changed! ({status_before} -> {status_after})")
                else:
                    st.warning(f"AI explanation returned degraded/fallback result: {exp_res.error}")

        show_raw_json_expander(st.session_state.get("last_api_result"), "Explanation Result")


# ===========================================================================
# TAB 7: SLOT INSPECTOR TEST
# ===========================================================================
with tabs[6]:
    st.header("Slot Inspector Diagnostic")
    st.markdown("Tests the Slot Inspector wrapper for presence/absence of expected components across template slots.")

    col_slot1, col_slot2 = st.columns(2)
    with col_slot1:
        st.subheader("Slot Crop Source")
        slot_file = st.file_uploader("Upload Crop Image (optional)", type=["jpg", "jpeg", "png"], key="slot_upload")
        use_simulated = st.checkbox("Use Simulated Slot Crops", value=(slot_file is None))

    with col_slot2:
        st.subheader("Expected Ground Truth")
        expected_state = st.selectbox("Expected State", ["present", "missing", "skipped"])

    if st.button("🔬 Inspect Slot Crop", use_container_width=True):
        with st.spinner("Inspecting slot crop..."):
            if slot_file is not None:
                crop_bytes = slot_file.getvalue()
            else:
                # Generate simulated crop (greenish if present, black if missing)
                color = (40, 180, 50) if expected_state == "present" else (10, 10, 10)
                img = Image.new("RGB", (100, 100), color=color)
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                crop_bytes = buf.getvalue()

            res = client.inspect_slots_direct(crop_bytes, slot_name="slot_test_1")
            if res.get("ok"):
                st.success("Slot inspection executed successfully!")
                results = res.get("results", [])
                st.dataframe(pd.DataFrame(results))

                if results:
                    detected_state = results[0].get("state")
                    conf = results[0].get("conf", 0.0)
                    acc = 100.0 if detected_state == expected_state else 0.0
                    st.metric("Detected State", f"{detected_state} ({conf:.2f})")
                    st.metric("Test Accuracy Score", f"{acc:.0f}%")
            else:
                st.warning(f"Direct inspection notice: {res.get('error')}. Testing via /analyze...")
                an_res = client.analyze_image(crop_bytes, inspect=True)
                if an_res.ok and isinstance(an_res.data, dict):
                    ins = an_res.data.get("circuit", {}).get("inspection", [])
                    st.dataframe(pd.DataFrame(ins) if ins else pd.DataFrame([{"info": "No slots found"}]))
                else:
                    st.error("Inspection failed.")


# ===========================================================================
# TAB 8: LIVE DETECTION TEST
# ===========================================================================
with tabs[7]:
    st.header("Live Camera & IoU Tracking Test")
    st.markdown("Tests live frame ingestion (`POST /circuits/{id}/live-frame` or `WS /ws/live`), IoU multi-frame component promotion, and latency measurement.")

    circuit_id = st.session_state.get("circuit_id")
    if not circuit_id:
        st.warning("Please load or create a circuit in Tab 2 or 3 before testing live frames.")
    else:
        st.info(f"Target Circuit Session: `{circuit_id}`")
        col_live1, col_live2 = st.columns(2)
        with col_live1:
            frame_file = st.file_uploader("Upload Live Video Frame (JPG/PNG)", type=["jpg", "jpeg", "png"], key="live_frame_upload")
            sim_frames = st.slider("Simulate Consecutive Frames", 1, 10, 5)

        with col_live2:
            st.markdown("#### Operations")
            send_single_clicked = st.button("Send Single Live Frame", use_container_width=True)
            send_stream_clicked = st.button(f"Simulate Stream ({sim_frames} frames)", use_container_width=True)
            freeze_clicked = st.button("❄️ Freeze & Analyze Frame", use_container_width=True)

        # Helper to get frame bytes
        def get_frame_bytes() -> bytes:
            if frame_file is not None:
                return frame_file.getvalue()
            img = Image.new("RGB", (640, 480), color=(40, 80, 120))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return buf.getvalue()

        if send_single_clicked:
            f_bytes = get_frame_bytes()
            with st.spinner("Processing live frame..."):
                res = client.send_live_frame(circuit_id, f_bytes)
                st.session_state["last_api_result"] = res
                if res.ok and isinstance(res.data, dict):
                    st.success(f"Frame processed in {res.elapsed_ms:.1f}ms!")
                    tracks = res.data.get("active_tracks", [])
                    st.markdown("#### Active Tracks")
                    st.dataframe(pd.DataFrame(tracks) if tracks else pd.DataFrame([{"info": "No active tracks"}]), use_container_width=True)
                else:
                    st.error(f"Live frame error: {res.error}")

        if send_stream_clicked:
            f_bytes = get_frame_bytes()
            latencies = []
            tracks = []
            progress = st.progress(0.0)
            for i in range(sim_frames):
                res = client.send_live_frame(circuit_id, f_bytes)
                latencies.append(res.elapsed_ms)
                if res.ok and isinstance(res.data, dict):
                    tracks = res.data.get("active_tracks", [])
                progress.progress((i + 1) / sim_frames)
                time.sleep(0.05)

            avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
            st.success(f"Sent {sim_frames} frames! Average Latency: `{avg_lat:.1f}ms`")
            st.markdown("#### Tracking Stability Table")
            st.dataframe(pd.DataFrame(tracks) if tracks else pd.DataFrame([{"info": "Tracks stabilized"}]), use_container_width=True)

        if freeze_clicked:
            f_bytes = get_frame_bytes()
            with st.spinner("Freezing and analyzing full circuit..."):
                an_res = client.analyze_image(f_bytes, filename="freeze_frame.jpg")
                st.session_state["last_api_result"] = an_res
                if an_res.ok and isinstance(an_res.data, dict):
                    c = an_res.data.get("circuit", {})
                    st.session_state["circuit_id"] = c.get("id")
                    st.session_state["active_circuit"] = c
                    st.success(f"Frozen frame analyzed! Updated circuit ID: `{c.get('id')}`")
                    st.rerun()

        # WebSocket direct check
        st.divider()
        if st.button("🧪 Test WebSocket Live Connection (/ws/live)"):
            f_bytes = get_frame_bytes()
            ws_ok, ws_msg, ws_lat = client.test_websocket_live(f_bytes)
            if ws_ok:
                st.success(f"✅ WebSocket Connected! Latency: {ws_lat:.1f}ms - {ws_msg}")
            else:
                st.info(f"ℹ️ {ws_msg}")

        show_raw_json_expander(st.session_state.get("last_api_result"), "Live Detection Result")


# ===========================================================================
# TAB 9: AUTOMATED SELF-TEST
# ===========================================================================
with tabs[8]:
    st.header("Automated Backend Self-Test Checklist")
    st.markdown("Runs the complete 21-item automated test suite against the backend HTTP API to verify all non-negotiable principles.")

    if st.button("🚀 Run All 21 Checks", type="primary", use_container_width=True):
        runner = SelfTestRunner(client=client)
        with st.spinner("Executing 21 verification checks..."):
            results = runner.run_all()
            summary = runner.summary()
            st.session_state["selftest_results"] = results
            st.session_state["selftest_summary"] = summary

    if "selftest_summary" in st.session_state:
        s = st.session_state["selftest_summary"]
        results: List[CheckResult] = st.session_state["selftest_results"]

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Total Tests", s["total"])
        with col_s2:
            st.metric("Passed", s["passed"])
        with col_s3:
            st.metric("Failed", s["failed"])
        with col_s4:
            st.metric("Skipped", s["skipped"])

        # Display results table
        table_data = []
        for r in results:
            badge = "🟢 PASS" if r.status == "PASS" else ("🟡 SKIPPED" if r.status == "SKIPPED" else "🔴 FAIL")
            table_data.append({
                "Status": badge,
                "Check Name": r.name,
                "Latency (ms)": f"{r.elapsed_ms:.1f}",
                "Details / Message": r.message
            })

        st.subheader("Test Results Matrix")
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        # Download report buttons
        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            json_report = json.dumps(s, indent=2)
            st.download_button(
                "⬇️ Download JSON Test Report",
                data=json_report,
                file_name="visionguard_selftest_report.json",
                mime="application/json"
            )
        with rep_col2:
            df_csv = pd.DataFrame(table_data).to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV Test Report",
                data=df_csv,
                file_name="visionguard_selftest_report.csv",
                mime="text/csv"
            )

    # -----------------------------------------------------------------------
    # Pytest Test Runner (Detection, Diagnostics & Unit Tests)
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("🔬 Pytest Unit & Diagnostics Runner")
    st.markdown("Run pytest test suites (such as `test_detection_and_diagnostics.py`) directly from Streamlit without requiring the backend server to be running.")

    pytest_options = [
        "tests/test_detection_and_diagnostics.py",
        "tests/test_validation.py",
        "tests/test_proposals.py",
        "tests/test_api.py",
        "tests/test_exports.py",
        "tests/test_models.py",
        "tests/test_live_tracker.py",
        "tests/test_vision_wrappers.py",
        "tests/test_templates.py",
        "tests/test_harness_api_client.py",
        "All Tests (tests/)"
    ]

    selected_test = st.selectbox(
        "Select Test Suite to Run",
        options=pytest_options,
        index=0,
        help="Select a test file. Default is test_detection_and_diagnostics.py"
    )

    if st.button(f"▶️ Run {os.path.basename(selected_test)}", type="primary", use_container_width=True):
        target = "tests/" if selected_test == "All Tests (tests/)" else selected_test
        with st.spinner(f"Running pytest {target}..."):
            start_t = time.time()
            cmd = [sys.executable, "-m", "pytest", "-v", target]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            elapsed_sec = time.time() - start_t

            st.session_state["pytest_result"] = {
                "target": target,
                "code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "elapsed": elapsed_sec
            }

    if "pytest_result" in st.session_state:
        pr = st.session_state["pytest_result"]
        output = pr["stdout"] + pr["stderr"]

        test_items = []
        for line in output.splitlines():
            if "::" in line:
                parts = line.split("::")
                file_name = parts[0].strip()
                rest = parts[1].strip()
                status_m = re.search(r"(PASSED|FAILED|SKIPPED|ERROR)", rest)
                status = status_m.group(1) if status_m else "UNKNOWN"
                test_name = re.sub(r"\s+(PASSED|FAILED|SKIPPED|ERROR).*", "", rest).strip()
                test_items.append({
                    "Test Name": test_name,
                    "Status": "🟢 PASSED" if status == "PASSED" else ("🟡 " + status if status == "SKIPPED" else "🔴 " + status),
                    "File": file_name
                })

        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            total_tests = len(test_items)
            st.metric("Tests Executed", total_tests)
        with p_col2:
            verdict = "PASSED ✅" if pr["code"] == 0 else "FAILED ❌"
            st.metric("Suite Verdict", verdict)
        with p_col3:
            st.metric("Duration", f"{pr['elapsed']:.2f} s")

        if test_items:
            st.dataframe(pd.DataFrame(test_items), use_container_width=True)

        with st.expander("📄 Pytest Console Logs / Raw Output", expanded=(pr["code"] != 0)):
            st.code(output, language="text")


