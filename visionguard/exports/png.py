import logging
from typing import Optional, Tuple
from visionguard.models.circuit import Circuit
from visionguard.exports.svg import generate_circuit_svg

logger = logging.getLogger("visionguard.exports.png")

class CairosvgNotAvailableError(Exception):
    """Raised when cairosvg or libcairo runtime is not available."""
    pass

def render_circuit_png(circuit: Circuit) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Renders circuit SVG to PNG using cairosvg.
    Degrades gracefully: if cairosvg or libcairo is missing, returns (None, error_message).
    Never crashes with an unhandled 500.
    """
    try:
        import cairosvg
    except ImportError:
        msg = "cairosvg is not installed. Install with 'pip install cairosvg' and ensure libcairo is available."
        logger.warning(msg)
        return None, msg

    try:
        svg_content = generate_circuit_svg(circuit)
        png_bytes = cairosvg.svg2png(bytestring=svg_content.encode("utf-8"))
        return png_bytes, None
    except Exception as exc:
        msg = f"Cairo rasterization failed: {str(exc)}"
        logger.error(msg)
        return None, msg

