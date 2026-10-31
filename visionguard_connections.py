"""
VisionGuard Connection Proposals Module
Re-exports connection proposal algorithms:
- propose_from_template(circuit)
- propose_from_proximity(circuit)
- propose_from_vision_llm(image_bytes)
"""

from visionguard.proposals.template import propose_from_template
from visionguard.proposals.proximity import propose_from_proximity
from visionguard.proposals.llm import propose_from_vision_llm, ProposalResult
from visionguard.proposals.manager import apply_proposals, apply_user_updates

__all__ = [
    "propose_from_template",
    "propose_from_proximity",
    "propose_from_vision_llm",
    "ProposalResult",
    "apply_proposals",
    "apply_user_updates",
]

