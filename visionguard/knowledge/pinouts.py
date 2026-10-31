import os
import json
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("visionguard.pinouts")

PINOUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "pinouts.json")

class PinoutKnowledgeBase:
    """
    Local pinout and specifications knowledge base.
    Guarantees that pinouts are strictly loaded from verified local specs,
    never invented by the LLM or guessed.
    """
    def __init__(self, filepath: str = PINOUT_FILE):
        self.filepath = filepath
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load pinout file {self.filepath}: {e}")
                self._data = {}
        else:
            logger.warning(f"Pinout file {self.filepath} not found.")
            self._data = {}

    def lookup(self, query: Optional[str]) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Looks up verified pinout by exact, alias, or fuzzy normalized match.
        Returns: (pinout_dict_or_None, status_message)
        """
        if not query or not query.strip():
            return None, "no verified pinout available"

        norm_query = query.strip().lower().replace("_", "-").replace(" ", "-")

        # 1. Exact key match
        if norm_query in self._data:
            entry = self._data[norm_query]
            return entry, f"Found verified pinout for '{entry.get('name', norm_query)}'"

        # 2. Check aliases
        for key, entry in self._data.items():
            aliases = [a.lower().replace("_", "-").replace(" ", "-") for a in entry.get("aliases", [])]
            if norm_query in aliases or key == norm_query:
                return entry, f"Found verified pinout for '{entry.get('name', key)}'"

        # 3. Substring match (e.g. part number 'atmega328p-pu' matching 'atmega328p')
        for key, entry in self._data.items():
            if key in norm_query or norm_query in key:
                return entry, f"Found verified pinout for '{entry.get('name', key)}'"
            for alias in entry.get("aliases", []):
                norm_alias = alias.lower().replace("_", "-").replace(" ", "-")
                if norm_alias in norm_query or norm_query in norm_alias:
                    return entry, f"Found verified pinout for '{entry.get('name', key)}'"

        return None, "no verified pinout available"

default_pinout_kb = PinoutKnowledgeBase()

