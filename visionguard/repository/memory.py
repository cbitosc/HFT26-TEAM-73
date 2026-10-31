import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from visionguard.models.circuit import Circuit

class CircuitRepository(ABC):
    @abstractmethod
    def get(self, circuit_id: str) -> Optional[Circuit]:
        """Retrieve a circuit by id, or None if not found."""
        pass

    @abstractmethod
    def save(self, circuit: Circuit) -> Circuit:
        """Save or update a circuit document. Returns the saved document."""
        pass

    @abstractmethod
    def list_all(self) -> List[Circuit]:
        """List all stored circuits."""
        pass

    @abstractmethod
    def delete(self, circuit_id: str) -> bool:
        """Delete a circuit by id. Returns True if deleted, False if not found."""
        pass

    @abstractmethod
    def save_image(self, image_id: str, image_bytes: bytes) -> None:
        """Store raw image bytes associated with an image_id."""
        pass

    @abstractmethod
    def get_image(self, image_id: str) -> Optional[bytes]:
        """Retrieve raw image bytes by image_id, or None if not found."""
        pass

class InMemoryCircuitRepository(CircuitRepository):
    """
    Thread-safe in-memory circuit repository.
    Can be swapped for SQLite / Postgres later.
    """
    def __init__(self):
        self._store: Dict[str, Circuit] = {}
        self._image_store: Dict[str, bytes] = {}
        self._lock = threading.Lock()

    def get(self, circuit_id: str) -> Optional[Circuit]:
        with self._lock:
            circuit = self._store.get(circuit_id)
            if circuit:
                # Return deep copy via model_copy so caller mutations do not alter store directly
                return circuit.model_copy(deep=True)
            return None

    def save(self, circuit: Circuit) -> Circuit:
        with self._lock:
            # Store a deep copy
            saved = circuit.model_copy(deep=True)
            self._store[circuit.id] = saved
            return saved.model_copy(deep=True)

    def list_all(self) -> List[Circuit]:
        with self._lock:
            return [c.model_copy(deep=True) for c in self._store.values()]

    def delete(self, circuit_id: str) -> bool:
        with self._lock:
            if circuit_id in self._store:
                del self._store[circuit_id]
                return True
            return False

    def save_image(self, image_id: str, image_bytes: bytes) -> None:
        with self._lock:
            self._image_store[image_id] = image_bytes

    def get_image(self, image_id: str) -> Optional[bytes]:
        with self._lock:
            return self._image_store.get(image_id)

# Global default repository instance
default_repository = InMemoryCircuitRepository()

