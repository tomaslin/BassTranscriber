from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NoteEvent:
    """Dataclass representing a parsed musical note event through all pipeline stages."""

    start: float
    end: float
    pitch: int
    amplitude: float = 1.0
    bends: Optional[List[float]] = None
    tag: str = "normal"  # "normal", "staccato", "ghost", "slap", "pop"
    duty_cycle: float = 1.0
    string_idx: Optional[int] = None
    fret_val: Optional[int] = None

    @property
    def duration(self) -> float:
        return self.end - self.start
