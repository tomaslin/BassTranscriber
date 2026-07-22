from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NoteEvent:
    """Dataclass representing a parsed musical note event through all pipeline stages."""

    start: float
    end: float
    pitch: int
    amplitude: float = 1.0  # Normalized RMS audio energy
    bends: Optional[List[float]] = None  # Semitone pitch bends over time
    tag: str = "normal"  # "normal", "staccato", "ghost", "slap", "pop", "palm_mute"
    duty_cycle: float = 1.0
    string_idx: Optional[int] = None
    fret_val: Optional[int] = None
    
    # Expressive & Rhythmic Markers
    is_triplet: bool = False
    is_accent: bool = False
    dynamic_mark: Optional[str] = None  # "p", "mp", "mf", "f"
    is_legato: bool = False
    is_slide: bool = False
    is_rake: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start
