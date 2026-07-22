from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class NoteEvent:
    """Dataclass representing a parsed musical note event through all pipeline stages."""

    start: float
    end: float
    pitch: int
    pitches: List[int] = field(default_factory=list)  # Polyphonic pitches (chords, double stops)
    amplitude: float = 1.0  # Normalized RMS audio energy
    bends: Optional[List[float]] = None  # Semitone pitch bends over time
    microtone_cents: float = 0.0  # Fretless microtonal pitch offset in cents
    tag: str = "normal"  # "normal", "staccato", "ghost", "slap", "pop", "palm_mute", "let_ring"
    duty_cycle: float = 1.0
    string_idx: Optional[int] = None
    fret_val: Optional[int] = None
    finger_val: Optional[int] = None
    positions: List[Tuple[int, int, int]] = field(default_factory=list)  # [(string, fret, finger), ...]
    
    # Expressive & Rhythmic Markers
    is_triplet: bool = False
    is_accent: bool = False
    dynamic_mark: Optional[str] = None  # "p", "mp", "mf", "f"
    is_legato: bool = False
    is_slide: bool = False
    is_rake: bool = False
    is_pickup: bool = False
    spanner_tag: Optional[str] = None  # "let_ring", "palm_mute"
    spanner_type: Optional[str] = None  # "start", "stop", "continue"

    def __post_init__(self):
        if not self.pitches and self.pitch is not None:
            self.pitches = [self.pitch]
        elif self.pitches and self.pitch is None:
            self.pitch = self.pitches[0]

    @property
    def duration(self) -> float:
        return max(0.001, self.end - self.start)

    @property
    def is_polyphonic(self) -> bool:
        return len(self.pitches) > 1
