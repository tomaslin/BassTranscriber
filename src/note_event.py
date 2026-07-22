from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class NoteEvent:
    start: float
    end: float
    pitch: int
    pitches: List[int] = field(default_factory=list)
    amplitude: float = 0.5
    bends: List[float] = field(default_factory=list)
    microtone_cents: float = 0.0
    tag: str = "normal"  # normal, staccato, slap, pop, palm_mute, ghost, harmonic, hammer_on, pull_off, slide
    duty_cycle: float = 1.0
    is_triplet: bool = False
    is_accent: bool = False
    dynamic_mark: str = "mf"  # p, mp, mf, f
    is_pickup: bool = False
    is_harmonic: bool = False
    slide_from: Optional[int] = None  # Source MIDI pitch if note arrived via a slide/hammer-on

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def update_pitch(self, new_pitch: int):
        self.pitch = new_pitch
        self.pitches = [new_pitch]

    def to_dict(self) -> dict:
        """Helper to serialize NoteEvent for tab engines, UI rendering, or JSON export."""
        d = asdict(self)
        d['duration'] = self.duration
        return d
