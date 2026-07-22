from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

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
    slide_from: Optional[int] = None
    
    # Genre-driven finger position assignment (String, Fret, Finger)
    fret_position: Optional[Tuple[int, int, int]] = None
    is_downpick: bool = False

    # Category and Anchor Pattern Encoding Attributes
    category: str = "melodic"  # groove_anchor, percussive, expressive, melodic
    anchor_pattern: Optional[str] = None  # e.g., downbeat_anchor, Box-Fret-5
    anchor_fret: Optional[int] = None
    is_anchor: bool = False

    def __post_init__(self):
        if not self.pitches:
            self.pitches = [self.pitch]
        if self.category == "melodic":
            self.determine_category()

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def update_pitch(self, new_pitch: int):
        self.pitch = new_pitch
        self.pitches = [new_pitch]

    def determine_category(self) -> str:
        """Categorizes note events based on performance tags, dynamics, and structural anchor attributes."""
        if self.tag in ["ghost", "slap", "pop", "palm_mute", "staccato"]:
            self.category = "percussive"
        elif self.tag in ["hammer_on", "pull_off", "slide"] or self.is_harmonic or len(self.bends) > 0 or abs(self.microtone_cents) > 10.0:
            self.category = "expressive"
        elif self.is_anchor or self.is_pickup or self.is_accent:
            self.category = "groove_anchor"
        else:
            self.category = "melodic"
        return self.category

    def to_dict(self) -> dict:
        d = asdict(self)
        d['duration'] = self.duration
        d['category'] = self.determine_category()
        return d
