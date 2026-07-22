from dataclasses import dataclass, asdict
from typing import Any, Dict

@dataclass
class Level:
    level_id: int
    name: str
    description: str
    ghost_notes_allowed: bool
    min_duration: float
    snaps_to_scale: bool
    downbeat_only: bool

    @classmethod
    def from_id(cls, level_id: int) -> "Level":
        """Factory method to construct a standard Level object based on level_id (0-5)."""
        clamped_id = max(0, min(5, level_id))
        
        # Define attributes for each of the 6 standard performance levels (0-5)
        configs = {
            0: {
                "name": "Minimalist Roots",
                "description": "Downbeats and half-measure anchors only, creating a highly spacious foundational root note layout.",
                "ghost_notes_allowed": False,
                "min_duration": 0.20,
                "snaps_to_scale": True,
                "downbeat_only": True
            },
            1: {
                "name": "Fundamental Anchors",
                "description": "Retains core groove anchors and primary subdivisions to establish the main structural chord progression.",
                "ghost_notes_allowed": False,
                "min_duration": 0.20,
                "snaps_to_scale": True,
                "downbeat_only": True
            },
            2: {
                "name": "Laid-Back Simple",
                "description": "Brings in eighth-note pulses and on-beat subdivisions, filtering out complex rapid fills and ghost notes.",
                "ghost_notes_allowed": False,
                "min_duration": 0.20,
                "snaps_to_scale": False,
                "downbeat_only": False
            },
            3: {
                "name": "Authentic Direct",
                "description": "Original transcription minus soft percussive clicks and ghost notes, resulting in a clean melodic line.",
                "ghost_notes_allowed": False,
                "min_duration": 0.12,
                "snaps_to_scale": False,
                "downbeat_only": False
            },
            4: {
                "name": "Unfiltered Dynamic",
                "description": "Matches the original recording's full notation, including syncopated microtones and selective ghost notes.",
                "ghost_notes_allowed": True,
                "min_duration": 0.0,
                "snaps_to_scale": False,
                "downbeat_only": False
            },
            5: {
                "name": "Complete Original",
                "description": "The exact high-fidelity transcription featuring all expressive micro-dynamics, slides, and ghost notes.",
                "ghost_notes_allowed": True,
                "min_duration": 0.0,
                "snaps_to_scale": False,
                "downbeat_only": False
            }
        }
        
        cfg = configs[clamped_id]
        return cls(
            level_id=clamped_id,
            name=cfg["name"],
            description=cfg["description"],
            ghost_notes_allowed=cfg["ghost_notes_allowed"],
            min_duration=cfg["min_duration"],
            snaps_to_scale=cfg["snaps_to_scale"],
            downbeat_only=cfg["downbeat_only"]
        )

    def to_dict(self) -> dict:
        return asdict(self)

    # Dict compatibility methods for convenience
    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        d = self.to_dict()
        return d.get(key, default)
