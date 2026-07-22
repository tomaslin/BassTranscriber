from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

@dataclass
class Genre:
    name: str = "default"
    tuning: str = "4_string_standard"
    technique: str = "fingerstyle"
    rhythmic_grid: str = "16th_syncopated"
    rhythmic_anchor: Dict[str, Any] = field(default_factory=lambda: {
        "pattern": "downbeat_one",
        "lock_strength": "moderate"
    })
    features: Dict[str, bool] = field(default_factory=lambda: {
        "ghost_notes": True,
        "compound_meter": False,
        "downpicking_preference": False,
        "synth_emulation": False
    })
    costs: Dict[str, float] = field(default_factory=lambda: {
        "pop_non_treble_penalty": 25.0,
        "slap_non_bass_penalty": 18.0,
        "fret_stretch_penalty": 10.0,
        "position_shift_multiplier": 2.0,
        "open_string_bonus": -1.5
    })

    def __post_init__(self):
        # Ensure sub-dictionaries have default structures if some keys are missing
        if not isinstance(self.rhythmic_anchor, dict):
            self.rhythmic_anchor = {}
        if not isinstance(self.features, dict):
            self.features = {}
        if not isinstance(self.costs, dict):
            self.costs = {}

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "Genre":
        """Creates a Genre instance from a raw dictionary configuration."""
        return cls(
            name=name,
            tuning=data.get("tuning", "4_string_standard"),
            technique=data.get("technique", "fingerstyle"),
            rhythmic_grid=data.get("rhythmic_grid", "16th_syncopated"),
            rhythmic_anchor=data.get("rhythmic_anchor", {
                "pattern": "downbeat_one",
                "lock_strength": "moderate"
            }),
            features=data.get("features", {
                "ghost_notes": True,
                "compound_meter": False,
                "downpicking_preference": False,
                "synth_emulation": False
            }),
            costs=data.get("costs", {
                "pop_non_treble_penalty": 25.0,
                "slap_non_bass_penalty": 18.0,
                "fret_stretch_penalty": 10.0,
                "position_shift_multiplier": 2.0,
                "open_string_bonus": -1.5
            })
        )

    def to_dict(self) -> dict:
        """Returns the dictionary representation matching the config format."""
        return {
            "name": self.name,
            "tuning": self.tuning,
            "technique": self.technique,
            "rhythmic_grid": self.rhythmic_grid,
            "rhythmic_anchor": self.rhythmic_anchor,
            "features": self.features,
            "costs": self.costs
        }

    # Dict compatibility methods for backward compatibility
    def __getitem__(self, key: str) -> Any:
        if key == "name":
            return self.name
        elif key == "tuning":
            return self.tuning
        elif key == "technique":
            return self.technique
        elif key == "rhythmic_grid":
            return self.rhythmic_grid
        elif key == "rhythmic_anchor":
            return self.rhythmic_anchor
        elif key == "features":
            return self.features
        elif key == "costs":
            return self.costs
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return ["name", "tuning", "technique", "rhythmic_grid", "rhythmic_anchor", "features", "costs"]
