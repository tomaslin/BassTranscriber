import re
import numpy as np
import librosa
from music21 import key, pitch

MIN_BASS_MIDI = 28  # E1
MAX_BASS_MIDI = 67  # G4


def fold_pitch_to_bass_range(midi_pitch: int) -> int:
    """Folds any MIDI pitch into 4-string bass octave range (E1 to G4)."""
    while midi_pitch > MAX_BASS_MIDI:
        midi_pitch -= 12
    while midi_pitch < MIN_BASS_MIDI:
        midi_pitch += 12
    return midi_pitch


def normalize_key_str(raw_key: str):
    """Normalizes key strings into music21 pitch notation."""
    if not raw_key:
        return None

    k = re.sub(r'(?i)sharp', '#', raw_key.strip().replace('_', ' '))
    k = re.sub(r'(?i)flat', '-', k)
    k = re.sub(r'([A-Ga-g])b(?![a-zA-Z])', r'\1-', k)

    if any(k.lower().endswith(ext) for ext in ["min", "minor"]) or (
        len(k) > 1 and k.endswith("m") and not k.endswith("-m")
    ):
        clean_root = re.sub(r'(?i)[\s\-_]*(min|minor|m)$', '', k).strip()
        return clean_root.lower()

    if any(k.lower().endswith(ext) for ext in ["maj", "major"]):
        clean_root = re.sub(r'(?i)[\s\-_]*(maj|major)$', '', k).strip()
        return clean_root.capitalize()

    return k


def detect_key_signature(audio_y, sr, parsed_key=None, bass_filter_fn=None):
    """Detects musical key signature using chroma profiles or parsed key metadata."""
    if parsed_key:
        normalized = normalize_key_str(parsed_key)
        try:
            return key.Key(normalized), True
        except Exception:
            pass

    filtered_y = bass_filter_fn(audio_y, sr, lowcut=30.0, highcut=600.0) if bass_filter_fn else audio_y

    try:
        chroma = librosa.feature.chroma_cqt(y=filtered_y, sr=sr, fmin=librosa.note_to_hz('C1'), n_octaves=4)
    except Exception:
        chroma = librosa.feature.chroma_cqt(y=filtered_y, sr=sr)

    chroma_sum = np.sum(np.log1p(chroma * 10), axis=1)
    if np.sum(chroma_sum) == 0:
        return key.Key('C'), False

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.98, 2.69, 3.34, 3.17])
    pitch_names = ['C', 'C#', 'D', 'E-', 'E', 'F', 'F#', 'G', 'A-', 'A', 'B-', 'B']

    best_score, best_key_str = -float('inf'), 'C'

    for i in range(12):
        rot_chroma = np.roll(chroma_sum, -i)
        maj_corr = np.nan_to_num(np.corrcoef(rot_chroma, major_profile)[0, 1])
        min_corr = np.nan_to_num(np.corrcoef(rot_chroma, minor_profile)[0, 1])

        if maj_corr > best_score:
            best_score, best_key_str = maj_corr, pitch_names[i]
        if min_corr > best_score:
            best_score, best_key_str = min_corr, pitch_names[i].lower()

    try:
        return key.Key(best_key_str), False
    except Exception:
        return key.Key('C'), False


def snap_pitch_to_scale(midi_val: int, key_obj, level: int = 5) -> int:
    """
    Folds pitch into 4-string bass range.
    Preserves chromatic passing tones for levels >= 2, only quantizing diatonic pitches at simplified levels 0-1.
    """
    midi_val = fold_pitch_to_bass_range(midi_val)

    if key_obj is None or level >= 2:
        return midi_val

    scale_pcs = [p.pitchClass for p in key_obj.getPitches()]
    curr_pc = midi_val % 12

    if curr_pc in scale_pcs:
        return midi_val

    distances = sorted([((sp - curr_pc + 6) % 12 - 6, sp) for sp in scale_pcs], key=lambda x: abs(x[0]))

    if level <= 1 or abs(distances[0][0]) <= 1:
        return midi_val + distances[0][0]

    return midi_val


def get_key_aware_pitch(midi_val: int, key_obj):
    """Returns key-aware music21 Pitch object for clean enharmonic spelling."""
    if key_obj is None:
        return pitch.Pitch(midi=midi_val)
    try:
        return key_obj.getPitchFromMidi(midi_val)
    except Exception:
        return pitch.Pitch(midi=midi_val)
