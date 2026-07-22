import re
import numpy as np
import librosa
from music21 import key, pitch

MIN_BASS_MIDI = 23  # B0 (5/6-string)
MAX_BASS_MIDI = 84  # C6 (Solos / high harmonics / double stops)


def fold_pitch_to_bass_range(midi_pitch: int, min_pitch: int = 23, max_pitch: int = 84) -> int:
    """Folds pitch into valid bass register while allowing extended solo / upper fretboard ranges."""
    if min_pitch >= max_pitch:
        return min_pitch
    while midi_pitch < min_pitch:
        midi_pitch += 12
    while midi_pitch > max_pitch:
        midi_pitch -= 12
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
    """
    Detects musical key signature and mode using chroma profiles across Major,
    Minor, Mixolydian, Dorian, and Blues profiles. Correctly instantiates modal key objects.
    """
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
    mixolydian_profile = np.array([6.20, 2.10, 3.40, 2.20, 4.30, 4.00, 2.40, 5.10, 2.30, 3.50, 4.80, 2.50])
    dorian_profile = np.array([6.20, 2.50, 3.50, 5.20, 2.50, 3.80, 2.40, 4.80, 2.80, 4.00, 3.20, 2.80])
    blues_profile = np.array([6.50, 1.80, 2.20, 5.50, 2.00, 5.00, 4.80, 5.20, 1.80, 2.20, 4.50, 2.00])

    pitch_names = ['C', 'C#', 'D', 'E-', 'E', 'F', 'F#', 'G', 'A-', 'A', 'B-', 'B']
    profiles = [
        (major_profile, "major"),
        (minor_profile, "minor"),
        (mixolydian_profile, "mixolydian"),
        (dorian_profile, "dorian"),
        (blues_profile, "minor"),
    ]

    best_score = -float('inf')
    best_root = 'C'
    best_mode = 'major'

    for i in range(12):
        rot_chroma = np.roll(chroma_sum, -i)
        for prof, mode_type in profiles:
            corr = np.nan_to_num(np.corrcoef(rot_chroma, prof)[0, 1])
            if corr > best_score:
                best_score = corr
                best_root = pitch_names[i]
                best_mode = mode_type

    try:
        # Proper music21 modal initialization
        if best_mode in ["major", "minor"]:
            k_str = best_root if best_mode == "major" else best_root.lower()
            return key.Key(k_str), False
        else:
            return key.Key(best_root, best_mode), False
    except Exception:
        return key.Key('C'), False


def snap_pitch_to_scale(midi_val: int, key_obj, level: int = 5, next_midi: int = None) -> int:
    """Folds pitch into bass range and snaps to detected scale if requested by level."""
    midi_val = fold_pitch_to_bass_range(midi_val)

    if key_obj is None or level >= 2:
        return midi_val

    scale_pcs = [p.pitchClass for p in key_obj.getPitches()]
    curr_pc = midi_val % 12

    if curr_pc in scale_pcs:
        return midi_val

    if next_midi is not None and abs(next_midi - midi_val) == 1:
        return midi_val

    distances = sorted([((sp - curr_pc + 6) % 12 - 6, sp) for sp in scale_pcs], key=lambda x: abs(x[0]))

    if level <= 1 or abs(distances[0][0]) <= 1:
        return midi_val + distances[0][0]

    return midi_val


def get_directional_enharmonic_pitch(midi_val: int, key_obj=None, prev_midi: int = None) -> pitch.Pitch:
    """Returns a key-aware and line-direction-aware music21 Pitch object."""
    p = pitch.Pitch(midi=midi_val)

    if key_obj is not None:
        try:
            curr_pc = midi_val % 12
            key_pitches = key_obj.getPitches()
            matching_p = next((kp for kp in key_pitches if kp.pitchClass == curr_pc), None)
            if matching_p is not None:
                octave_val = (midi_val // 12) - 1
                return pitch.Pitch(f"{matching_p.name}{octave_val}")
        except Exception:
            pass

    if p.accidental is not None and prev_midi is not None and prev_midi != midi_val:
        is_ascending = midi_val > prev_midi
        if is_ascending and p.accidental.name == 'flat':
            p.getEnharmonic(inPlace=True)
        elif not is_ascending and p.accidental.name == 'sharp':
            p.getEnharmonic(inPlace=True)

    return p


def get_key_aware_pitch(midi_val: int, key_obj=None, prev_midi: int = None) -> pitch.Pitch:
    return get_directional_enharmonic_pitch(midi_val, key_obj, prev_midi)
