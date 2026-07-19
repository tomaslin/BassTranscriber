#!/usr/bin/env bash
# ==============================================================================
# Bass Transcription Pipeline (M1 / Apple Silicon Optimized Edition v3.0)
# Usage: ./advanced.sh <path_to_stems_folder> [genre]
# ==============================================================================

set -euo pipefail

STEMS_DIR="${1:-}"
TARGET_GENRE="${2:-none}"

if [ -z "$STEMS_DIR" ] || [ ! -d "$STEMS_DIR" ]; then
    echo "Error: Invalid stems directory path. Usage: $0 <path_to_stems_folder> [genre]"
    exit 1
fi

TARGET_GENRE=$(echo "${TARGET_GENRE}" | tr '[:upper:]' '[:lower:]')

if [ -d "/opt/homebrew/bin" ]; then export PATH="/opt/homebrew/bin:$PATH"; fi

if command -v python3.11 &> /dev/null; then
    PY_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PY_CMD="python3"
else
    echo "Error: Python 3.11+ required."; exit 1;
fi

OUT_DIR="./output_bass_advanced"
ENV_DIR=".venv_bass_advanced"
mkdir -p "$OUT_DIR"

if [ ! -d "$ENV_DIR" ]; then
    $PY_CMD -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

"$ENV_DIR/bin/python" -m pip install --upgrade pip wheel &> /dev/null
"$ENV_DIR/bin/python" -m pip install "setuptools<82" &> /dev/null

OS_NAME=$(uname -s)
ARCH_NAME=$(uname -m)

if [ "$OS_NAME" = "Darwin" ] && [ "$ARCH_NAME" = "arm64" ]; then
    "$ENV_DIR/bin/python" -m pip install --no-cache-dir "tensorflow-macos<2.16.0" "tensorflow-metal==1.1.0" &> /dev/null
else
    "$ENV_DIR/bin/python" -m pip install --no-cache-dir "tensorflow<2.16.0" &> /dev/null
fi

"$ENV_DIR/bin/python" -m pip install --no-cache-dir \
    "numpy==1.26.4" "scipy==1.12.0" "soundfile==0.12.1" "soxr==0.3.7" \
    "librosa==0.10.1" "music21==9.1.0" "pretty_midi==0.2.10" \
    "basic-pitch>=0.4.0" "resampy==0.4.2" &> /dev/null

cleanup() { rm -f run_engine_bass.py; }
trap cleanup EXIT

cat << 'EOF' > run_engine_bass.py
import sys
import os
import logging
import warnings
import contextlib
import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['TF_NUM_INTRAOP_THREADS'] = '8'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, filtfilt
from basic_pitch.inference import predict as bp_predict
from music21 import instrument, clef, metadata, tempo, stream, note, chord, meter, key, articulations, pitch

# ==============================================================================
# SUBGENRE OPTIMIZATION REGISTRY
# ==============================================================================
SUBGENRE_REGISTRY = {
    "metal": {
        "low_cut": 60, "high_cut": 2500,
        "onset_threshold": 0.70, "frame_threshold": 0.40, "minimum_note_length": 0.05,
        "preferred_fret_range": (0, 15), "string_bias": "bright_clear",
        "legato_gap_tolerance": 0.03, "use_double_bass": False
    },
    "rock": {
        "low_cut": 50, "high_cut": 2000,
        "onset_threshold": 0.60, "frame_threshold": 0.45, "minimum_note_length": 0.06,
        "preferred_fret_range": (0, 12), "string_bias": "standard",
        "legato_gap_tolerance": 0.04, "use_double_bass": False
    },
    "salsa": {
        "low_cut": 40, "high_cut": 1200,
        "onset_threshold": 0.65, "frame_threshold": 0.50, "minimum_note_length": 0.08,
        "preferred_fret_range": (0, 5), "string_bias": "low_fat",
        "legato_gap_tolerance": 0.05, "use_double_bass": True
    },
    "timba": {
        "low_cut": 40, "high_cut": 1300,
        "onset_threshold": 0.68, "frame_threshold": 0.48, "minimum_note_length": 0.07,
        "preferred_fret_range": (0, 7), "string_bias": "low_fat",
        "legato_gap_tolerance": 0.04, "use_double_bass": True
    },
    "tango": {
        "low_cut": 30, "high_cut": 700,
        "onset_threshold": 0.45, "frame_threshold": 0.60, "minimum_note_length": 0.10,
        "preferred_fret_range": (0, 9), "string_bias": "extended_linear",
        "legato_gap_tolerance": 0.08, "use_double_bass": True
    },
    "funk": {
        "low_cut": 30, "high_cut": 4000,
        "onset_threshold": 0.55, "frame_threshold": 0.45, "minimum_note_length": 0.04,
        "preferred_fret_range": (0, 18), "string_bias": "standard",
        "legato_gap_tolerance": 0.02, "use_double_bass": False
    },
    "rnb": {
        "low_cut": 35, "high_cut": 1000,
        "onset_threshold": 0.50, "frame_threshold": 0.50, "minimum_note_length": 0.06,
        "preferred_fret_range": (0, 7), "string_bias": "low_fat",
        "legato_gap_tolerance": 0.06, "use_double_bass": False
    },
    "hiphop": {
        "low_cut": 30, "high_cut": 800,
        "onset_threshold": 0.50, "frame_threshold": 0.55, "minimum_note_length": 0.08,
        "preferred_fret_range": (0, 5), "string_bias": "low_fat",
        "legato_gap_tolerance": 0.07, "use_double_bass": False
    },
    "jazz": {
        "low_cut": 40, "high_cut": 1500,
        "onset_threshold": 0.55, "frame_threshold": 0.50, "minimum_note_length": 0.07,
        "preferred_fret_range": (0, 9), "string_bias": "standard",
        "legato_gap_tolerance": 0.05, "use_double_bass": True
    },
    "swing": {
        "low_cut": 40, "high_cut": 1200,
        "onset_threshold": 0.52, "frame_threshold": 0.52, "minimum_note_length": 0.08,
        "preferred_fret_range": (0, 7), "string_bias": "standard",
        "legato_gap_tolerance": 0.06, "use_double_bass": True
    },
    "none": {
        "low_cut": 40, "high_cut": 800,
        "onset_threshold": 0.50, "frame_threshold": 0.50, "minimum_note_length": 0.07,
        "preferred_fret_range": (0, 12), "string_bias": "standard",
        "legato_gap_tolerance": 0.05, "use_double_bass": False
    }
}

def get_config(genre):
    return SUBGENRE_REGISTRY.get(genre, SUBGENRE_REGISTRY["none"])

def estimate_harmonic_key(y, sr):
    if len(y) == 0 or np.all(y == 0): return "C major"
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=12)
    mean_chroma = np.mean(chroma, axis=1)
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    root_idx = int(np.argmax(mean_chroma))
    mode = "minor" if mean_chroma[(root_idx + 3) % 12] > mean_chroma[(root_idx + 4) % 12] else "major"
    return f"{pitch_classes[root_idx]} {mode}"

def map_pedagogical_fretboard(midi_pitch, genre):
    cfg = get_config(genre)
    pitch_val = max(23, min(midi_pitch, 67))
    
    # Define standard string fundamentals (MIDI numbers)
    if genre in ["metal", "rock"]:
        string_baselines = [43, 38, 33, 28]  # G, D, A, E (4-string)
        string_names = ["G", "D", "A", "E"]
    else:
        string_baselines = [43, 38, 33, 28, 23]  # G, D, A, E, B (5-string/Double Bass)
        string_names = ["G", "D", "A", "E", "B"]

    if cfg["use_double_bass"]:
        for idx, base in enumerate(string_baselines):
            if pitch_val >= base:
                fret_equiv = pitch_val - base
                if fret_equiv == 0: return string_names[idx], "Open"
                elif fret_equiv <= 2: return string_names[idx], "1/2"
                elif fret_equiv <= 4: return string_names[idx], "1st"
                else: return string_names[idx], "High"
        return "G", "High"

    # Fretboard scanning using positional boundaries
    min_fret, max_fret = cfg["preferred_fret_range"]
    bias = cfg["string_bias"]

    candidates = []
    for idx, base in enumerate(string_baselines):
        fret = pitch_val - base
        if 0 <= fret <= 24:
            candidates.append((idx, string_names[idx], fret))

    if not candidates:
        return "E", "0"

    # Strategy-based path selection
    if bias == "low_fat":
        # Sort prioritizing thicker strings (higher baseline indexes/lower pitches)
        candidates.sort(key=lambda x: (-x[0], x[2]))
    elif bias == "bright_clear":
        # Sort prioritizing thin strings (lower baseline indexes)
        candidates.sort(key=lambda x: (x[0], x[2]))
    elif bias == "extended_linear":
        # Minimize shifting across strings, focus on moving linearly up a single string
        candidates.sort(key=lambda x: x[2])
    else:
        # Standard layout: balance fret target range
        candidates.sort(key=lambda x: abs(x[2] - ((min_fret + max_fret) / 2)))

    # Return first option fitting within threshold or fallback
    for target in candidates:
        if min_fret <= target[2] <= max_fret:
            return target[1], str(int(target[2]))
            
    return candidates[0][1], str(int(candidates[0][2]))

def apply_bass_cultural_grid(time_val, beat_times, genre, level):
    if len(beat_times) == 0 or genre == "none": return time_val
        
    closest_beat_idx = np.argmin(np.abs(beat_times - time_val))
    closest_beat_time = beat_times[closest_beat_idx]
    time_diff = time_val - closest_beat_time
    subdivision = (closest_beat_idx % 4) + 1

    if level == "simplex":
        if abs(time_diff) < 0.22: return closest_beat_time
        return time_val

    if genre in ["metal", "rock"]:
        diff = (time_val - closest_beat_time) % 0.25
        if diff < 0.04 or diff > 0.21: return round(time_val * 4) / 4
    elif genre in ["salsa", "timba"]:
        # Anticipation adjustment: traditional Latin downbeat push
        if subdivision == 4 and abs(time_diff) < 0.20:
            return closest_beat_time - 0.08
    elif genre == "tango":
        # Rigid downbeat alignment for standard marcato accents
        if abs(time_diff) < 0.18: return closest_beat_time
    elif genre in ["funk", "jazz", "swing"]:
        if subdivision == 1 and abs(time_diff) < 0.15: return closest_beat_time
        elif 0.08 < time_diff < 0.25: return closest_beat_time + 0.16 # Swing feel
    elif genre in ["hiphop", "rnb"]:
        if abs(time_diff) < 0.12 and time_diff >= 0: return closest_beat_time + 0.02

    return time_val

def merge_legato_phrases(raw_notes, gap_tolerance):
    if not raw_notes: return []
    
    sorted_notes = sorted(raw_notes, key=lambda x: x.start)
    merged = []
    current = sorted_notes[0]
    
    for next_note in sorted_notes[1:]:
        # If notes share a pitch and are grouped closer than tolerance threshold, merge them
        if next_note.pitch == current.pitch and (next_note.start - current.end) <= gap_tolerance:
            current.end = max(current.end, next_note.end)
            current.velocity = max(current.velocity, next_note.velocity)
        else:
            merged.append(current)
            current = next_note
    merged.append(current)
    return merged

def process_score_tier(raw_notes, level, genre, bpm, beat_times, project_name, detected_key):
    cfg = get_config(genre)
    
    sc = stream.Score()
    sc.insert(0, metadata.Metadata(title=f"{project_name.title()} - Bass [{level.upper()}]"))
    sc.insert(0, tempo.MetronomeMark(number=bpm))
    
    k_tonic, k_mode = detected_key.split()
    sc.insert(0, key.Key(k_tonic, k_mode))

    filtered_events = []
    for ev in raw_notes:
        if ev.pitch < 23 or ev.pitch > 68: continue
        if level in ["simplex", "normal"] and ev.pitch > 57: continue
        if level == "simplex" and ev.velocity < 45 and genre not in ["funk", "rnb"]: continue
        filtered_events.append(ev)

    if not filtered_events: return None

    # Apply core cleaning to remove artificial ties
    cleaned_events = merge_legato_phrases(filtered_events, cfg["legato_gap_tolerance"])

    grouped_events = []
    for ev in cleaned_events:
        if not grouped_events:
            grouped_events.append([ev])
        else:
            if abs(ev.start - grouped_events[-1][0].start) < 0.03:
                grouped_events[-1].append(ev)
            else:
                grouped_events.append([ev])
                
    for i in range(len(grouped_events) - 1):
        next_start = grouped_events[i+1][0].start
        for n_ev in grouped_events[i]:
            if n_ev.end > next_start:
                n_ev.end = next_start

    part = stream.Part(id=f"Bass_{level}")
    part.insert(0, meter.TimeSignature('4/4'))
    part.insert(0, instrument.DoubleBass() if cfg["use_double_bass"] else instrument.ElectricBass())
    part.insert(0, clef.BassClef())

    current_dynamic = None
    factor = 2 if level == "simplex" else 4

    for idx, event_group in enumerate(grouped_events):
        base_ev = event_group[0]
        
        adjusted_start = apply_bass_cultural_grid(base_ev.start, beat_times, genre, level)
        adjusted_end = apply_bass_cultural_grid(base_ev.end, beat_times, genre, level)
        
        if adjusted_end <= adjusted_start:
            adjusted_end = adjusted_start + max(0.05, cfg["minimum_note_length"])
            
        raw_offset = adjusted_start * (bpm / 60.0)
        raw_dur = (adjusted_end - adjusted_start) * (bpm / 60.0)
        
        offset_val = float(round(raw_offset * factor) / factor)
        q_length = float(max(1.0 if level == "simplex" else 0.25, round(raw_dur * factor) / factor))

        if len(event_group) == 1:
            music_element = note.Note(pitch.Pitch(midi=int(round(base_ev.pitch))))
        else:
            pitches = [pitch.Pitch(midi=int(round(e.pitch))) for e in event_group]
            music_element = chord.Chord(pitches)

        music_element.duration.quarterLength = q_length
        
        target_dynamic = None
        if base_ev.velocity > 100: target_dynamic = 'ff'
        elif base_ev.velocity > 80: target_dynamic = 'f'
        elif base_ev.velocity < 50: target_dynamic = 'p'
        else: target_dynamic = 'mf'

        if target_dynamic != current_dynamic and target_dynamic not in ['p']:
            music_element.dynamic = target_dynamic
            current_dynamic = target_dynamic

        if base_ev.velocity < 50 and genre in ["funk", "rnb", "hiphop"]:
            music_element.notehead = 'x'
        elif genre == "funk" and base_ev.velocity > 90 and raw_dur < 0.2:
            music_element.addLyric('P' if base_ev.pitch > 45 else 'T')

        if idx < len(grouped_events) - 1:
            next_start = grouped_events[idx+1][0].start
            if genre in ["jazz", "tango"] and (next_start - base_ev.end) < 0.15:
                music_element.articulations.append(articulations.Tenuto())

        if len(event_group) == 1:
            s_name, f_val = map_pedagogical_fretboard(base_ev.pitch, genre)
            music_element.addLyric(f"{s_name}:{f_val}")

        part.insert(offset_val, music_element)

    if len(part.flatten().notesAndRests) > 0:
        part.makeRests(fillGaps=True, inPlace=True)
        part = part.makeMeasures()
        try:
            part.makeBeams(inPlace=True)
        except Exception:
            pass
        sc.append(part)

    sc.makeAccidentals(inPlace=True)
    return sc

def main():
    stems_dir = Path(sys.argv[1]).resolve()
    genre = sys.argv[2].lower()
    out_dir = Path("./output_bass_advanced").resolve()
    project_name = stems_dir.name.replace('stems_', '')

    bass_wav = stems_dir / "bass.wav"
    drums_wav = stems_dir / "drums.wav"

    if not bass_wav.exists():
        print(f"Error: bass.wav missing from {stems_dir}")
        sys.exit(1)

    cfg = get_config(genre)
    print(f"Processing '{project_name}' using optimization profile for [{genre.upper()}]...")

    tracking_wav = drums_wav if drums_wav.exists() else bass_wav
    y_track, sr_track = librosa.load(str(tracking_wav), sr=22050, mono=True, res_type='soxr_hq')
    
    start_bpms = {
        'metal': 160.0, 'edm': 130.0, 'rock': 140.0, 'salsa': 95.0, 'timba': 100.0,
        'tango': 120.0, 'funk': 105.0, 'rnb': 90.0, 'hiphop': 90.0, 'jazz': 130.0
    }
    prior_tempo = start_bpms.get(genre, 120.0)

    if len(y_track) > 0 and not np.all(y_track == 0):
        y_track = librosa.util.normalize(y_track)
        onset_env = librosa.onset.onset_strength(y=y_track, sr=sr_track, aggregate=np.median)
        onset_median = np.median(onset_env)
        
        if onset_median < 0.1:
            bpm = prior_tempo
            total_dur = librosa.get_duration(y=y_track, sr=sr_track)
            beat_times = np.arange(0, total_dur, 60.0 / bpm)
        else:
            tempo_est, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr_track, start_bpm=prior_tempo)
            bpm = float(np.median(tempo_est))
            beat_times = librosa.frames_to_time(beats, sr=sr_track)
    else:
        bpm, beat_times = 120.0, np.arange(0.0, 300.0, 0.5)

    y_b, sr_b = librosa.load(str(bass_wav), sr=22050, mono=True, res_type='soxr_hq')
    detected_key = estimate_harmonic_key(y_b, sr_b)

    # Context-Aware Adaptive Bandpass Filtering
    nyq = 0.5 * sr_b
    low_hz = max(1.0, cfg["low_cut"])
    high_hz = min(nyq - 1.0, cfg["high_cut"])
    b, a = butter(4, [low_hz / nyq, high_hz / nyq], btype='band')
    y_b_filtered = filtfilt(b, a, y_b)

    cond_wav = out_dir / "temp_conditioned_bass.wav"
    sf.write(str(cond_wav), librosa.util.normalize(y_b_filtered), sr_b)
    
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _, bass_midi, _ = bp_predict(
                audio_path=str(cond_wav),
                onset_threshold=cfg["onset_threshold"],
                frame_threshold=cfg["frame_threshold"],
                minimum_note_length=cfg["minimum_note_length"]
            )
            
        raw_notes = bass_midi.instruments[0].notes
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)
    finally:
        if cond_wav.exists(): cond_wav.unlink()

    if len(raw_notes) == 0:
        print("Error: No midi entries detected.")
        sys.exit(1)

    for lvl in ["simplex", "normal", "complex"]:
        score_obj = process_score_tier(raw_notes, lvl, genre, bpm, beat_times, project_name, detected_key)
        if score_obj:
            genre_flag = "standard" if genre == "none" else genre
            target_output = out_dir / f"{project_name}_bass_{lvl}_{genre_flag}.musicxml"
            score_obj.write('musicxml', fp=str(target_output))

    print(f"Success: Files written to {out_dir}/")

if __name__ == "__main__":
    main()
EOF

"$ENV_DIR/bin/python" run_engine_bass.py "$STEMS_DIR" "$TARGET_GENRE"
