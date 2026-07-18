#!/usr/bin/env bash
# ==============================================================================
# Bass Transcription Pipeline (Clean Output Edition)
# Usage: ./advanced.sh <path_to_stems_folder> [genre]
# ==============================================================================

set -euo pipefail

STEMS_DIR="${1:-}"
TARGET_GENRE="${2:-none}"

if [ -z "$STEMS_DIR" ] || [ ! -d "$STEMS_DIR" ]; then
    echo "Error: Invalid stems directory path."
    echo "Usage: $0 <path_to_stems_folder> [genre: salsa|bachata|kizomba|zouk|hiphop|swing|reggae|funk|neosoul]"
    exit 1
fi

TARGET_GENRE=$(echo "${TARGET_GENRE}" | tr '[:upper:]' '[:lower:]')

if [ "$TARGET_GENRE" != "none" ]; then
    case "${TARGET_GENRE}" in
        salsa|bachata|kizomba|zouk|hiphop|swing|reggae|funk|neosoul) ;;
        *)
            echo "Error: Unsupported genre '${TARGET_GENRE}'"
            exit 1
            ;;
    esac
fi

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
    echo "Creating virtual environment..."
    $PY_CMD -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

echo "Checking pipeline dependencies..."
"$ENV_DIR/bin/python" -m pip install --upgrade pip wheel &> /dev/null
"$ENV_DIR/bin/python" -m pip install "setuptools<82" &> /dev/null

"$ENV_DIR/bin/python" -m pip install --no-cache-dir \
    "numpy==1.26.4" "scipy==1.12.0" "soundfile==0.12.1" \
    "librosa==0.10.1" "music21==9.1.0" "pretty_midi==0.2.10" \
    "basic-pitch>=0.4.0" "torch==2.2.1" "resampy==0.4.2" &> /dev/null

cleanup() { rm -f run_engine_bass.py; }
trap cleanup EXIT

cat << 'EOF' > run_engine_bass.py
import sys
import os
import logging
import torch
import warnings
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

# Silence all third-party package logger text & runtime warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# HARDWARE OPTIMIZATION: Enforce Metal Performance Shaders for M1
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

from basic_pitch.inference import predict as bp_predict
from music21 import instrument, clef, metadata, tempo, stream, note, meter, key, spanner, pitch

def estimate_harmonic_key(y, sr):
    if len(y) == 0 or np.all(y == 0):
        return "C major"
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=12)
    mean_chroma = np.mean(chroma, axis=1)
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    root_idx = int(np.argmax(mean_chroma))
    dominant_pc = pitch_classes[root_idx]
    
    maj_third_idx = (root_idx + 4) % 12
    min_third_idx = (root_idx + 3) % 12
    mode = "minor" if mean_chroma[min_third_idx] > mean_chroma[maj_third_idx] else "major"
    return f"{dominant_pc} {mode}"

def map_pedagogical_fretboard(midi_pitch, level, use_double_bass=False):
    pitch_val = max(28, min(midi_pitch, 67))
    string_baselines = [43, 38, 33, 28]
    string_names = ["G", "D", "A", "E"]
    best_string = "E"
    selected_fret = 0
    
    if use_double_bass:
        for idx, base in enumerate(string_baselines):
            if pitch_val >= base:
                fret_equiv = pitch_val - base
                if fret_equiv == 0: return string_names[idx], "Open"
                elif fret_equiv <= 2: return string_names[idx], "Pos:1/2"
                elif fret_equiv <= 4: return string_names[idx], "Pos:1"
                elif fret_equiv <= 7: return string_names[idx], "Pos:2"
                else: return string_names[idx], "Pos:High"
        return "G", "Pos:High"

    if level == "simplex":
        for idx, base in enumerate(string_baselines):
            if pitch_val >= base:
                fret = pitch_val - base
                if fret <= 5: return string_names[idx], f"F:{int(fret)}"
        return "G", f"F:{int(pitch_val - 43)}"
    elif level == "normal":
        for idx, base in enumerate(string_baselines):
            if pitch_val >= base:
                fret = pitch_val - base
                if 2 <= fret <= 10: return string_names[idx], f"F:{int(fret)}"
        for idx, base in enumerate(string_baselines):
            if pitch_val >= base: return string_names[idx], f"F:{int(pitch_val - base)}"
        return "E", "F:0"
    else:
        min_fret = 99
        for idx, base in enumerate(string_baselines):
            if pitch_val >= base:
                fret = pitch_val - base
                if fret < min_fret and fret <= 21:
                    min_fret = fret
                    best_string = string_names[idx]
                    selected_fret = fret
        return best_string, f"F:{int(selected_fret)}"

def apply_bass_cultural_grid(note_start, beat_times, genre, level):
    if len(beat_times) == 0 or genre == "none":
        return note_start
        
    closest_beat_idx = np.argmin(np.abs(beat_times - note_start))
    closest_beat_time = beat_times[closest_beat_idx]
    time_diff = note_start - closest_beat_time
    subdivision = (closest_beat_idx % 4) + 1

    if level == "simplex":
        if abs(time_diff) < 0.22: return closest_beat_time
        return note_start

    if genre == "salsa" and subdivision == 4 and abs(time_diff) < 0.20:
        return closest_beat_time - 0.08
    elif genre == "bachata" and subdivision == 4 and abs(time_diff) < 0.14:
        return closest_beat_time
    elif genre in ["kizomba", "zouk"] and abs(time_diff) < 0.09:
        return closest_beat_time
    elif genre == "hiphop" and abs(time_diff) < 0.12 and time_diff >= 0:
        return closest_beat_time + 0.02
    elif genre == "swing" and 0.08 < time_diff < 0.24:
        return closest_beat_time + 0.15
    elif genre == "reggae":
        if subdivision == 1 and abs(time_diff) < 0.16: return closest_beat_time + 0.06
        elif subdivision == 3 and abs(time_diff) < 0.18: return closest_beat_time
    elif genre == "funk" and subdivision == 1 and abs(time_diff) < 0.15:
        return closest_beat_time
    elif genre == "neosoul" and abs(time_diff) < 0.15:
        return closest_beat_time + 0.038

    return note_start

def process_score_tier(raw_notes, level, genre, bpm, beat_times, project_name, detected_key):
    use_double_bass = genre in ["swing", "salsa"]
    genre_display = "Standard" if genre == "none" else genre.upper()
    
    sc = stream.Score()
    sc.insert(0, metadata.Metadata(title=f"{project_name.title()} - Bass Line [{level.upper()}] ({genre_display})"))
    sc.insert(0, tempo.MetronomeMark(number=bpm))
    
    k_tonic, k_mode = detected_key.split()
    master_key = key.Key(k_tonic, k_mode)

    sorted_events = sorted(raw_notes, key=lambda x: x.start)
    cleaned_events = []
    
    # Pre-processing note cleanups
    for ev in sorted_events:
        if ev.pitch < 28 or ev.pitch > 68: continue
        if level in ["simplex", "normal"] and ev.pitch > 57: continue
        if level == "simplex" and ev.velocity < 45: continue
            
        if cleaned_events:
            prev_ev = cleaned_events[-1]
            if ev.start < prev_ev.end:
                if abs(ev.start - prev_ev.start) < 0.015:
                    if ev.velocity > prev_ev.velocity: cleaned_events[-1] = ev
                    continue
                else:
                    prev_ev.end = ev.start
                    if prev_ev.end - prev_ev.start < 0.06: cleaned_events.pop()
                        
        cleaned_events.append(ev)

    # POLYPHONY FIX: Split overlapping events into separate musical layers
    layers = [[]]
    for ev in cleaned_events:
        placed = False
        for layer_notes in layers:
            # If layer is empty, or the note begins after the previous note ends (with a 50ms tolerance)
            if not layer_notes or ev.start >= layer_notes[-1].end - 0.05:
                layer_notes.append(ev)
                placed = True
                break
        if not placed:
            layers.append([ev])

    divs = (1, 2) if level == "simplex" else ((2, 4) if level == "normal" else (2, 4, 8))

    # Process each polyphonic layer as an independent Part
    for layer_idx, layer_notes in enumerate(layers):
        part = stream.Part(id=f"Bass_{level}_Layer_{layer_idx+1}")
        part.insert(0, meter.TimeSignature('4/4'))
        part.insert(0, master_key)
        
        if use_double_bass: part.insert(0, instrument.DoubleBass())
        else: part.insert(0, instrument.ElectricBass())
        part.insert(0, clef.BassClef())

        for idx, n_event in enumerate(layer_notes):
            adjusted_start = apply_bass_cultural_grid(n_event.start, beat_times, genre, level)
            raw_offset = adjusted_start * (bpm / 60.0)
            raw_duration = (n_event.end - n_event.start) * (bpm / 60.0)
            
            factor = 2 if level == "simplex" else 4
            offset_val = round(raw_offset * factor) / factor
            q_length = max(1.0 if level == "simplex" else 0.25, round(raw_duration * factor) / factor)

            p_obj = pitch.Pitch(midi=int(round(n_event.pitch)))
            n = note.Note(p_obj)
            n.duration.quarterLength = q_length
            
            if n_event.velocity > 100: n.dynamic = 'ff'
            elif n_event.velocity > 80: n.dynamic = 'f'
            elif n_event.velocity < 50:
                n.dynamic = 'p'
                if level == "complex": n.addLyric("(ghost)")

            s_name, pos_val = map_pedagogical_fretboard(n_event.pitch, level, use_double_bass)
            n.addLyric(f"{s_name}:{pos_val}")
            
            if idx < len(layer_notes) - 1:
                next_ev = layer_notes[idx + 1]
                if next_ev.start - n_event.end < 0.04 and abs(next_ev.pitch - n_event.pitch) <= 2:
                    n.is_legato_start = True

            part.insert(offset_val, n)

        if len(part.flatten().notes) > 0:
            # OVERFLOW FIX: Pad rests strictly before allowing makeMeasures() to dynamically slice and tie
            part.quantize(quarterLengthDivisors=divs, inPlace=True)
            part.makeRests(fillGaps=True, inPlace=True)
            
            # Avoid using inPlace=True here. Reassigning ensures valid, non-corrupted measure boundaries.
            part = part.makeMeasures()
            
            part.makeBeams(inPlace=True)
            part.makeTies(inPlace=True)
            
            flat_notes = list(part.flatten().notes)
            for idx in range(len(flat_notes) - 1):
                if getattr(flat_notes[idx], 'is_legato_start', False):
                    try:
                        slur = spanner.Slur(flat_notes[idx], flat_notes[idx+1])
                        sc.insert(0, slur)
                    except: pass

            sc.append(part)

    if len(sc.parts) > 0:
        sc.makeAccidentals(inPlace=True)
        return sc
    return None

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

    print("Analyzing audio and tracking beats...")
    tracking_wav = drums_wav if drums_wav.exists() else bass_wav
    y_track, sr_track = librosa.load(str(tracking_wav), sr=22050, mono=True)
    
    if len(y_track) > 0 and not np.all(y_track == 0):
        y_track = librosa.util.normalize(y_track)
        onset_env = librosa.onset.onset_strength(y=y_track, sr=sr_track, aggregate=np.median)
        tempo_est, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr_track)
        bpm = float(np.median(tempo_est))
        beat_times = librosa.frames_to_time(beats, sr=sr_track)
        print(f" -> Tempo: {bpm:.2f} BPM")
    else:
        bpm, beat_times = 120.0, np.arange(0.0, 300.0, 0.5)
        print(" -> Warning: Silent track. Defaulting to 120 BPM.")

    y_b, sr_b = librosa.load(str(bass_wav), sr=22050, mono=True)
    y_b = librosa.util.normalize(y_b)
    
    detected_key = estimate_harmonic_key(y_b, sr_b)
    print(f" -> Key: {detected_key.upper()}")

    cond_wav = out_dir / "temp_conditioned_bass.wav"
    sf.write(str(cond_wav), y_b, sr_b)
    
    print("Running transcription models...")
    try:
        # Route standard output out into null space to eat basic-pitch/coreml loop prints
        sys.stdout = open(os.devnull, 'w')
        _, bass_midi, _ = bp_predict(str(cond_wav))
        raw_notes = bass_midi.instruments[0].notes
    finally:
        sys.stdout = sys.__stdout__
        if cond_wav.exists(): cond_wav.unlink()

    if len(raw_notes) == 0:
        print("Error: No midi entries detected from audio stream.")
        sys.exit(1)

    print("Generating notation output tiers...")
    for lvl in ["simplex", "normal", "complex"]:
        score_obj = process_score_tier(raw_notes, lvl, genre, bpm, beat_times, project_name, detected_key)
        if score_obj:
            genre_flag = "standard" if genre == "none" else genre
            target_output = out_dir / f"{project_name}_bass_{lvl}_{genre_flag}.musicxml"
            score_obj.write('musicxml', fp=str(target_output))
            print(f" -> Exported: {target_output.name}")

    print(f"\nProcessing complete. Files written to: {out_dir.name}\n")

if __name__ == "__main__":
    main()
EOF

"$ENV_DIR/bin/python" run_engine_bass.py "$STEMS_DIR" "$TARGET_GENRE"
