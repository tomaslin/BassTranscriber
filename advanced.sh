#!/usr/bin/env bash
# ==============================================================================
# Bass Transcription Pipeline (M1 / Apple Silicon Optimized Edition v7.1)
# Usage: ./advanced_2.sh <path_to_stems_folder> [--tuning <tuning_type>]
# ==============================================================================

set -euo pipefail

STEMS_DIR=""
TUNING="auto"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --tuning)
      TUNING="$2"
      shift 2
      ;;
    *)
      if [ -z "$STEMS_DIR" ]; then
          STEMS_DIR="$1"
      fi
      shift
      ;;
  esac
done

if [ -z "$STEMS_DIR" ] || [ ! -d "$STEMS_DIR" ]; then
    echo "Error: Invalid stems directory path. Usage: $0 <path_to_stems_folder> [--tuning <tuning_type>]"
    exit 1
fi

TUNING=$(echo "${TUNING}" | tr '[:upper:]' '[:lower:]')

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

echo "Installing dependencies (logging to $ENV_DIR/pip_install.log)..."
"$ENV_DIR/bin/python" -m pip install --upgrade pip wheel > "$ENV_DIR/pip_install.log" 2>&1
"$ENV_DIR/bin/python" -m pip install "setuptools<82" >> "$ENV_DIR/pip_install.log" 2>&1

OS_NAME=$(uname -s)
ARCH_NAME=$(uname -m)

if [ "$OS_NAME" = "Darwin" ] && [ "$ARCH_NAME" = "arm64" ]; then
    "$ENV_DIR/bin/python" -m pip install --no-cache-dir "tensorflow-macos<2.16.0" "tensorflow-metal==1.1.0" >> "$ENV_DIR/pip_install.log" 2>&1
else
    "$ENV_DIR/bin/python" -m pip install --no-cache-dir "tensorflow<2.16.0" >> "$ENV_DIR/pip_install.log" 2>&1
fi

"$ENV_DIR/bin/python" -m pip install --no-cache-dir \
    "numpy==1.26.4" "scipy==1.12.0" "soundfile==0.12.1" "soxr==0.3.7" \
    "librosa==0.10.1" "music21==9.1.0" "pretty_midi==0.2.10" \
    "basic-pitch>=0.4.0" "resampy==0.4.2" >> "$ENV_DIR/pip_install.log" 2>&1

cleanup() { rm -f run_engine_bass.py; }
trap cleanup EXIT

cat << 'EOF' > run_engine_bass.py
import sys
import os
import logging
import warnings
import contextlib
import io
from fractions import Fraction
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['TF_NUM_INTRAOP_THREADS'] = '8'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import librosa
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, filtfilt
from basic_pitch.inference import predict as bp_predict
from music21 import instrument, clef, metadata, tempo, stream, note, chord, meter, key, articulations, pitch, spanner, expressions

# ==============================================================================
# EXPANDED SUBGENRE OPTIMIZATION REGISTRY
# ==============================================================================
SUBGENRE_REGISTRY = {
    "metal": {"low_cut": 30, "high_cut": 2500, "onset_threshold": 0.70, "frame_threshold": 0.40, "minimum_note_length": 0.05, "legato_gap_tolerance": 0.03},
    "rock": {"low_cut": 50, "high_cut": 2000, "onset_threshold": 0.60, "frame_threshold": 0.45, "minimum_note_length": 0.06, "legato_gap_tolerance": 0.04},
    "salsa": {"low_cut": 40, "high_cut": 1200, "onset_threshold": 0.65, "frame_threshold": 0.50, "minimum_note_length": 0.08, "legato_gap_tolerance": 0.05},
    "funk": {"low_cut": 30, "high_cut": 4000, "onset_threshold": 0.65, "frame_threshold": 0.45, "minimum_note_length": 0.04, "legato_gap_tolerance": 0.02},
    "jazz": {"low_cut": 40, "high_cut": 1500, "onset_threshold": 0.55, "frame_threshold": 0.50, "minimum_note_length": 0.07, "legato_gap_tolerance": 0.05},
    "reggae": {"low_cut": 25, "high_cut": 500, "onset_threshold": 0.45, "frame_threshold": 0.60, "minimum_note_length": 0.10, "legato_gap_tolerance": 0.08},
    "electronic": {"low_cut": 20, "high_cut": 4500, "onset_threshold": 0.65, "frame_threshold": 0.40, "minimum_note_length": 0.03, "legato_gap_tolerance": 0.02},
    "rnb": {"low_cut": 35, "high_cut": 1000, "onset_threshold": 0.50, "frame_threshold": 0.50, "minimum_note_length": 0.06, "legato_gap_tolerance": 0.06},
    "country": {"low_cut": 40, "high_cut": 1000, "onset_threshold": 0.55, "frame_threshold": 0.50, "minimum_note_length": 0.08, "legato_gap_tolerance": 0.04},
    "punk": {"low_cut": 50, "high_cut": 3000, "onset_threshold": 0.65, "frame_threshold": 0.45, "minimum_note_length": 0.04, "legato_gap_tolerance": 0.03},
    "house": {"low_cut": 25, "high_cut": 1500, "onset_threshold": 0.65, "frame_threshold": 0.45, "minimum_note_length": 0.04, "legato_gap_tolerance": 0.02},
    "disco": {"low_cut": 40, "high_cut": 3000, "onset_threshold": 0.55, "frame_threshold": 0.50, "minimum_note_length": 0.05, "legato_gap_tolerance": 0.04},
    "synthwave": {"low_cut": 20, "high_cut": 4500, "onset_threshold": 0.60, "frame_threshold": 0.40, "minimum_note_length": 0.03, "legato_gap_tolerance": 0.02},
    "reggaeton": {"low_cut": 25, "high_cut": 800, "onset_threshold": 0.50, "frame_threshold": 0.55, "minimum_note_length": 0.06, "legato_gap_tolerance": 0.05},
    "afrobeats": {"low_cut": 35, "high_cut": 2000, "onset_threshold": 0.55, "frame_threshold": 0.50, "minimum_note_length": 0.06, "legato_gap_tolerance": 0.06},
    "blues": {"low_cut": 45, "high_cut": 1500, "onset_threshold": 0.50, "frame_threshold": 0.50, "minimum_note_length": 0.08, "legato_gap_tolerance": 0.05},
    "bossanova": {"low_cut": 40, "high_cut": 1200, "onset_threshold": 0.55, "frame_threshold": 0.55, "minimum_note_length": 0.08, "legato_gap_tolerance": 0.06},
    "hiphop": {"low_cut": 20, "high_cut": 800, "onset_threshold": 0.40, "frame_threshold": 0.60, "minimum_note_length": 0.10, "legato_gap_tolerance": 0.08},
    "dnb": {"low_cut": 20, "high_cut": 3000, "onset_threshold": 0.60, "frame_threshold": 0.50, "minimum_note_length": 0.03, "legato_gap_tolerance": 0.02},
    "pop": {"low_cut": 35, "high_cut": 2000, "onset_threshold": 0.55, "frame_threshold": 0.50, "minimum_note_length": 0.05, "legato_gap_tolerance": 0.04},
    "ska": {"low_cut": 45, "high_cut": 2500, "onset_threshold": 0.65, "frame_threshold": 0.40, "minimum_note_length": 0.04, "legato_gap_tolerance": 0.02},
    "classical": {"low_cut": 30, "high_cut": 1500, "onset_threshold": 0.30, "frame_threshold": 0.65, "minimum_note_length": 0.15, "legato_gap_tolerance": 0.10},
    "none": {"low_cut": 40, "high_cut": 800, "onset_threshold": 0.50, "frame_threshold": 0.50, "minimum_note_length": 0.07, "legato_gap_tolerance": 0.05}
}

def get_config(genre):
    return SUBGENRE_REGISTRY.get(genre, SUBGENRE_REGISTRY["none"])

# ==============================================================================
# K-NEAREST NEIGHBORS AUDIO PROFILING (WEIGHTED AUTO-DETECTION)
# ==============================================================================
def auto_detect_profile(bass_wav_path, drums_wav_path):
    print("Initiating Auto-Detection profiling...")
    y_b, sr = librosa.load(str(bass_wav_path), sr=22050, mono=True, res_type='soxr_hq')
    
    if len(y_b) == 0:
        return "none", 120.0, y_b, sr

    # 1. Band-pass filtering (30-500Hz) to isolate bass character from noise
    nyq = 0.5 * sr
    b, a = butter(4, [30.0 / nyq, 500.0 / nyq], btype='band')
    y_filtered = filtfilt(b, a, y_b)
    
    # 2. Bass Features: Brightness (Centroid of filtered signal)
    avg_centroid = np.median(librosa.feature.spectral_centroid(y=y_filtered, sr=sr))

    # 3. Rhythm Features
    y_rhythm, _ = librosa.load(str(drums_wav_path if drums_wav_path.exists() else bass_wav_path), sr=22050, mono=True)
    onset_env = librosa.onset.onset_strength(y=y_rhythm, sr=sr)
    tempo_est, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    bpm = float(np.median(tempo_est)) if tempo_est.size > 0 else 120.0
    duration_sec = librosa.get_duration(y=y_rhythm, sr=sr)
    note_density = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)) / duration_sec if duration_sec > 0 else 0

    # Anchors: (BPM, Filtered_Centroid, Note_Density)
    genre_anchors = {
        "metal": (140, 400, 4.0), "house": (124, 150, 2.5),
        "disco": (115, 250, 3.5), "reggae": (80, 100, 1.5),
        "synthwave": (110, 300, 5.0), "funk": (105, 300, 4.0),
        "hiphop": (90, 120, 2.0), "jazz": (130, 200, 2.5),
        "rock": (120, 250, 3.0), "electronic": (128, 400, 4.5),
        "rnb": (90, 180, 2.0), "country": (100, 200, 2.5),
        "punk": (160, 400, 4.5), "reggaeton": (95, 150, 3.0),
        "afrobeats": (105, 200, 3.0), "blues": (80, 200, 2.0),
        "bossanova": (140, 180, 2.5), "dnb": (174, 400, 4.5),
        "pop": (115, 220, 2.5), "ska": (150, 350, 5.0),
        "classical": (75, 150, 1.0)
    }

    best_genre = "none"
    min_dist = float('inf')

    # Weighted Euclidean: BPM (2.0), Tone (1.5), Density (0.5)
    for g, (t_bpm, t_cent, t_dens) in genre_anchors.items():
        dist = np.sqrt(2.0*((bpm-t_bpm)/200)**2 + 1.5*((avg_centroid-t_cent)/500)**2 + 0.5*((note_density-t_dens)/8)**2)
        if dist < min_dist:
            min_dist, best_genre = dist, g

    print(f"  -> Analyzed BPM: {bpm:.1f} | Tone: {avg_centroid:.0f}Hz | Density: {note_density:.1f} | Profile: [{best_genre.upper()}]")
    return best_genre, bpm, y_b, sr

# ... (REST OF THE FUNCTIONS REMAIN UNCHANGED) ...

def estimate_harmonic_key(y, sr):
    if len(y) == 0 or np.all(y == 0): return "C major"
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=12)
    except librosa.util.exceptions.ParameterError:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)
    root_idx = int(np.argmax(mean_chroma))
    is_minor = mean_chroma[(root_idx + 3) % 12] > mean_chroma[(root_idx + 4) % 12]
    major_map = {0: 'C', 1: 'Db', 2: 'D', 3: 'Eb', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'Ab', 9: 'A', 10: 'Bb', 11: 'B'}
    minor_map = {0: 'C', 1: 'C#', 2: 'D', 3: 'Eb', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'Bb', 11: 'B'}
    return f"{minor_map[root_idx] if is_minor else major_map[root_idx]} {'minor' if is_minor else 'major'}"

def determine_tuning(min_pitch, requested_tuning):
    tunings = {
        "standard": ([43, 38, 33, 28], ["G", "D", "A", "E"]),
        "drop-d": ([43, 38, 33, 26], ["G", "D", "A", "D"]),
        "5-string": ([43, 38, 33, 28, 23], ["G", "D", "A", "E", "B"]),
        "6-string": ([48, 43, 38, 33, 28, 23], ["C", "G", "D", "A", "E", "B"])
    }
    if requested_tuning in tunings: return tunings[requested_tuning]
    if min_pitch < 28 and min_pitch >= 23: return tunings["5-string"]
    elif min_pitch < 23: return tunings["6-string"]
    return tunings["standard"]

def apply_viterbi_fretboard(notes, baselines, string_names):
    if not notes: return []
    states = []
    for s_idx, base in enumerate(baselines):
        for f in range(0, 25): states.append((s_idx, f, base + f))
    path_data = []
    for i, n in enumerate(notes):
        pitch_val = int(round(n.pitch))
        valid_states = [s for s in states if s[2] == pitch_val]
        if not valid_states: valid_states = [(0, max(0, pitch_val - baselines[0]), pitch_val)]
        step_paths = {}
        if i == 0:
            for vs in valid_states: step_paths[vs] = (abs(vs[1] - 5), [vs])
        else:
            prev_paths = path_data[-1]
            for vs in valid_states:
                best_cost, best_hist = float('inf'), []
                for ps, (prev_cost, prev_hist) in prev_paths.items():
                    move_cost = abs(vs[1] - ps[1]) + (abs(vs[0] - ps[0]) * 3) + (5 if (vs[1] == 0 and ps[1] > 5) else 0)
                    total_cost = prev_cost + move_cost
                    if total_cost < best_cost: best_cost, best_hist = total_cost, prev_hist + [vs]
                step_paths[vs] = (best_cost, best_hist)
        path_data.append(step_paths)
    best_final_state = min(path_data[-1].items(), key=lambda x: x[1][0])
    return [(string_names[s[0]], s[1]) for s in best_final_state[1][1]]

def extract_timbre(y_b, sr_b, start_time, end_time):
    start_samp, end_samp = int(start_time * sr_b), int(end_time * sr_b)
    if end_samp <= start_samp: return "normal"
    segment = y_b[start_samp:end_samp]
    if len(segment) < 512: return "normal"
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=segment))
    centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr_b))
    rms = np.mean(librosa.feature.rms(y=segment))
    if rms < 0.01: return "mute"
    if zcr > 0.08 and centroid > 2500: return "pop"
    if zcr > 0.05 and centroid > 1500: return "slap"
    return "normal"

def process_score_tier(raw_notes, pitch_bends, level, genre, bpm, detected_key, tuning, y_b, sr_b):
    cfg = get_config(genre)
    sc = stream.Score()
    sc.insert(0, metadata.Metadata(title=f"Bass Transcription [{level.upper()}]"))
    sc.insert(0, tempo.MetronomeMark(number=bpm))
    k_tonic, k_mode = detected_key.split()
    sc.insert(0, key.Key(k_tonic, k_mode))
    filtered_events = [ev for ev in raw_notes if not (ev.pitch < 15 or ev.pitch > 70) and (level != "simplex" or abs((ev.start * (bpm / 60.0)) % 2) <= 0.35) and (level != "normal" or ev.velocity >= 45)]
    if not filtered_events: return None
    filtered_events.sort(key=lambda x: x.start)
    grouped_events = []
    curr = [filtered_events[0]]
    for ev in filtered_events[1:]:
        if ev.start - curr[0].start < 0.03: curr.append(ev)
        else: grouped_events.append(curr); curr = [ev]
    grouped_events.append(curr)
    
    resolved_events = []
    for grp in grouped_events:
        grp.sort(key=lambda x: x.pitch)
        root = grp[0]
        chord_notes = [root] + [g for g in grp[1:] if (g.pitch - root.pitch) in [7, 12, 15, 16]]
        resolved_events.append(chord_notes)

    part = stream.Part(id=f"Bass_{level}")
    part.insert(0, meter.TimeSignature('4/4'))
    part.insert(0, clef.BassClef())
    baselines, string_names = determine_tuning(min([n.pitch for n in raw_notes]), tuning)
    optimal_fingerings = apply_viterbi_fretboard([grp[0] for grp in resolved_events], baselines, string_names)
    factor, min_len = 2 if level == "simplex" else 12, 1.0 if level == "simplex" else 0.25
    quantized_events = []
    
    for idx, event_group in enumerate(resolved_events):
        base_ev = event_group[0]
        raw_start = base_ev.start * (bpm / 60.0)
        q_start = round(raw_start * factor) / factor
        q_end = (round(base_ev.end * (bpm / 60.0) * factor) / factor) if level != "simplex" else (q_start + 2.0)
        if q_end <= q_start: q_end = q_start + min_len
        quantized_events.append({"start": q_start, "end": q_end, "group": event_group, "base": base_ev, "fingering": optimal_fingerings[idx]})

    for i in range(len(quantized_events) - 1):
        if quantized_events[i+1]["start"] - quantized_events[i]["start"] < min_len: quantized_events[i+1]["start"] = quantized_events[i]["start"] + min_len
        if quantized_events[i]["end"] > quantized_events[i+1]["start"]: quantized_events[i]["end"] = quantized_events[i+1]["start"]

    for idx, q_ev in enumerate(quantized_events):
        event_group, base_ev = q_ev["group"], q_ev["base"]
        s_name, f_val = q_ev["fingering"]
        is_staccato = (q_ev["end"] - q_ev["start"] == Fraction(3, 12)) and level != "simplex" and (idx < len(quantized_events)-1) and (quantized_events[idx+1]["start"] - q_ev["end"] >= 0.25)
        
        music_element = chord.Chord([pitch.Pitch(midi=int(round(e.pitch))) for e in event_group]) if len(event_group) > 1 else note.Note(pitch.Pitch(midi=int(round(base_ev.pitch))))
        music_element.duration.quarterLength = Fraction(int(round((q_ev["end"] - q_ev["start"]) * factor)), factor)
        if is_staccato: music_element.articulations.append(articulations.Staccato())
        
        timbre = extract_timbre(y_b, sr_b, base_ev.start, base_ev.end)
        if base_ev.velocity < 50 or timbre == "mute":
            if music_element.isChord:
                for n in music_element.notes: n.notehead = 'x'
            else:
                music_element.notehead = 'x'
        if timbre == "pop": music_element.addLyric('P')
        elif timbre == "slap": music_element.addLyric('T')
        
        str_num = {"G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "C": 6}.get(s_name, 4)
        music_element.articulations.append(articulations.StringIndication(str_num))
        music_element.articulations.append(articulations.FretIndication(f_val))
        part.insert(Fraction(int(round(q_ev["start"] * factor)), factor), music_element)

    if len(part.flatten().notesAndRests) > 0:
        part.makeRests(fillGaps=True, inPlace=True)
        part = part.makeMeasures()
        part.makeNotation(inPlace=True)
        sc.append(part)
    sc.makeAccidentals(inPlace=True, overrideStatus=True)
    return sc

def main():
    stems_dir = Path(sys.argv[1]).resolve()
    tuning_pref = sys.argv[2].lower() if len(sys.argv) > 2 else "auto"
    out_dir = Path("./output_bass_advanced").resolve()
    project_name = stems_dir.name.replace('stems_', '')
    bass_wav, drums_wav = stems_dir / "bass.wav", stems_dir / "drums.wav"
    if not bass_wav.exists(): print(f"Error: bass.wav missing"); sys.exit(1)
    
    genre, bpm, y_b, sr_b = auto_detect_profile(bass_wav, drums_wav)
    cfg = get_config(genre)
    detected_key = estimate_harmonic_key(y_b, sr_b)
    
    nyq = 0.5 * sr_b
    b, a = butter(4, [max(1.0, cfg["low_cut"]) / nyq, min(nyq - 1.0, cfg["high_cut"]) / nyq], btype='band')
    cond_wav = out_dir / "temp_conditioned_bass.wav"
    sf.write(str(cond_wav), filtfilt(b, a, y_b), int(sr_b))
    
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _, bass_midi, _ = bp_predict(audio_path=str(cond_wav), onset_threshold=cfg["onset_threshold"], frame_threshold=cfg["frame_threshold"], minimum_note_length=cfg["minimum_note_length"])
        if not bass_midi.instruments: sys.exit(0)
        raw_notes, pitch_bends = bass_midi.instruments[0].notes, bass_midi.instruments[0].pitch_bends
    finally:
        if cond_wav.exists(): cond_wav.unlink()

    for lvl in ["simplex", "normal", "complex"]:
        score_obj = process_score_tier(raw_notes, pitch_bends, lvl, genre, bpm, detected_key, tuning_pref, y_b, sr_b)
        if score_obj: score_obj.write('musicxml', fp=str(out_dir / f"{project_name}_bass_{lvl}_{genre}.musicxml"))
    print(f"Success: Files written to {out_dir}/")

if __name__ == "__main__":
    main()
EOF

"$ENV_DIR/bin/python" run_engine_bass.py "$STEMS_DIR" "$TUNING"
