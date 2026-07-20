#!/usr/bin/env bash
# ==============================================================================
# Bass Transcription Pipeline (M1 Optimized)
# Usage: ./process.sh <path_to_stems_folder> [--tuning <tuning_type>]
# ==============================================================================

set -euo pipefail

STEMS_DIRS=()
TUNING="auto"
GENRE_OVERRIDE="auto"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --tuning)
      TUNING="$2"
      shift 2
      ;;
    --genre)
      GENRE_OVERRIDE="$2"
      shift 2
      ;;
    *)
      STEMS_DIRS+=("$1")
      shift
      ;;
  esac
done

if [ ${#STEMS_DIRS[@]} -eq 0 ]; then
    echo "Error: No stems directory provided. Usage: $0 <path_to_stems_folder1> [<path_to_stems_folder2> ...] [--tuning <tuning_type>] [--genre <genre_type>]"
    exit 1
fi

TUNING=$(echo "${TUNING}" | tr '[:upper:]' '[:lower:]')
GENRE_OVERRIDE=$(echo "${GENRE_OVERRIDE}" | tr '[:upper:]' '[:lower:]')

if [ -d "/opt/homebrew/bin" ]; then export PATH="/opt/homebrew/bin:$PATH"; fi

if command -v python3.11 &> /dev/null; then
    PY_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PY_CMD="python3"
else
    echo "Error: Python 3.11+ required."; exit 1;
fi

OUT_DIR="./output_bass"
ENV_DIR=".venv_bass"
mkdir -p "$OUT_DIR"

if [ ! -d "$ENV_DIR" ]; then
    $PY_CMD -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

# Installing dependencies silently
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
from music21 import instrument, clef, metadata, tempo, stream, note, chord, meter, key, articulations, pitch, spanner, expressions, harmony, dynamics

# ==============================================================================
# EXPANDED SUBGENRE OPTIMIZATION REGISTRY
# ==============================================================================
SUBGENRE_REGISTRY = {
    "metal": {"low_cut": 30, "high_cut": 2500, "onset_threshold": 0.70, "frame_threshold": 0.40, "minimum_note_length": 0.05, "legato_gap_tolerance": 0.03},
    "rock": {"low_cut": 50, "high_cut": 2000, "onset_threshold": 0.60, "frame_threshold": 0.45, "minimum_note_length": 0.06, "legato_gap_tolerance": 0.04},
    "salsa": {"low_cut": 40, "high_cut": 1200, "onset_threshold": 0.65, "frame_threshold": 0.50, "minimum_note_length": 0.08, "legato_gap_tolerance": 0.05},
    "funk": {"low_cut": 30, "high_cut": 4000, "onset_threshold": 0.65, "frame_threshold": 0.45, "minimum_note_length": 0.04, "legato_gap_tolerance": 0.02},
    "jazz": {"low_cut": 40, "high_cut": 1500, "onset_threshold": 0.55, "frame_threshold": 0.50, "minimum_note_length": 0.07, "legato_gap_tolerance": 0.05},
    "swing": {"low_cut": 40, "high_cut": 1200, "onset_threshold": 0.55, "frame_threshold": 0.50, "minimum_note_length": 0.08, "legato_gap_tolerance": 0.05},
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
    "bachata": {"low_cut": 40, "high_cut": 1000, "onset_threshold": 0.65, "frame_threshold": 0.50, "minimum_note_length": 0.05, "legato_gap_tolerance": 0.03},
    "zouk": {"low_cut": 30, "high_cut": 1000, "onset_threshold": 0.55, "frame_threshold": 0.50, "minimum_note_length": 0.08, "legato_gap_tolerance": 0.06},
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
# K-NEAREST NEIGHBORS AUDIO PROFILING (AUTO-DETECTION)
# ==============================================================================
def auto_detect_profile(bass_wav_path, drums_wav_path):
    y_b, sr_b = librosa.load(str(bass_wav_path), sr=22050, mono=True, res_type='soxr_hq')
    
    if len(y_b) == 0:
        return "none", 120.0, y_b, sr_b

    # Pre-filter the bass audio before analyzing to prevent high-frequency cymbal bleed from skewing metrics
    b_feat, a_feat = butter(2, 800 / (sr_b / 2), btype='low')
    y_b_feat = filtfilt(b_feat, a_feat, y_b)

    # 1. Bass Features: Brightness & ZCR
    cent = librosa.feature.spectral_centroid(y=y_b_feat, sr=sr_b)
    avg_centroid = np.median(cent)
    
    zcr = librosa.feature.zero_crossing_rate(y=y_b_feat)
    avg_zcr = np.median(zcr)

    # 2. Rhythm Features
    if drums_wav_path.exists():
        y_rhythm, sr_rhythm = librosa.load(str(drums_wav_path), sr=22050, mono=True, res_type='soxr_hq')
    else:
        y_rhythm, sr_rhythm = y_b, sr_b
        
    onset_env = librosa.onset.onset_strength(y=y_rhythm, sr=sr_rhythm)
    duration_sec = librosa.get_duration(y=y_rhythm, sr=sr_rhythm)
    
    tempo_est, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr_rhythm)
    bpm = float(np.median(tempo_est)) if tempo_est.size > 0 else 120.0
    
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr_rhythm)
    note_density = len(onsets) / duration_sec if duration_sec > 0 else 0

    academic_profiles = {
        "metal":      {"bpm": (140, 25), "centroid": (600, 250), "density": (4.5, 1.5), "zcr": (0.070, 0.030)},
        "rock":       {"bpm": (120, 20), "centroid": (450, 150), "density": (3.2, 1.0), "zcr": (0.040, 0.020)},
        "jazz":       {"bpm": (110, 30), "centroid": (250, 80),  "density": (2.8, 0.8), "zcr": (0.020, 0.010)},
        "swing":      {"bpm": (130, 30), "centroid": (200, 80),  "density": (3.0, 0.8), "zcr": (0.015, 0.010)},
        "bachata":    {"bpm": (130, 15), "centroid": (180, 50), "density": (4.1, 0.8), "zcr": (0.010, 0.005)},
        "funk":       {"bpm": (105, 15), "centroid": (650, 200), "density": (4.0, 1.2), "zcr": (0.060, 0.020)},
        "hiphop":     {"bpm": (90, 15),  "centroid": (150, 60),  "density": (2.2, 0.8), "zcr": (0.015, 0.010)},
        "reggae":     {"bpm": (75, 15),  "centroid": (130, 40),  "density": (1.8, 0.6), "zcr": (0.012, 0.008)},
        "electronic": {"bpm": (128, 10), "centroid": (350, 150), "density": (3.5, 1.2), "zcr": (0.030, 0.020)},
        "house":      {"bpm": (124, 6),  "centroid": (280, 100), "density": (2.8, 0.7), "zcr": (0.025, 0.015)},
        "disco":      {"bpm": (115, 10), "centroid": (320, 100), "density": (3.2, 0.8), "zcr": (0.030, 0.015)},
        "synthwave":  {"bpm": (110, 15), "centroid": (300, 120), "density": (3.5, 1.0), "zcr": (0.025, 0.015)},
        "rnb":        {"bpm": (85, 15),  "centroid": (200, 80),  "density": (2.4, 0.7), "zcr": (0.018, 0.010)},
        "country":    {"bpm": (105, 20), "centroid": (300, 100), "density": (2.6, 0.7), "zcr": (0.025, 0.015)},
        "punk":       {"bpm": (160, 25), "centroid": (550, 180), "density": (4.8, 1.2), "zcr": (0.060, 0.025)},
        "reggaeton":  {"bpm": (95, 10),  "centroid": (180, 70),  "density": (2.8, 0.6), "zcr": (0.015, 0.010)},
        "afrobeats":  {"bpm": (105, 10), "centroid": (220, 80),  "density": (3.0, 0.7), "zcr": (0.020, 0.010)},
        "zouk":       {"bpm": (90, 10),  "centroid": (250, 100), "density": (3.5, 1.0), "zcr": (0.020, 0.010)},
        "blues":      {"bpm": (80, 20),  "centroid": (280, 90),  "density": (2.2, 0.6), "zcr": (0.025, 0.015)},
        "bossanova":  {"bpm": (130, 20), "centroid": (200, 70),  "density": (2.5, 0.6), "zcr": (0.015, 0.010)},
        "dnb":        {"bpm": (174, 10), "centroid": (400, 150), "density": (4.5, 1.5), "zcr": (0.040, 0.020)},
        "pop":        {"bpm": (115, 20), "centroid": (300, 100), "density": (2.8, 0.8), "zcr": (0.025, 0.015)},
        "ska":        {"bpm": (140, 20), "centroid": (400, 120), "density": (4.2, 1.0), "zcr": (0.040, 0.020)},
        "salsa":      {"bpm": (90, 20),  "centroid": (300, 100), "density": (3.5, 1.0), "zcr": (0.025, 0.015)},
        "classical":  {"bpm": (80, 30),  "centroid": (200, 100), "density": (1.5, 0.8), "zcr": (0.015, 0.010)}
    }

    best_genre = "none"
    shortest_distance = float('inf')

    bpm_variants = [bpm, bpm / 2.0, bpm * 2.0]

    for g, stats in academic_profiles.items():
        for b_variant in bpm_variants:
            if b_variant < 40 or b_variant > 220:
                continue
                
            z_bpm = ((b_variant - stats["bpm"][0]) / stats["bpm"][1]) ** 2
            z_cent = ((avg_centroid - stats["centroid"][0]) / stats["centroid"][1]) ** 2
            z_dens = ((note_density - stats["density"][0]) / stats["density"][1]) ** 2
            z_zcr = ((avg_zcr - stats["zcr"][0]) / stats["zcr"][1]) ** 2
            
            total_distance = np.sqrt(z_bpm + z_cent + z_dens + z_zcr)
            
            if total_distance < shortest_distance:
                shortest_distance = total_distance
                best_genre = g
    
    return best_genre, bpm, y_b, sr_b

def estimate_harmonic_key(y, sr):
    if len(y) == 0 or np.all(y == 0): return "C major"
    
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=12)
    except librosa.util.exceptions.ParameterError:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
    mean_chroma = np.mean(chroma, axis=1)
    
    ks_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    ks_minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    ks_major = (ks_major - np.mean(ks_major)) / np.std(ks_major)
    ks_minor = (ks_minor - np.mean(ks_minor)) / np.std(ks_minor)
    mean_chroma_norm = (mean_chroma - np.mean(mean_chroma)) / (np.std(mean_chroma) + 1e-6)
    
    best_corr = -1.0
    best_key_idx = 0
    is_minor = False
    
    for i in range(12):
        maj_profile = np.roll(ks_major, i)
        min_profile = np.roll(ks_minor, i)
        
        maj_corr = np.corrcoef(mean_chroma_norm, maj_profile)[0, 1]
        min_corr = np.corrcoef(mean_chroma_norm, min_profile)[0, 1]
        
        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key_idx = i
            is_minor = False
            
        if min_corr > best_corr:
            best_corr = min_corr
            best_key_idx = i
            is_minor = True
    
    major_map = {0: 'C', 1: 'D-', 2: 'D', 3: 'E-', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'A-', 9: 'A', 10: 'B-', 11: 'B'}
    minor_map = {0: 'C', 1: 'C#', 2: 'D', 3: 'E-', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'B-', 11: 'B'}

    root_name = minor_map[best_key_idx] if is_minor else major_map[best_key_idx]
    mode = "minor" if is_minor else "major"
    return f"{root_name} {mode}"

def determine_tuning(min_pitch, requested_tuning):
    tunings = {
        "standard": ([43, 38, 33, 28], ["G", "D", "A", "E"]),
        "drop-d": ([43, 38, 33, 26], ["G", "D", "A", "D"]),
        "5-string": ([43, 38, 33, 28, 23], ["G", "D", "A", "E", "B"]),
        "6-string": ([48, 43, 38, 33, 28, 23], ["C", "G", "D", "A", "E", "B"])
    }
    
    if requested_tuning in tunings:
        return tunings[requested_tuning]
        
    if min_pitch < 28 and min_pitch >= 23:
        return tunings["5-string"]
    elif min_pitch < 23:
        return tunings["6-string"]
    return tunings["standard"]

def apply_viterbi_fretboard(notes, baselines, string_names):
    if not notes: return []
    
    states = []
    for s_idx, base in enumerate(baselines):
        for f in range(0, 25):
            states.append((s_idx, f, base + f))
            
    path_data = []
    
    for i, n in enumerate(notes):
        pitch_val = int(round(n.pitch))
        valid_states = [s for s in states if s[2] == pitch_val]
        
        if not valid_states:
            valid_states = [(0, max(0, pitch_val - baselines[0]), pitch_val)]
            
        step_paths = {}
        if i == 0:
            for vs in valid_states:
                cost = abs(vs[1] - 5)
                step_paths[vs] = (cost, [vs])
        else:
            prev_paths = path_data[-1]
            for vs in valid_states:
                best_cost = float('inf')
                best_hist = []
                
                for ps, (prev_cost, prev_hist) in prev_paths.items():
                    if vs[1] == 0:
                        fret_cost = 0.5
                    elif ps[1] == 0:
                        fret_cost = 1.0 + (vs[1] * 0.5)
                    else:
                        span = abs(vs[1] - ps[1])
                        if span <= 4:
                            fret_cost = span * 0.5
                        else:
                            fret_cost = 2.0 + ((span - 4) ** 2)
                            
                    if vs[1] > 7 and vs[0] < 2:
                        fret_cost += (vs[1] - 7) * 1.5
                            
                    string_cost = abs(vs[0] - ps[0]) * 1.5
                    move_cost = fret_cost + string_cost
                    
                    total_cost = prev_cost + move_cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_hist = prev_hist + [vs]
                
                step_paths[vs] = (best_cost, best_hist)
        path_data.append(step_paths)
        
    best_final_state = min(path_data[-1].items(), key=lambda x: x[1][0])
    optimal_path = best_final_state[1][1]
    
    return [(string_names[s[0]], s[1]) for s in optimal_path]

def extract_timbre(y_b, sr_b, start_time, end_time):
    start_samp = int(start_time * sr_b)
    end_samp = int(end_time * sr_b)
    if end_samp <= start_samp: return "normal"
    
    segment = y_b[start_samp:end_samp]
    if len(segment) < 512: return "normal"
    
    rms = np.mean(librosa.feature.rms(y=segment))
    if rms < 0.01: return "mute"
    
    S = np.abs(librosa.stft(segment))
    if S.shape[1] > 1:
        flux = np.mean(librosa.onset.onset_strength(S=librosa.amplitude_to_db(S, ref=np.max), sr=sr_b))
    else:
        flux = 0
        
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr_b, roll_percent=0.85))
    
    if flux > 3.0 and rolloff > 3500: return "pop"
    if flux > 1.5 and rolloff > 2000: return "slap"
    return "normal"

def spell_pitch(midi_val, detected_key):
    sharp_keys = ['G major', 'D major', 'A major', 'E major', 'B major', 'F# major', 'C# major',
                  'E minor', 'B minor', 'F# minor', 'C# minor', 'G# minor', 'D# minor', 'A# minor']
    flat_keys = ['F major', 'B- major', 'E- major', 'A- major', 'D- major', 'G- major', 'C- major',
                 'D minor', 'G minor', 'C minor', 'F minor', 'B- minor', 'E- minor', 'A- minor']
                 
    p = pitch.Pitch(midi=int(round(midi_val)))
    if detected_key in flat_keys:
        if '#' in p.name:
            p = p.getEnharmonic()
    elif detected_key in sharp_keys:
        if '-' in p.name:
            p = p.getEnharmonic()
    return p

def extract_chords(y, sr, bpm):
    if len(y) == 0: return []
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=12)
    except:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
    maj_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    min_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    dom7_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    
    templates = []
    labels = []
    notes = ['C', 'C#', 'D', 'E-', 'E', 'F', 'F#', 'G', 'A-', 'A', 'B-', 'B']
    
    for i in range(12):
        templates.append(np.roll(maj_template, i))
        labels.append(notes[i])
        templates.append(np.roll(min_template, i))
        labels.append(notes[i] + 'm')
        templates.append(np.roll(dom7_template, i))
        labels.append(notes[i] + '7')
        
    templates = np.array(templates)
    
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, bpm=bpm)
    chords = []
    
    for i in range(0, len(beat_frames), 4):
        start_frame = beat_frames[i]
        end_frame = beat_frames[i+4] if i+4 < len(beat_frames) else chroma.shape[1]
        if start_frame >= end_frame: continue
        
        segment = chroma[:, start_frame:end_frame]
        mean_chroma = np.mean(segment, axis=1)
        mean_chroma = (mean_chroma - np.mean(mean_chroma)) / (np.std(mean_chroma) + 1e-6)
        
        corrs = np.dot(templates, mean_chroma)
        best_idx = np.argmax(corrs)
        chords.append((i, labels[best_idx]))
        
    return chords

def process_score_tier(raw_notes, pitch_bends, level, genre, bpm, detected_key, tuning, y_b, sr_b):
    cfg = get_config(genre)
    sc = stream.Score()
    sc.insert(0, metadata.Metadata(title=f"Bass Transcription [{level.upper()}]"))
    sc.insert(0, tempo.MetronomeMark(number=bpm))
    
    k_tonic, k_mode = detected_key.split()
    music_key = key.Key(k_tonic, k_mode)
    sc.insert(0, music_key)
    
    if genre in ['jazz', 'blues', 'hiphop', 'reggae', 'swing', 'zouk']:
        feel_txt = expressions.TextExpression("Swing / Laid back")
        feel_txt.placement = 'above'
        sc.insert(0, feel_txt)
    elif genre in ['funk', 'disco', 'slap']:
        feel_txt = expressions.TextExpression("16th note groove")
        feel_txt.placement = 'above'
        sc.insert(0, feel_txt)
    elif genre in ['rock', 'metal', 'punk']:
        feel_txt = expressions.TextExpression("Driving / Straight")
        feel_txt.placement = 'above'
        sc.insert(0, feel_txt)

    part = stream.Part()
    part.insert(0, clef.BassClef())

    filtered_events = []
    
    if len(raw_notes) > 0:
        avg_vel = np.median([ev.velocity for ev in raw_notes])
        vel_threshold = max(20, avg_vel * 0.45)
        ghost_threshold = max(30, avg_vel * 0.70)
    else:
        vel_threshold = 45
        ghost_threshold = 50
    
    for ev in raw_notes:
        if ev.pitch < 15 or ev.pitch > 72: continue
        
        if level == "simplex":
            beat_loc = ev.start * (bpm / 60.0)
            if abs(beat_loc % 1) > 0.35:
                continue
        elif level == "normal":
            if ev.velocity < vel_threshold:
                continue
        filtered_events.append(ev)

    if not filtered_events: return None

    grouped_events = []
    filtered_events.sort(key=lambda x: x.start)
    
    current_group = [filtered_events[0]]
    for ev in filtered_events[1:]:
        if ev.start - current_group[0].start < 0.03:
            current_group.append(ev)
        else:
            grouped_events.append(current_group)
            current_group = [ev]
    if current_group: grouped_events.append(current_group)

    resolved_events = []
    for group in grouped_events:
        group.sort(key=lambda x: x.pitch)
        resolved_events.append([group[0]])
    part = stream.Part(id=f"Bass_{level}")
    ts = meter.TimeSignature('4/4')
    part.insert(0, ts)
    
    chord_symbols = extract_chords(y_b, sr_b, bpm)
    for q_offset, chord_name in chord_symbols:
        ch = harmony.ChordSymbol(chord_name)
        part.insert(q_offset, ch)

    part.insert(0, clef.BassClef())

    min_pitch = min([n.pitch for n in raw_notes])
    baselines, string_names = determine_tuning(min_pitch, tuning)
    
    viterbi_roots = [grp[0] for grp in resolved_events]
    optimal_fingerings = apply_viterbi_fretboard(viterbi_roots, baselines, string_names)

    if level == "simplex":
        factor = 2
        min_len = 1.0
    elif level == "normal":
        if genre in ['jazz', 'blues', 'reggae', 'swing', 'zouk']:
            factor = 3
        else:
            factor = 4
        min_len = 0.25
    else:
        if genre in ['jazz', 'blues', 'reggae', 'swing', 'zouk']:
            factor = 6
        elif genre in ['rock', 'pop', 'funk', 'disco', 'electronic', 'metal', 'punk']:
            factor = 4
        else:
            factor = 12
        min_len = 0.25

    quantized_events = []
    for idx, event_group in enumerate(resolved_events):
        base_ev = event_group[0]
        
        raw_start_beat = base_ev.start * (bpm / 60.0)
        raw_end_beat = base_ev.end * (bpm / 60.0)
        
        if level == "simplex":
            q_start = round(raw_start_beat * factor) / factor
            q_end = q_start + 2.0
        else:
            q_start = round(raw_start_beat * factor) / factor
            q_end = round(raw_end_beat * factor) / factor
        
        if q_end <= q_start: q_end = q_start + min_len
            
        quantized_events.append({
            "start": q_start, "end": q_end,
            "group": event_group, "base": base_ev,
            "fingering": optimal_fingerings[idx]
        })

    for i in range(len(quantized_events) - 1):
        if quantized_events[i+1]["start"] - quantized_events[i]["start"] < min_len:
            quantized_events[i+1]["start"] = quantized_events[i]["start"] + min_len
            
        if quantized_events[i]["end"] > quantized_events[i+1]["start"]:
            quantized_events[i]["end"] = quantized_events[i+1]["start"]
            
        gap = quantized_events[i+1]["start"] - quantized_events[i]["end"]
        orig_len = quantized_events[i]["end"] - quantized_events[i]["start"]
        
        quantized_events[i]["staccato"] = False
        if 0 < gap <= 0.5:
            quantized_events[i]["end"] = quantized_events[i+1]["start"]
            if orig_len <= 0.25 and gap >= 0.15:
                quantized_events[i]["staccato"] = True

    current_dyn_str = None

    for idx, q_ev in enumerate(quantized_events):
        event_group = q_ev["group"]
        base_ev = q_ev["base"]
        s_name, f_val = q_ev["fingering"]

        start_num = int(round(q_ev["start"] * factor))
        dur_num = int(round((q_ev["end"] - q_ev["start"]) * factor))
        if dur_num <= 0: dur_num = 1

        offset_val = Fraction(start_num, factor)
        q_length = Fraction(dur_num, factor)

        is_staccato = q_ev.get("staccato", False)
        if q_length == Fraction(3, 12) and level != "simplex":
            if idx < len(quantized_events) - 1:
                gap = quantized_events[idx+1]["start"] - q_ev["end"]
                if gap >= 0.25:
                    q_length = Fraction(6, 12)
                    is_staccato = True

        is_grace = False
        if (q_ev["end"] - q_ev["start"]) < 0.08 and idx < len(quantized_events) - 1:
            next_q_ev = quantized_events[idx+1]
            gap_to_next = next_q_ev["start"] - q_ev["end"]
            if gap_to_next < 0.05:
                is_grace = True

        if len(event_group) == 1:
            p = spell_pitch(base_ev.pitch, detected_key)
            if is_grace:
                music_element = note.GraceNote(p)
            else:
                music_element = note.Note(p)
        else:
            pitches = [spell_pitch(e.pitch, detected_key) for e in event_group]
            music_element = chord.Chord(pitches)

        if not is_grace:
            music_element.duration.quarterLength = q_length
        
        if is_staccato:
            music_element.articulations.append(articulations.Staccato())

        timbre = extract_timbre(y_b, sr_b, base_ev.start, base_ev.end)
        
        if base_ev.velocity < ghost_threshold or timbre == "mute":
            if music_element.isChord:
                for n in music_element.notes:
                    n.notehead = 'x'
            else:
                music_element.notehead = 'x'
        
        if timbre == "pop":
            music_element.addLyric('P')
        elif timbre == "slap":
            music_element.addLyric('T')

        string_map = {"G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "C": 6}
        str_num = string_map.get(s_name, 4)
        
        music_element.articulations.append(articulations.StringIndication(str_num))
        music_element.articulations.append(articulations.FretIndication(f_val))

        vel = base_ev.velocity
        if vel < 40: dyn_str = 'p'
        elif vel < 65: dyn_str = 'mp'
        elif vel < 85: dyn_str = 'mf'
        elif vel < 105: dyn_str = 'f'
        else: dyn_str = 'ff'
        
        if dyn_str != current_dyn_str and (current_dyn_str is None or idx % 4 == 0):
            dyn_obj = dynamics.Dynamic(dyn_str)
            part.insert(offset_val, dyn_obj)
            current_dyn_str = dyn_str

        part.insert(offset_val, music_element)
        q_ev["music_element"] = music_element

    for idx, q_ev in enumerate(quantized_events):
        el1 = q_ev.get("music_element")
        if not el1: continue
        
        bends_in_note = [pb for pb in pitch_bends if q_ev["start"] <= pb.time <= q_ev["end"]]
        has_slide = False
        has_vibrato = False
        if bends_in_note:
            max_bend = max([pb.pitch for pb in bends_in_note])
            min_bend = min([pb.pitch for pb in bends_in_note])
            if max_bend > 1000 or min_bend < -1000:
                has_slide = True
            
            if (q_ev["end"] - q_ev["start"]) > 0.4 and (max_bend - min_bend) > 200:
                bend_vals = [pb.pitch for pb in bends_in_note]
                diffs = [bend_vals[i] - bend_vals[i-1] for i in range(1, len(bend_vals))]
                crossings = sum(1 for i in range(1, len(diffs)) if diffs[i-1] * diffs[i] < 0)
                if crossings > 3:
                    has_vibrato = True

        if has_vibrato:
            el1.articulations.append(articulations.Doit())
            el1.addLyric("vib.")
                
        if idx < len(quantized_events) - 1:
            next_q_ev = quantized_events[idx+1]
            gap = next_q_ev["start"] - q_ev["end"]
            
            s_name = q_ev["fingering"][0]
            next_s_name = next_q_ev["fingering"][0]
            
            if gap <= 0.15 and s_name == next_s_name and q_ev["base"].pitch != next_q_ev["base"].pitch:
                el2 = next_q_ev.get("music_element")
                if el2:
                    if has_slide:
                        gl = spanner.Glissando([el1, el2])
                        part.insert(0, gl)
                    else:
                        sl = spanner.Slur([el1, el2])
                        part.insert(0, sl)

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
    genre_override = sys.argv[3].lower() if len(sys.argv) > 3 else "auto"
    out_dir = Path("./output_bass").resolve()
    project_name = stems_dir.name.replace('stems_', '')

    bass_wav = stems_dir / "bass.wav"
    drums_wav = stems_dir / "drums.wav"

    if not bass_wav.exists():
        print(f"Error: bass.wav missing from {stems_dir}")
        sys.exit(1)

    if genre_override != "auto":
        auto_genre, bpm, y_b, sr_b = auto_detect_profile(bass_wav, drums_wav)
        genre = genre_override
    else:
        genre, bpm, y_b, sr_b = auto_detect_profile(bass_wav, drums_wav)
    
    cfg = get_config(genre)
    detected_key = estimate_harmonic_key(y_b, sr_b)

    nyq = 0.5 * sr_b
    low_hz = max(1.0, cfg["low_cut"])
    high_hz = min(nyq - 1.0, cfg["high_cut"])
    b, a = butter(4, [low_hz / nyq, high_hz / nyq], btype='band')
    y_b_filtered = filtfilt(b, a, y_b)

    if np.max(np.abs(y_b_filtered)) > 0:
        y_b_filtered = librosa.util.normalize(y_b_filtered)

    cond_wav = out_dir / "temp_conditioned_bass.wav"
    sf.write(str(cond_wav), y_b_filtered, int(sr_b))
    
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _, bass_midi, _ = bp_predict(
                audio_path=str(cond_wav),
                onset_threshold=cfg["onset_threshold"],
                frame_threshold=cfg["frame_threshold"],
                minimum_note_length=cfg["minimum_note_length"]
            )
            
        if not bass_midi.instruments:
            sys.exit(0)
            
        raw_notes = bass_midi.instruments[0].notes
        pitch_bends = bass_midi.instruments[0].pitch_bends
        
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)
    finally:
        if cond_wav.exists(): cond_wav.unlink()

    if len(raw_notes) == 0:
        sys.exit(1)

    for lvl in ["simplex", "normal", "complex"]:
        score_obj = process_score_tier(raw_notes, pitch_bends, lvl, genre, bpm, detected_key, tuning_pref, y_b, sr_b)
        if score_obj:
            target_output = out_dir / f"{project_name}_bass_{lvl}_{genre}.musicxml"
            score_obj.write('musicxml', fp=str(target_output))

    print(f"Success: Files written to {out_dir}/")

if __name__ == "__main__":
    main()
EOF

for STEMS_DIR in "${STEMS_DIRS[@]}"; do
    if [ ! -d "$STEMS_DIR" ]; then
        echo "Warning: Directory not found -> $STEMS_DIR. Skipping..."
        continue
    fi
    echo "Processing $STEMS_DIR..."
    "$ENV_DIR/bin/python" run_engine_bass.py "$STEMS_DIR" "$TUNING" "$GENRE_OVERRIDE"
done
