#!/usr/bin/env bash
# ==============================================================================
# Bass Transcription Pipeline (M1 Optimized) - Articulation & Ergonomics Mod
# Usage: ./process.sh <path_to_stems_folder> [--tuning <tuning_type>] [--genre <genre_name>]
# ==============================================================================

set -euo pipefail

STEMS_DIRS=()
TUNING="auto"
GENRE_OVERRIDE="auto"

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
    exit 1
fi

OUT_DIR="./output_bass"
ENV_DIR=".venv_bass"
mkdir -p "$OUT_DIR"

if [ ! -d "$ENV_DIR" ]; then
    $PY_CMD -m venv "$ENV_DIR" > /dev/null 2>&1
fi
source "$ENV_DIR/bin/activate"

"$ENV_DIR/bin/python" -m pip install --upgrade pip wheel > /dev/null 2>&1
"$ENV_DIR/bin/python" -m pip install "setuptools<82" > /dev/null 2>&1

OS_NAME=$(uname -s)
ARCH_NAME=$(uname -m)

if [ "$OS_NAME" = "Darwin" ] && [ "$ARCH_NAME" = "arm64" ]; then
    "$ENV_DIR/bin/python" -m pip install --no-cache-dir "tensorflow-macos<2.16.0" "tensorflow-metal==1.1.0" > /dev/null 2>&1
else
    "$ENV_DIR/bin/python" -m pip install --no-cache-dir "tensorflow<2.16.0" > /dev/null 2>&1
fi

"$ENV_DIR/bin/python" -m pip install --no-cache-dir \
    "numpy==1.26.4" "scipy==1.14.1" "soundfile==0.12.1" "soxr==0.3.7" \
    "librosa>=0.10.2" "music21==9.1.0" "pretty_midi==0.2.10" \
    "basic-pitch>=0.4.0" "resampy==0.4.2" > /dev/null 2>&1

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
import music21
from pathlib import Path
from scipy.signal import butter, filtfilt
from basic_pitch.inference import predict as bp_predict
from music21 import instrument, clef, metadata, tempo, stream, note, chord, meter, key, articulations, pitch, spanner, expressions, harmony, dynamics

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

def auto_detect_profile(bass_wav_path, drums_wav_path):
    y_b, sr_b = librosa.load(str(bass_wav_path), sr=22050, mono=True, res_type='soxr_hq')
    if len(y_b) == 0: return "none", 120.0, y_b, sr_b

    b_feat, a_feat = butter(2, 800 / (sr_b / 2), btype='low')
    y_b_feat = filtfilt(b_feat, a_feat, y_b)

    cent = librosa.feature.spectral_centroid(y=y_b_feat, sr=sr_b)
    avg_centroid = np.median(cent)
    
    zcr = librosa.feature.zero_crossing_rate(y=y_b_feat)
    avg_zcr = np.median(zcr)

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
        "bachata":    {"bpm": (130, 15), "centroid": (180, 50),  "density": (4.1, 0.8), "zcr": (0.010, 0.005)},
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

    best_genre, shortest_distance = "none", float('inf')
    bpm_variants = [bpm, bpm / 2.0, bpm * 2.0]

    for g, stats in academic_profiles.items():
        for b_variant in bpm_variants:
            if b_variant < 40 or b_variant > 220: continue
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
    try: chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=12)
    except: chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)
    
    ks_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    ks_minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    ks_major = (ks_major - np.mean(ks_major)) / np.std(ks_major)
    ks_minor = (ks_minor - np.mean(ks_minor)) / np.std(ks_minor)
    mean_chroma_norm = (mean_chroma - np.mean(mean_chroma)) / (np.std(mean_chroma) + 1e-6)
    
    best_corr, best_key_idx, is_minor = -1.0, 0, False
    
    for i in range(12):
        maj_corr = np.corrcoef(mean_chroma_norm, np.roll(ks_major, i))[0, 1]
        min_corr = np.corrcoef(mean_chroma_norm, np.roll(ks_minor, i))[0, 1]
        
        if maj_corr > best_corr: best_corr, best_key_idx, is_minor = maj_corr, i, False
        if min_corr > best_corr: best_corr, best_key_idx, is_minor = min_corr, i, True
    
    major_map = {0: 'C', 1: 'D-', 2: 'D', 3: 'E-', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'A-', 9: 'A', 10: 'B-', 11: 'B'}
    minor_map = {0: 'C', 1: 'C#', 2: 'D', 3: 'E-', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'B-', 11: 'B'}
    return f"{minor_map[best_key_idx] if is_minor else major_map[best_key_idx]} {'minor' if is_minor else 'major'}"

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
    states = [(s_idx, f, base + f) for s_idx, base in enumerate(baselines) for f in range(0, 21)]
    path_data = []
    
    for i, n in enumerate(notes):
        pitch_val = int(round(n.pitch))
        valid_states = [s for s in states if s[2] == pitch_val]
        if not valid_states: valid_states = [(0, max(0, pitch_val - baselines[0]), pitch_val)]
            
        step_paths = {}
        if i == 0:
            for vs in valid_states:
                pref_cost = abs(vs[1] - 5) * 0.2
                if vs[1] > 12: pref_cost += (vs[1] - 12) * 1.0
                step_paths[vs] = (pref_cost, [vs])
        else:
            prev_paths = path_data[-1]
            for vs in valid_states:
                best_cost, best_hist = float('inf'), []
                
                for ps, (prev_cost, prev_hist) in prev_paths.items():
                    span = abs(vs[1] - ps[1])
                    string_diff = abs(vs[0] - ps[0])
                    
                    if vs[1] == 0:
                        fret_cost = 0.5
                    elif span <= 4:
                        fret_cost = span * 0.1
                    else:
                        fret_cost = 1.0 + ((span - 4) ** 1.8)
                            
                    if vs[1] > 12: fret_cost += (vs[1] - 12) * 0.5
                    
                    string_cost = string_diff * 0.3
                    total_cost = prev_cost + fret_cost + string_cost
                    
                    if total_cost < best_cost:
                        best_cost, best_hist = total_cost, prev_hist + [vs]
                
                step_paths[vs] = (best_cost, best_hist)
        path_data.append(step_paths)
        
    best_final_state = min(path_data[-1].items(), key=lambda x: x[1][0])
    return [(string_names[s[0]], s[1]) for s in best_final_state[1][1]]

def extract_timbre(y_b, sr_b, start_time, end_time):
    start_samp, end_samp = int(start_time * sr_b), int(end_time * sr_b)
    if end_samp <= start_samp: return "normal"
    segment = y_b[start_samp:end_samp]
    if len(segment) < 512: return "normal"
    
    if np.mean(librosa.feature.rms(y=segment)) < 0.01: return "mute"
    
    S = np.abs(librosa.stft(segment))
    flux = np.mean(librosa.onset.onset_strength(S=librosa.amplitude_to_db(S, ref=np.max), sr=sr_b)) if S.shape[1] > 1 else 0
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr_b, roll_percent=0.85))
    
    if flux > 3.0 and rolloff > 3500: return "pop"
    if flux > 1.5 and rolloff > 2000: return "slap"
    return "normal"

def spell_pitch(midi_val, detected_key):
    sharp_keys = ['G major', 'D major', 'A major', 'E major', 'B major', 'F# major', 'C# major', 'E minor', 'B minor', 'F# minor', 'C# minor', 'G# minor', 'D# minor', 'A# minor']
    flat_keys = ['F major', 'B- major', 'E- major', 'A- major', 'D- major', 'G- major', 'C- major', 'D minor', 'G minor', 'C minor', 'F minor', 'B- minor', 'E- minor', 'A- minor']
    p = pitch.Pitch(midi=int(round(midi_val)))
    if detected_key in flat_keys and '#' in p.name: p = p.getEnharmonic()
    elif detected_key in sharp_keys and '-' in p.name: p = p.getEnharmonic()
    return p

def extract_chords(y, sr, bpm):
    if len(y) == 0: return []
    try: chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=12)
    except: chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
    maj_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    min_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    dom7_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    
    templates, labels = [], []
    notes = ['C', 'C#', 'D', 'E-', 'E', 'F', 'F#', 'G', 'A-', 'A', 'B-', 'B']
    
    for i in range(12):
        templates.append(np.roll(maj_template, i)); labels.append(notes[i])
        templates.append(np.roll(min_template, i)); labels.append(notes[i] + 'm')
        templates.append(np.roll(dom7_template, i)); labels.append(notes[i] + '7')
        
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
        
        best_idx = np.argmax(np.dot(templates, mean_chroma))
        chords.append((i, labels[best_idx]))
        
    return chords

def correct_octave_errors(raw_notes, y, sr):
    corrected = []
    for n in raw_notes:
        start_s, end_s = int(n.start * sr), int(n.end * sr)
        if end_s - start_s > 2048:
            segment = y[start_s:end_s]
            f0 = librosa.yin(segment, fmin=30, fmax=150, sr=sr)
            median_f0 = np.nanmedian(f0)
            if median_f0 > 0:
                est_midi = librosa.hz_to_midi(median_f0)
                if n.pitch - est_midi > 8: n.pitch -= 12
        corrected.append(n)
    return corrected

def process_score_tier(raw_notes, pitch_bends, level, genre, bpm, detected_key, tuning, y_b, sr_b):
    cfg = get_config(genre)
    sc = stream.Score()
    lvl_label = level.lower()
    
    sc.insert(0, metadata.Metadata(title=f"Bass Transcription [{lvl_label.upper()}]"))
    sc.insert(0, tempo.MetronomeMark(number=round(bpm)))
    
    k_tonic, k_mode = detected_key.split()
    sc.insert(0, key.Key(k_tonic, k_mode))
    
    if genre in ['jazz', 'blues', 'hiphop', 'reggae', 'swing', 'zouk']:
        txt = expressions.TextExpression("Swing / Laid back")
        txt.placement = 'above'; sc.insert(0, txt)
    elif genre in ['funk', 'disco', 'slap']:
        txt = expressions.TextExpression("16th note groove")
        txt.placement = 'above'; sc.insert(0, txt)
    elif genre in ['rock', 'metal', 'punk']:
        txt = expressions.TextExpression("Driving / Straight")
        txt.placement = 'above'; sc.insert(0, txt)

    part = stream.Part(id=f"Bass_{lvl_label}")
    part.insert(0, meter.TimeSignature('4/4'))
    part.insert(0, clef.BassClef())

    if len(raw_notes) > 0:
        avg_vel = np.median([ev.velocity for ev in raw_notes])
        vel_threshold = max(20, avg_vel * 0.40)
        ghost_threshold = max(20, avg_vel * 0.35)
    else:
        vel_threshold, ghost_threshold = 30, 25

    chord_symbols = extract_chords(y_b, sr_b, bpm)
    chord_roots = [pitch.Pitch(c[1].replace('m', '').replace('7', '')).pitchClass for c in chord_symbols]

    sanitized_raw_notes = []
    for ev in raw_notes:
        if ev.pitch < 15 or ev.pitch > 72:
            continue
            
        duration = ev.end - ev.start
        if duration < 0.05 and ev.velocity < (avg_vel * 0.75):
            continue
            
        if level == "easy":
            beat_loc = ev.start * (bpm / 60.0)
            is_root = (ev.pitch % 12) in chord_roots
            if not is_root and abs(beat_loc % 1) > 0.25 and ev.velocity < vel_threshold * 1.5:
                continue
        elif level == "normal":
            if ev.velocity < vel_threshold:
                continue
                
        sanitized_raw_notes.append(ev)

    if not sanitized_raw_notes:
        return None

    grid_subdiv = 3 if genre in ['jazz', 'blues', 'swing', 'zouk'] else 4
    if level == "easy":
        grid_subdiv = 2
    min_q_len = Fraction(1, grid_subdiv)

    raw_quantized = []
    for ev in sanitized_raw_notes:
        raw_start_beat = ev.start * (bpm / 60.0)
        raw_end_beat = ev.end * (bpm / 60.0)

        q_start = Fraction(int(round(raw_start_beat * grid_subdiv)), grid_subdiv)
        q_end = Fraction(int(round(raw_end_beat * grid_subdiv)), grid_subdiv)
        if q_end <= q_start:
            q_end = q_start + min_q_len

        raw_quantized.append({
            "start": q_start,
            "end": q_end,
            "raw_start": ev.start,
            "raw_end": ev.end,
            "base": ev
        })

    raw_quantized.sort(key=lambda x: x["start"])
    grid_groups = {}
    for q_ev in raw_quantized:
        grid_groups.setdefault(q_ev["start"], []).append(q_ev)

    # -------------------------------------------------------------------------
    # Strict Monophony Selection Pass (Stops vertical chords & stacked notes)
    # -------------------------------------------------------------------------
    quantized_events = []
    for start_tick in sorted(grid_groups.keys()):
        group = grid_groups[start_tick]
        group.sort(key=lambda x: x["base"].velocity, reverse=True)
        best_ev = group[0]
        best_ev["group"] = [best_ev["base"]]
        best_ev["staccato"] = False
        quantized_events.append(best_ev)

    # -------------------------------------------------------------------------
    # Chronological Flattening Pass (Stops timeline overlap & multi-voice rests)
    # -------------------------------------------------------------------------
    final_sanitized = []
    for q_ev in quantized_events:
        if not final_sanitized:
            final_sanitized.append(q_ev)
            continue
        
        prev_ev = final_sanitized[-1]
        
        if q_ev["start"] < prev_ev["end"]:
            if q_ev["start"] <= prev_ev["start"]:
                continue
            else:
                prev_ev["end"] = q_ev["start"]
                if prev_ev["end"] <= prev_ev["start"]:
                    prev_ev["end"] = prev_ev["start"] + min_q_len
                    q_ev["start"] = prev_ev["end"]
                    
        if q_ev["end"] <= q_ev["start"]:
            q_ev["end"] = q_ev["start"] + min_q_len
            
        final_sanitized.append(q_ev)
        
    quantized_events = final_sanitized

    # -------------------------------------------------------------------------
    # RHYTHMIC SMOOTHING & GAP CLOSING LAYER (Heals Disco/Pop Fragmentation)
    # -------------------------------------------------------------------------
    cleaned_quantized_events = []
    for idx, q_ev in enumerate(quantized_events):
        note_duration_sec = q_ev["raw_end"] - q_ev["raw_start"]
        if note_duration_sec < 0.05 and idx < len(quantized_events) - 1:
            continue

        if cleaned_quantized_events:
            prev_clean_ev = cleaned_quantized_events[-1]
            gap_beats = q_ev["start"] - prev_clean_ev["end"]
            
            if 0 < gap_beats <= Fraction(3, 4):
                prev_clean_ev["end"] = q_ev["start"]
                
        cleaned_quantized_events.append(q_ev)
    quantized_events = cleaned_quantized_events

    if quantized_events:
        min_pitch = min([q["base"].pitch for q in quantized_events])
        baselines, string_names = determine_tuning(min_pitch, tuning)
        optimal_fingerings = apply_viterbi_fretboard([q["base"] for q in quantized_events], baselines, string_names)
        for idx, q_ev in enumerate(quantized_events):
            q_ev["fingering"] = optimal_fingerings[idx]

    # Context-Aware Legato Adjustment (Safely heals rounding anomalies within profiles)
    gap_tolerance_beats = Fraction(int(round(cfg["legato_gap_tolerance"] * (bpm / 60.0) * grid_subdiv)), grid_subdiv)
    for i in range(len(quantized_events) - 1):
        curr_ev = quantized_events[i]
        next_ev = quantized_events[i+1]
        gap = next_ev["start"] - curr_ev["end"]

        if 0 <= gap <= max(min_q_len, gap_tolerance_beats):
            curr_ev["end"] = next_ev["start"]

    for q_offset, chord_name in chord_symbols:
        part.insert(Fraction(q_offset, 1), harmony.ChordSymbol(chord_name))

    dyn_windows = {}
    for q_ev in quantized_events:
        meas_idx = int(q_ev["start"] // 8)
        if meas_idx not in dyn_windows:
            dyn_windows[meas_idx] = []
        dyn_windows[meas_idx].append(q_ev["base"].velocity)

    window_dynamics = {}
    last_dyn = None
    for m_idx in sorted(dyn_windows.keys()):
        avg_v = np.mean(dyn_windows[m_idx])
        d_str = 'p' if avg_v < 45 else ('mp' if avg_v < 65 else ('mf' if avg_v < 85 else ('f' if avg_v < 105 else 'ff')))
        if d_str != last_dyn:
            window_dynamics[Fraction(m_idx * 8, 1)] = d_str
            last_dyn = d_str

    for idx, q_ev in enumerate(quantized_events):
        event_group, base_ev = q_ev["group"], q_ev["base"]
        s_name, f_val = q_ev["fingering"]

        offset_val = q_ev["start"]
        q_length = q_ev["end"] - q_ev["start"]

        p = spell_pitch(base_ev.pitch, detected_key)
        music_element = note.Note(p)
        safe_q_length = max(min_q_len, q_length)
        
        music_element.duration.quarterLength = safe_q_length
        music_element.duration.type = music21.duration.quarterLengthToClosestType(safe_q_length)
        
        if q_ev["staccato"]:
            music_element.articulations.append(articulations.Staccato())

        timbre = extract_timbre(y_b, sr_b, base_ev.start, base_ev.end)

        if (base_ev.velocity < ghost_threshold and (base_ev.end - base_ev.start) < 0.08) or timbre == "mute":
            music_element.notehead = 'x'

        if timbre == "pop": music_element.addLyric('P')
        elif timbre == "slap": music_element.addLyric('T')

        string_num = {"G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "C": 6}.get(s_name, 4)
        music_element.articulations.append(articulations.StringIndication(string_num))
        music_element.articulations.append(articulations.FretIndication(f_val))

        if offset_val in window_dynamics:
            part.insert(offset_val, dynamics.Dynamic(window_dynamics[offset_val]))

        part.insert(offset_val, music_element)
        q_ev["music_element"] = music_element

    if len(part.flatten().notesAndRests) > 0:
        part.makeRests(fillGaps=True, inPlace=True)
        part = part.makeMeasures()

        # -------------------------------------------------------------------------
        # Deferred Spanner Engine (Executes safely inside established measures)
        # -------------------------------------------------------------------------
        for idx, q_ev in enumerate(quantized_events):
            el1 = q_ev.get("music_element")
            if not el1:
                continue

            bends_in_note = [pb for pb in pitch_bends if q_ev["raw_start"] <= pb.time <= q_ev["raw_end"]]
            has_slide, has_vibrato = False, False
            if bends_in_note:
                max_b, min_b = max([pb.pitch for pb in bends_in_note]), min([pb.pitch for pb in bends_in_note])
                if max_b > 1000 or min_b < -1000: has_slide = True
                if (q_ev["raw_end"] - q_ev["raw_start"]) > 0.4 and (max_b - min_b) > 200:
                    bend_vals = [pb.pitch for pb in bends_in_note]
                    diffs = [bend_vals[i] - bend_vals[i-1] for i in range(1, len(bend_vals))]
                    if sum(1 for i in range(1, len(diffs)) if diffs[i-1] * diffs[i] < 0) > 3:
                        has_vibrato = True

            if has_vibrato:
                el1.articulations.append(articulations.Doit())
                el1.addLyric("vib.")

            if idx < len(quantized_events) - 1:
                next_q_ev = quantized_events[idx+1]
                if (next_q_ev["raw_start"] - q_ev["raw_end"]) <= 0.15 and q_ev["fingering"][0] == next_q_ev["fingering"][0] and q_ev["base"].pitch != next_q_ev["base"].pitch:
                    el2 = next_q_ev.get("music_element")
                    if el2:
                        try:
                            part.insert(el1.offset, spanner.Glissando([el1, el2]) if has_slide else spanner.Slur([el1, el2]))
                        except Exception:
                            pass

        for el in part.flatten().notesAndRests:
            if not el.duration.type or el.duration.type == 'unrepresentable':
                if el.duration.quarterLength == 0:
                    el.duration.type = 'zero'
                else:
                    el.duration.type = music21.duration.quarterLengthToClosestType(el.duration.quarterLength)

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
        sys.exit(1)

    genre, bpm, y_b, sr_b = auto_detect_profile(bass_wav, drums_wav) if genre_override == "auto" else (genre_override, *auto_detect_profile(bass_wav, drums_wav)[1:])
    
    print(f"Processing: {project_name} | Genre: {genre}")
    
    cfg = get_config(genre)
    detected_key = estimate_harmonic_key(y_b, sr_b)

    nyq = 0.5 * sr_b
    b, a = butter(6, 1200 / nyq, btype='low')
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
        
        raw_notes = correct_octave_errors(raw_notes, y_b, sr_b)
        
    except Exception:
        sys.exit(1)
    finally:
        if cond_wav.exists():
            cond_wav.unlink()

    if len(raw_notes) == 0:
        sys.exit(1)

    for lvl in ["easy", "normal", "advanced"]:
        score_obj = process_score_tier(raw_notes, pitch_bends, lvl, genre, bpm, detected_key, tuning_pref, y_b, sr_b)
        if score_obj:
            score_obj.write('musicxml', fp=str(out_dir / f"{project_name}_bass_{lvl}_{genre}.musicxml"))

if __name__ == "__main__":
    main()
EOF

for STEMS_DIR in "${STEMS_DIRS[@]}"; do
    if [ ! -d "$STEMS_DIR" ]; then continue; fi
    "$ENV_DIR/bin/python" run_engine_bass.py "$STEMS_DIR" "$TUNING" "$GENRE_OVERRIDE"
done
