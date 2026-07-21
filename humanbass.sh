#!/bin/bash
# ==============================================================================
# Humanized Bass Transcription Pipeline (pYIN / MusicXML Pro Edition)
# Native Apple Silicon (M1/M2/M3) & Linux Support — Dual Staff + Key Detection
# ==============================================================================
set -euo pipefail

OS_TYPE="$(uname -s)"
ARCH_TYPE="$(uname -m)"

ARGS=()
for arg in "$@"; do
    case "$arg" in
        *)
            ARGS+=("$arg")
            ;;
    esac
done

echo "Detected OS: ${OS_TYPE} | Architecture: ${ARCH_TYPE}"

PYTHON_BIN=""

# ------------------------------------------------------------------------------
# 1. Platform-Specific System Dependency Provisioning
# ------------------------------------------------------------------------------
if [[ "$OS_TYPE" == "Darwin" ]]; then
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    if ! brew ls --versions python@3.11 &> /dev/null; then
        echo "Installing Python 3.11..."
        brew install python@3.11 || true
    fi

    if [ -f "/opt/homebrew/opt/python@3.11/bin/python3.11" ]; then
        PYTHON_BIN="/opt/homebrew/opt/python@3.11/bin/python3.11"
    else
        PYTHON_BIN="$(command -v python3.11 || command -v python3)"
    fi

elif [[ "$OS_TYPE" == "Linux" ]]; then
    if command -v python3.11 &> /dev/null; then
        PYTHON_BIN="$(command -v python3.11)"
    elif command -v python3 &> /dev/null; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo "Installing Python 3..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv ffmpeg
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y python3 python3-pip ffmpeg
        fi
        PYTHON_BIN="$(command -v python3)"
    fi
else
    echo "FATAL: Unsupported Operating System: ${OS_TYPE}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Virtual Environment Setup & Dependencies
# ------------------------------------------------------------------------------
ENV_DIR="${PWD}/.bass_pipeline_env"

if [ ! -d "$ENV_DIR" ]; then
    echo "Provisioning isolated Python environment..."
    "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

"$ENV_DIR/bin/pip" install --upgrade pip --quiet
"$ENV_DIR/bin/pip" install --quiet \
    "setuptools<82" \
    "numpy==1.26.4" \
    "scipy==1.14.1" \
    "soundfile==0.12.1" \
    "soxr==0.3.7" \
    "librosa>=0.10.2" \
    "music21==9.1.0"

# ------------------------------------------------------------------------------
# 3. Production Python Engine Generation
# ------------------------------------------------------------------------------
cat << 'EOF' > run_pipeline.py
import os
import sys

import re
import math
import fractions
import xml.etree.ElementTree as ET
import numpy as np
import scipy.signal as signal
from scipy.ndimage import median_filter
import librosa
import soundfile as sf
from music21 import stream, note, pitch, meter, tie, articulations, tempo, clef, instrument, metadata, key, spanner, dynamics


def parse_metadata_from_path(folder_path):
    """Extracts human-readable Artist and Title from stem directory names."""
    folder_name = os.path.basename(os.path.normpath(folder_path))
    
    base_name = re.sub(r'(?i)^stems_', '', folder_name)
    base_name = re.sub(r'(?i)stems_', '', base_name)
    
    clean = base_name.replace("_", " ").strip()
    
    if " - " in clean:
        parts = clean.split(" - ", 1)
        artist = parts[0].strip().title()
        title = parts[1].strip().title()
    elif "-" in clean:
        parts = clean.split("-", 1)
        artist = parts[0].strip().title()
        title = parts[1].strip().title()
    else:
        artist = "Unknown Artist"
        title = clean.title() if clean else "Bass Track"
        
    artist = re.sub(r'(?i)stems_?', '', artist).strip()
    title = re.sub(r'(?i)stems_?', '', title).strip()
        
    return artist, title, base_name


def detect_key_signature(audio_y, sr):
    """Estimates harmonic key using Chroma CQT and Krumhansl-Schmuckler profiles."""
    chroma = librosa.feature.chroma_cqt(y=audio_y, sr=sr)
    chroma_sum = np.sum(chroma, axis=1)

    if np.sum(chroma_sum) == 0:
        return key.Key('C')

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.98, 2.69, 3.34, 3.17])

    pitch_names = ['C', 'C#', 'D', 'E-', 'E', 'F', 'F#', 'G', 'A-', 'A', 'B-', 'B']

    best_score = -float('inf')
    best_key_str = 'C'

    for i in range(12):
        rot_chroma = np.roll(chroma_sum, -i)
        maj_corr = np.corrcoef(rot_chroma, major_profile)[0, 1]
        min_corr = np.corrcoef(rot_chroma, minor_profile)[0, 1]

        maj_corr = 0.0 if np.isnan(maj_corr) else maj_corr
        min_corr = 0.0 if np.isnan(min_corr) else min_corr

        if maj_corr > best_score:
            best_score = maj_corr
            best_key_str = pitch_names[i]
        if min_corr > best_score:
            best_score = min_corr
            best_key_str = pitch_names[i].lower()

    try:
        return key.Key(best_key_str)
    except Exception:
        return key.Key('C')


def decompose_duration(dur_quarter, is_compound=False):
    """Decomposes durations into standard beat-aligned values."""
    try:
        val = float(dur_quarter)
        if math.isnan(val) or math.isinf(val) or val <= 0:
            return [fractions.Fraction(1, 4)]
        rem = fractions.Fraction.from_float(val).limitdenominator(16)
    except Exception:
        try:
            rem = fractions.Fraction(dur_quarter).limitdenominator(16)
        except Exception:
            return [fractions.Fraction(1, 4)]

    if is_compound:
        standard_units = [
            fractions.Fraction(6, 1), fractions.Fraction(3, 1), fractions.Fraction(3, 2),
            fractions.Fraction(3, 4), fractions.Fraction(1, 2), fractions.Fraction(1, 4)
        ]
    else:
        standard_units = [
            fractions.Fraction(4, 1), fractions.Fraction(3, 1), fractions.Fraction(2, 1),
            fractions.Fraction(3, 2), fractions.Fraction(1, 1), fractions.Fraction(3, 4),
            fractions.Fraction(1, 2), fractions.Fraction(1, 4)
        ]

    pieces = []
    while rem > 0:
        fit = False
        for unit in standard_units:
            if unit <= rem:
                pieces.append(unit)
                rem -= unit
                fit = True
                break
        if not fit:
            if rem > 0:
                pieces.append(rem)
            break

    return pieces if pieces else [fractions.Fraction(1, 4)]


def pyin_predict_notes(audio_y, sr, conf_threshold=0.30):
    """C-accelerated pYIN pitch tracking."""
    hop_length = 512
    frame_length = 2048

    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_y, fmin=25.0, fmax=350.0, sr=sr,
        frame_length=frame_length, hop_length=hop_length
    )

    f0 = np.nan_to_num(f0)
    voiced_probs = np.nan_to_num(voiced_probs)
    voiced_probs = median_filter(voiced_probs, size=3)
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    raw_notes = []
    in_note = False
    start_time = 0.0
    pitch_buf = []
    conf_buf = []

    for t, f, c in zip(times, f0, voiced_probs):
        if f > 0.0 and c >= conf_threshold:
            midi_p = librosa.hz_to_midi(f)
            if not in_note:
                in_note = True
                start_time = t
                pitch_buf = [midi_p]
                conf_buf = [c]
            else:
                curr_med = np.median(pitch_buf)
                if abs(midi_p - curr_med) > 1.5:
                    end_time = t
                    if (end_time - start_time) >= 0.04:
                        med_pitch = int(round(np.median(pitch_buf)))
                        raw_notes.append((start_time, end_time, med_pitch, float(np.mean(conf_buf)), None))
                    start_time = t
                    pitch_buf = [midi_p]
                    conf_buf = [c]
                else:
                    pitch_buf.append(midi_p)
                    conf_buf.append(c)
        else:
            if in_note:
                end_time = t
                if pitch_buf and (end_time - start_time) >= 0.04:
                    med_pitch = int(round(np.median(pitch_buf)))
                    raw_notes.append((start_time, end_time, med_pitch, float(np.mean(conf_buf)), None))
                in_note = False
                pitch_buf, conf_buf = [], []

    return raw_notes


def fill_micro_gaps(note_events, max_gap=0.18):
    """Gap-Filling / Legato Pass."""
    if not note_events:
        return []

    smoothed = []
    curr = list(note_events[0])

    for next_n in note_events[1:]:
        c_start, c_end, c_pitch, c_amp, c_bends = curr
        n_start, n_end, n_pitch, n_amp, n_bends = next_n

        gap = n_start - c_end
        if 0 < gap <= max_gap:
            curr[1] = n_start

        smoothed.append(tuple(curr))
        curr = list(next_n)

    smoothed.append(tuple(curr))
    return smoothed


class ErgonomicFretboardHMMSolver:
    """Biomechanical Fretboard State Machine with Hand Position Inertia Cost."""
    def __init__(self, tuning_type='4_string_standard'):
        self.tuning_type = tuning_type
        if tuning_type == '5_string_low_b':
            self.strings = {1: 43, 2: 38, 3: 33, 4: 28, 5: 23}
        elif tuning_type == '4_string_drop_d':
            self.strings = {1: 43, 2: 38, 3: 33, 4: 26}
        else:
            self.strings = {1: 43, 2: 38, 3: 33, 4: 28}
        
        self.num_frets = 22

    def get_valid_positions(self, midi_pitch):
        positions = []
        for s_idx, open_pitch in self.strings.items():
            fret = midi_pitch - open_pitch
            if 0 <= fret <= self.num_frets:
                positions.append((s_idx, fret))
        return positions

    def solve(self, note_events):
        if not note_events:
            return [], [], []

        sequence_states = []
        for n in note_events:
            p = n[2]
            valid = self.get_valid_positions(p)
            if not valid:
                valid = self.get_valid_positions(p + 12) or self.get_valid_positions(p - 12)
            if not valid:
                valid = [(list(self.strings.keys())[0], 0)]
            sequence_states.append(valid)

        T = len(sequence_states)
        V = [{}]
        path = {}

        for state in sequence_states[0]:
            string_num, fret = state
            tag = note_events[0][5]
            box_cost = 0.0 if (1 <= fret <= 5 or fret == 0) else (fret - 5) * 0.3
            
            tech_cost = 0.0
            if tag == "pop" and string_num > 2:
                tech_cost += 10.0
            elif tag == "slap" and string_num < 3:
                tech_cost += 5.0

            V[0][state] = -(box_cost + tech_cost)
            path[state] = [state]

        for t in range(1, T):
            V.append({})
            new_path = {}
            tag = note_events[t][5]
            is_staccato = (tag == "ghost" or (note_events[t][1] - note_events[t][0]) < 0.15)

            for curr_state in sequence_states[t]:
                (c_string, c_fret) = curr_state
                best_cost = -float('inf')
                best_prev = None

                for prev_state in sequence_states[t-1]:
                    if prev_state not in V[t-1]:
                        continue
                    (p_string, p_fret) = prev_state

                    if c_fret == 0:
                        shift_cost = 0.5 if not is_staccato else 2.5
                    elif p_fret == 0:
                        shift_cost = 0.5
                    else:
                        fret_delta = abs(c_fret - p_fret)
                        shift_cost = (fret_delta ** 1.4) * 0.8

                    string_delta = abs(c_string - p_string)
                    string_jump_cost = string_delta * 0.7
                    high_fret_cost = (c_fret - 12) * 0.6 if c_fret > 12 else 0.0

                    tech_cost = 0.0
                    if tag == "pop" and c_string > 2:
                        tech_cost += 15.0
                    elif tag == "slap" and c_string < 3:
                        tech_cost += 8.0

                    total_trans_cost = shift_cost + string_jump_cost + high_fret_cost + tech_cost
                    total_score = V[t-1][prev_state] - total_trans_cost

                    if total_score > best_cost:
                        best_cost = total_score
                        best_prev = prev_state

                if best_prev is None and sequence_states[t-1]:
                    best_prev = sequence_states[t-1][0]
                    best_cost = V[t-1].get(best_prev, 0.0) - 10.0

                V[t][curr_state] = best_cost
                new_path[curr_state] = path.get(best_prev, [curr_state]) + [curr_state]

            path = new_path

        if V[-1]:
            best_final_state = max(V[-1], key=V[-1].get)
            optimal_states = path.get(best_final_state, [sequence_states[-1][0]])
        else:
            optimal_states = [s[0] for s in sequence_states]

        rakes = [False] * T
        legatos = [False] * T
        for i in range(1, T):
            prev_string, prev_fret = optimal_states[i-1]
            curr_string, curr_fret = optimal_states[i]
            dt = note_events[i][0] - note_events[i-1][1]
            
            if curr_string > prev_string and dt < 0.12:
                rakes[i] = True

            if curr_string == prev_string and abs(curr_fret - prev_fret) in [1, 2, 3] and dt < 0.04:
                legatos[i] = True

        return optimal_states, rakes, legatos


def sanitize_and_inject_tablature(xml_path, artist_name, song_title, tuning_type):
    """
    DOM Sanitizer & Technical Notation Injector.
    Injects technical string/fret data, hammer-ons/pull-offs, dead notes,
    and adds proper MusicXML <staff-details> attributes.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = ""
    if '}' in root.tag:
        ns = root.tag.split('}')[0] + '}'

    # 1. Guarantee Standard MusicXML Root & Part-List Header
    root.tag = f"{ns}score-partwise"
    
    part_list = root.find(f"{ns}part-list")
    if part_list is None:
        part_list = ET.Element(f"{ns}part-list")
        root.insert(0, part_list)
    
    score_part = part_list.find(f"{ns}score-part")
    if score_part is None:
        score_part = ET.SubElement(part_list, f"{ns}score-part", attrib={"id": "P1"})
    else:
        score_part.attrib["id"] = "P1"

    part_name = score_part.find(f"{ns}part-name")
    if part_name is None:
        part_name = ET.SubElement(score_part, f"{ns}part-name")
    part_name.text = "Electric Bass"

    # 2. Work & Identification Metadata
    work_elem = root.find(f"{ns}work")
    if work_elem is None:
        work_elem = ET.Element(f"{ns}work")
        root.insert(1, work_elem)
    work_title = work_elem.find(f"{ns}work-title")
    if work_title is None:
        work_title = ET.SubElement(work_elem, f"{ns}work-title")
    work_title.text = song_title

    ident = root.find(f"{ns}identification")
    if ident is None:
        ident = ET.SubElement(root, f"{ns}identification")
    creator_elem = ident.find(f"{ns}creator")
    if creator_elem is None:
        creator_elem = ET.SubElement(ident, f"{ns}creator", attrib={"type": "composer"})
    creator_elem.text = artist_name

    # 3. Inject Instrument Staff Details (Tuning & Lines) into Measure 1
    first_part = root.find(f"{ns}part")
    if first_part is not None:
        first_measure = first_part.find(f"{ns}measure")
        if first_measure is not None:
            attrs = first_measure.find(f"{ns}attributes")
            if attrs is None:
                attrs = ET.Element(f"{ns}attributes")
                first_measure.insert(0, attrs)

            staff_details = ET.SubElement(attrs, f"{ns}staff-details")
            num_strings = 5 if tuning_type == '5_string_low_b' else 4
            ET.SubElement(staff_details, f"{ns}staff-lines").text = str(num_strings)

            tuning_map = {
                '5_string_low_b': [('G', 2), ('D', 2), ('A', 1), ('E', 1), ('B', 0)],
                '4_string_drop_d': [('G', 2), ('D', 2), ('A', 1), ('D', 1)],
                '4_string_standard': [('G', 2), ('D', 2), ('A', 1), ('E', 1)]
            }
            tunings = tuning_map.get(tuning_type, tuning_map['4_string_standard'])
            for s_idx, (step, oct_val) in enumerate(tunings, 1):
                s_tuning = ET.SubElement(staff_details, f"{ns}staff-tuning", attrib={"line": str(s_idx)})
                ET.SubElement(s_tuning, f"{ns}tuning-step").text = step
                ET.SubElement(s_tuning, f"{ns}tuning-octave").text = str(oct_val)

    # 4. Technical Notation Attachment & Dynamic Refinement
    prev_fret_num = None
    prev_string_num = None

    for part in root.findall(f"{ns}part"):
        part.attrib["id"] = "P1"
        
        for measure in part.findall(f"{ns}measure"):
            for dummy_rest in list(measure.findall(f"{ns}note")):
                if dummy_rest.attrib.get("print-object") == "no":
                    measure.remove(dummy_rest)

            notes_to_process = list(measure.findall(f"{ns}note"))
            for note_elem in notes_to_process:
                # Strip raw numeric dynamics/velocity from individual notes
                if 'dynamics' in note_elem.attrib:
                    del note_elem.attrib['dynamics']

                lyric_elem = note_elem.find(f"{ns}lyric")
                string_num, fret_num = None, None
                if lyric_elem is not None:
                    text_elem = lyric_elem.find(f"{ns}text")
                    if text_elem is not None and text_elem.text and text_elem.text.startswith('S') and ':F' in text_elem.text:
                        try:
                            s_part, f_part = text_elem.text.split(':F')
                            string_num = s_part[1:]
                            fret_num = f_part
                        except Exception:
                            pass
                        note_elem.remove(lyric_elem)

                if string_num is not None and fret_num is not None:
                    notations = note_elem.find(f"{ns}notations")
                    if notations is None:
                        notations = ET.SubElement(note_elem, f"{ns}notations")
                    
                    technical = notations.find(f"{ns}technical")
                    if technical is None:
                        technical = ET.SubElement(notations, f"{ns}technical")

                    string_el = ET.SubElement(technical, f"{ns}string")
                    string_el.text = str(string_num)
                    fret_el = ET.SubElement(technical, f"{ns}fret")
                    fret_el.text = str(fret_num)

                    # Inject Hammer-on / Pull-off technical tags where slurs exist
                    if prev_string_num == string_num and prev_fret_num is not None and fret_num != prev_fret_num:
                        slur_elem = notations.find(f"{ns}slur")
                        if slur_elem is not None:
                            curr_f = int(fret_num)
                            prev_f = int(prev_fret_num)
                            if curr_f > prev_f:
                                ET.SubElement(technical, f"{ns}hammer-on", attrib={"type": "stop"}).text = "H"
                            elif curr_f < prev_f:
                                ET.SubElement(technical, f"{ns}pull-off", attrib={"type": "stop"}).text = "P"

                    prev_string_num = string_num
                    prev_fret_num = fret_num

                # Convert ghost noteheads to muted x noteheads
                notehead = note_elem.find(f"{ns}notehead")
                if notehead is not None and notehead.text == "cross":
                    notehead.text = "x"

    # 5. Clean Metadata Text
    for elem in root.iter():
        if elem.text and 'stems_' in elem.text.lower():
            elem.text = re.sub(r'(?i)stems_?', '', elem.text).strip()

    if ns:
        ET.register_namespace('', ns.strip('{}'))

    tree.write(xml_path, encoding='utf-8', xml_declaration=True)


def process_folder(stem_folder):
    artist_name, song_title, base_name = parse_metadata_from_path(stem_folder)

    print(f"\n=======================================================")
    print(f"TRACK: {artist_name} - {song_title}")
    print(f"STEM FOLDER: {os.path.abspath(stem_folder)}")
    print(f"=======================================================")

    bass_path = os.path.join(stem_folder, 'bass.wav')
    drums_path = os.path.join(stem_folder, 'drums.wav')

    if not os.path.exists(bass_path) or not os.path.exists(drums_path):
        print(f"SKIPPED: Missing bass.wav or drums.wav")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output_bass')
    os.makedirs(output_dir, exist_ok=True)

    sr = 22050

    # --------------------------------------------------------------------------
    # PHASE 1: Low-Pass Filtering & Harmonic Key Signature Detection
    # --------------------------------------------------------------------------
    print("[Phase 1/5] Filtering Audio & Estimating Harmonic Key Signature...")
    bass_y, _ = librosa.load(bass_path, sr=sr, mono=True)
    drums_y, _ = librosa.load(drums_path, sr=sr, mono=True)

    bass_y = np.asarray(bass_y, dtype=np.float32)
    drums_y = np.asarray(drums_y, dtype=np.float32)

    detected_key = detect_key_signature(bass_y, sr)
    print(f"        Detected Key: {detected_key.name}")

    sos_low = signal.butter(4, [25 / (sr / 2), 280 / (sr / 2)], 'bandpass', output='sos')
    bass_low = signal.sosfiltfilt(sos_low, bass_y)
    bass_low = np.asarray(bass_low, dtype=np.float32)

    sos_high = signal.butter(4, 1800 / (sr / 2), 'highpass', output='sos')
    bass_high = signal.sosfiltfilt(sos_high, bass_y)
    bass_high = np.asarray(bass_high, dtype=np.float32)

    # --------------------------------------------------------------------------
    # PHASE 2: High-Speed pYIN Pitch Tracking
    # --------------------------------------------------------------------------
    print("[Phase 2/5] Running pYIN Autocorrelation Pitch Tracking...")
    raw_pyin_notes = pyin_predict_notes(bass_low, sr, conf_threshold=0.30)
    print(f"        Detected {len(raw_pyin_notes)} raw pitch events.")

    if not raw_pyin_notes:
        print("    [WARNING] pYIN detected 0 notes. Retrying with lower threshold 0.15...")
        raw_pyin_notes = pyin_predict_notes(bass_low, sr, conf_threshold=0.15)
        print(f"        Fallback detected {len(raw_pyin_notes)} raw pitch events.")

    corrected_notes = []
    for start, end, midi_pitch, amp, bends in raw_pyin_notes:
        while midi_pitch > 67:
            midi_pitch -= 12
        while midi_pitch < 23:
            midi_pitch += 12

        clamped_pitch = max(23, min(67, int(round(midi_pitch))))
        corrected_notes.append((start, end, clamped_pitch, float(amp), bends))

    # --------------------------------------------------------------------------
    # PHASE 3: Micro-Rest Smoothing & Articulation Tagging
    # --------------------------------------------------------------------------
    print("[Phase 3/5] Micro-Rest Smoothing & Articulation Tagging...")
    smoothed_notes = fill_micro_gaps(corrected_notes, max_gap=0.18)

    drum_onsets = librosa.onset.onset_detect(y=drums_y, sr=sr, units='time')
    high_flux = librosa.onset.onset_strength(y=bass_high, sr=sr)
    flux_times = librosa.times_like(high_flux, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=bass_y)[0]

    performance_layer = []
    for start, end, pitch_val, amp, bends in smoothed_notes:
        duration = end - start
        closest_drum_dist = min([abs(start - d) for d in drum_onsets]) if len(drum_onsets) > 0 else 999.0
        is_drum_aligned = closest_drum_dist < 0.035

        frame_idx = np.argmin(np.abs(flux_times - start))
        local_flux = high_flux[frame_idx] if frame_idx < len(high_flux) else 0.0
        local_zcr = zcr[frame_idx] if frame_idx < len(zcr) else 0.0

        if is_drum_aligned and local_flux < 0.20 and duration < 0.10:
            continue

        tag = "normal"
        if local_flux > 1.4 and not is_drum_aligned:
            tag = "pop"
        elif local_flux > 1.6 and is_drum_aligned:
            tag = "slap"
        elif duration < 0.10 and amp < 0.40 and local_zcr > 0.12:
            tag = "ghost"

        performance_layer.append((start, end, pitch_val, amp, bends, tag))

    performance_layer.sort(key=lambda x: x[0])

    # --------------------------------------------------------------------------
    # PHASE 4: Meter Detection, Pocket Delta & Fingering HMM
    # --------------------------------------------------------------------------
    print("[Phase 4/5] Meter, Pocket Delta & Fingering Calculation...")
    tempo_val, beats = librosa.beat.beat_track(y=drums_y, sr=sr, units='time')
    bpm = float(np.atleast_1d(tempo_val)[0])
    bpm = 120.0 if bpm <= 0 else bpm

    beat_intervals = np.diff(beats) if len(beats) > 1 else [0.5]
    avg_interval = float(np.mean(beat_intervals))
    is_compound = (avg_interval > 0.65) and (bpm < 95.0)

    time_sig_str = '12/8' if is_compound else '4/4'
    m_capacity = fractions.Fraction(6, 1) if is_compound else fractions.Fraction(4, 1)
    
    print(f"        Tempo: {bpm:.2f} BPM | Time Signature: {time_sig_str}")

    bass_onsets = [n[0] for n in performance_layer]
    pocket_deltas = [b_onset - min(beats, key=lambda d: abs(d - b_onset))
                     for b_onset in bass_onsets if len(beats) > 0 and abs(b_onset - min(beats, key=lambda d: abs(d - b_onset))) < 0.08]

    pocket_delta = float(np.median(pocket_deltas)) if pocket_deltas else 0.0

    valid_pitches = [n[2] for n in performance_layer]
    lowest_pitch = min(valid_pitches) if valid_pitches else 40
    
    if lowest_pitch <= 25:
        tuning = '5_string_low_b'
    elif lowest_pitch <= 27:
        tuning = '4_string_drop_d'
    else:
        tuning = '4_string_standard'

    hmm = ErgonomicFretboardHMMSolver(tuning_type=tuning)
    fretboard_path, rakes, legatos = hmm.solve(performance_layer)

    # --------------------------------------------------------------------------
    # PHASE 5: Measure Assembly, Dynamic Mapping & Clean MusicXML Generation
    # --------------------------------------------------------------------------
    print("[Phase 5/5] Generating Measure-Bound Notation...")

    sec_per_quarter = 60.0 / bpm
    first_onset = performance_layer[0][0] if performance_layer else 0.0

    quantized_timeline = []
    current_q = fractions.Fraction(0, 1)

    for i, (start, end, pitch_val, amp, bends, tag) in enumerate(performance_layer):
        v_start = max(0.0, (start - first_onset) - pocket_delta)
        dur_s = max(0.05, end - start)

        start_q = fractions.Fraction(int(round((v_start / sec_per_quarter) * 4)), 4)
        dur_q = max(fractions.Fraction(1, 4), fractions.Fraction(int(round((dur_s / sec_per_quarter) * 4)), 4))

        if start_q < current_q:
            start_q = current_q

        if start_q > current_q:
            rest_len = start_q - current_q
            quantized_timeline.append(('rest', rest_len, None, None, None, None, None, False, False))
            current_q = start_q

        s_idx, f_val = fretboard_path[i] if i < len(fretboard_path) else (None, None)
        exact_midi = (hmm.strings[s_idx] + f_val) if (s_idx is not None and f_val is not None) else pitch_val
        is_rake = rakes[i] if i < len(rakes) else False
        is_legato = legatos[i] if i < len(legatos) else False

        quantized_timeline.append(('note', dur_q, exact_midi, amp, tag, s_idx, f_val, is_rake, is_legato))
        current_q += dur_q

    m21_score = stream.Score()
    m21_part = stream.Part()
    m21_part.partName = "Electric Bass"

    m21_score.metadata = metadata.Metadata()
    m21_score.metadata.title = song_title
    m21_score.metadata.composer = artist_name

    bass_inst = instrument.ElectricBass()
    bass_inst.partName = "Electric Bass"
    m21_part.insert(0.0, bass_inst)

    m_fill = fractions.Fraction(0, 1)
    m_num = 1

    curr_measure = stream.Measure(number=m_num)
    curr_measure.insert(0.0, clef.BassClef())
    curr_measure.insert(0.0, meter.TimeSignature(time_sig_str))
    curr_measure.insert(0.0, detected_key)
    curr_measure.insert(0.0, tempo.MetronomeMark(number=int(round(bpm))))

    # Phrase Dynamic Tracking
    last_dynamic_str = None
    prev_note_obj = None

    for event in quantized_timeline:
        ev_type = event[0]

        if ev_type == 'rest':
            rem_dur = event[1]
            while rem_dur > 0:
                space = m_capacity - m_fill
                if space <= 0:
                    m21_part.append(curr_measure)
                    m_num += 1
                    curr_measure = stream.Measure(number=m_num)
                    m_fill = fractions.Fraction(0, 1)
                    space = m_capacity

                take_dur = min(rem_dur, space)
                for sub_dur in decompose_duration(take_dur, is_compound):
                    r = note.Rest(quarterLength=float(sub_dur))
                    r.voice = 1
                    curr_measure.append(r)
                    
                m_fill += take_dur
                rem_dur -= take_dur

        elif ev_type == 'note':
            _, dur_q, exact_midi, amp, tag, s_idx, f_val, is_rake, is_legato = event
            rem_dur = dur_q
            is_first_piece = True

            # Convert amplitude to coarse dynamic marking
            if amp < 0.35:
                curr_dynamic_str = 'p'
            elif amp < 0.65:
                curr_dynamic_str = 'mf'
            elif amp < 0.85:
                curr_dynamic_str = 'f'
            else:
                curr_dynamic_str = 'ff'

            while rem_dur > 0:
                space = m_capacity - m_fill
                if space <= 0:
                    m21_part.append(curr_measure)
                    m_num += 1
                    curr_measure = stream.Measure(number=m_num)
                    m_fill = fractions.Fraction(0, 1)
                    space = m_capacity

                take_dur = min(rem_dur, space)
                dur_pieces = decompose_duration(take_dur, is_compound)
                for p_idx, sub_dur in enumerate(dur_pieces):
                    n = note.Note(pitch.Pitch(midi=exact_midi))
                    n.quarterLength = float(sub_dur)
                    n.voice = 1

                    # Apply phrase dynamic marking only when dynamic state changes
                    if is_first_piece and p_idx == 0 and curr_dynamic_str != last_dynamic_str:
                        d_mark = dynamics.Dynamic(curr_dynamic_str)
                        curr_measure.insert(m_fill, d_mark)
                        last_dynamic_str = curr_dynamic_str

                    if is_first_piece and p_idx == 0:
                        if tag == "ghost":
                            n.notehead = 'cross'
                            n.articulations.append(articulations.Staccato())
                        elif tag == "slap":
                            n.articulations.append(articulations.StrongAccent())
                        elif tag == "pop":
                            n.articulations.append(articulations.Accent())

                        if is_legato and prev_note_obj is not None:
                            sl = spanner.Slur(prev_note_obj, n)
                            curr_measure.insert(0, sl)

                        if s_idx is not None and f_val is not None:
                            n.addLyric(f"S{s_idx}:F{f_val}")

                        prev_note_obj = n

                    is_last_subpiece = (p_idx == len(dur_pieces) - 1) and (rem_dur == take_dur)
                    if not is_last_subpiece:
                        n.tie = tie.Tie('start') if (is_first_piece and p_idx == 0) else tie.Tie('continue')
                    else:
                        if not (is_first_piece and p_idx == 0):
                            n.tie = tie.Tie('stop')

                    curr_measure.append(n)

                m_fill += take_dur
                rem_dur -= take_dur
                is_first_piece = False

    if len(curr_measure.notesAndRests) > 0:
        if m_fill < m_capacity and m_fill > 0:
            for sub_dur in decompose_duration(m_capacity - m_fill, is_compound):
                r = note.Rest(quarterLength=float(sub_dur))
                r.voice = 1
                curr_measure.append(r)
        m21_part.append(curr_measure)

    # Post-process score with quantize and makeNotation to beam properly and consolidate rests
    m21_part = m21_part.quantize(quarterLengthDivisors=(4,)).makeNotation()
    m21_score.append(m21_part)

    xml_out = os.path.join(output_dir, f"{base_name}.musicxml")
    m21_score.write('musicxml', fp=xml_out)

    sanitize_and_inject_tablature(xml_out, artist_name, song_title, tuning)

    print(f"\nSUCCESS: Generated Clean MusicXML Asset inside {output_dir}")
    print(f" -> Notation (MusicXML): {xml_out}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./humanbass.sh <path_to_stem_folder_1> ...")
        sys.exit(1)

    for folder in sys.argv[1:]:
        if os.path.isdir(folder):
            process_folder(folder)
        else:
            print(f"Directory non-existent: {folder}")
EOF

# ------------------------------------------------------------------------------
# 4. Execution
# ------------------------------------------------------------------------------
if [ ${#ARGS[@]} -eq 0 ]; then
    echo "Usage: ./humanbass.sh <path_to_stem_folder_1> ..."
    exit 1
fi

"$ENV_DIR/bin/python" run_pipeline.py "${ARGS[@]}"

echo "Pipeline execution finished successfully."
