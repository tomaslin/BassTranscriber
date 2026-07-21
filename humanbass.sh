#!/bin/bash
# ==============================================================================
# Humanized Bass Transcription Pipeline (Cross-Platform: Linux & Apple Silicon)
# Default: Linux CPU / macOS Metal. Optional: Linux GPU via `--use-gpu` flag.
# ==============================================================================
set -euo pipefail

OS_TYPE="$(uname -s)"
ARCH_TYPE="$(uname -m)"
USE_GPU="${USE_GPU:-0}"

ARGS=()
for arg in "$@"; do
    case "$arg" in
        --use-gpu)
            USE_GPU=1
            ;;
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
# 2. Virtual Environment Setup & Resilient Dependency Provisioning
# ------------------------------------------------------------------------------
ENV_DIR="${PWD}/.bass_pipeline_env"

if [ ! -d "$ENV_DIR" ]; then
    echo "Provisioning isolated Python environment..."
    "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

source "$ENV_DIR/bin/activate"

pip install --upgrade pip --quiet
pip install --quiet \
    "setuptools<82" \
    "numpy==1.26.4" \
    "scipy==1.14.1" \
    "soundfile==0.12.1" \
    "soxr==0.3.7" \
    "librosa>=0.10.2" \
    "music21==9.1.0" \
    "pretty_midi==0.2.10" \
    "basic-pitch>=0.4.0" \
    "resampy==0.4.2"

# Resilient TensorFlow Stack Installation
if [[ "$OS_TYPE" == "Darwin" && "$ARCH_TYPE" == "arm64" ]]; then
    pip install --quiet "tensorflow-macos<2.16.0" "tensorflow-metal==1.1.0" 2>/dev/null || \
    pip install --quiet "tensorflow" || true
elif [[ "$OS_TYPE" == "Linux" ]]; then
    if [[ "$USE_GPU" -eq 1 ]]; then
        pip install --quiet "tensorflow[and-cuda]<2.16.0" 2>/dev/null || \
        pip install --quiet "tensorflow" || true
    else
        pip install --quiet "tensorflow<2.16.0" 2>/dev/null || \
        pip install --quiet "tensorflow" || true
    fi
fi

# ------------------------------------------------------------------------------
# 3. Production Python Engine Generation
# ------------------------------------------------------------------------------
cat << 'EOF' > run_pipeline.py
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import math
import contextlib
import fractions
import xml.etree.ElementTree as ET
import numpy as np
import scipy.signal as signal
import librosa
import soundfile as sf
import pretty_midi
from music21 import stream, note, pitch, meter, tie, articulations, spanner, tempo, clef, instrument, metadata

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            yield


def parse_metadata_from_path(folder_path):
    """Extracts human-readable Artist and Title from stem directory names while stripping 'stems_'."""
    folder_name = os.path.basename(os.path.normpath(folder_path))
    
    # Thoroughly strip 'stems_' prefix and occurrences from base name
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


class ErgonomicFretboardHMMSolver:
    """
    Biomechanical Fretboard State Machine.
    Models hand position boxes, string jump effort, open string muting context,
    and technique-specific constraints (Slap vs Pop geometry).
    """
    def __init__(self, tuning_type='4_string_standard'):
        self.tuning_type = tuning_type
        if tuning_type == '5_string_low_b':
            # String 5: B0(23), 4: E1(28), 3: A1(33), 2: D2(38), 1: G2(43)
            self.strings = {1: 43, 2: 38, 3: 33, 4: 28, 5: 23}
        elif tuning_type == '4_string_drop_d':
            # String 4: D1(26), 3: A1(33), 2: D2(38), 1: G2(43)
            self.strings = {1: 43, 2: 38, 3: 33, 4: 26}
        else:
            # String 4: E1(28), 3: A1(33), 2: D2(38), 1: G2(43)
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
            return [], []

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

                    high_fret_cost = 0.0
                    if c_fret > 12 and c_string >= 3:
                        high_fret_cost = (c_fret - 12) * 0.4

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

                V[t][curr_state] = best_cost
                new_path[curr_state] = path[best_prev] + [curr_state]

            path = new_path

        best_final_state = max(V[-1], key=V[-1].get)
        optimal_states = path[best_final_state]

        # Rake detection: Moving downward across adjacent higher-frequency strings
        rakes = [False] * T
        for i in range(1, T):
            prev_string, _ = optimal_states[i-1]
            curr_string, _ = optimal_states[i]
            dt = note_events[i][0] - note_events[i-1][1]
            if curr_string > prev_string and dt < 0.12:
                rakes[i] = True

        return optimal_states, rakes


def sanitize_and_inject_tablature(xml_path, artist_name, song_title, tuning_type):
    """
    Direct MusicXML DOM sanitizer. Injects metadata, cleans ghost rests,
    adds octave transposition, and structures native tablature attributes.
    Also strips any leftover 'stems_' references from all text fields.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = ""
    if '}' in root.tag:
        ns = root.tag.split('}')[0] + '}'

    # 1. Metadata Injection
    work_elem = root.find(f"{ns}work")
    if work_elem is None:
        work_elem = ET.Element(f"{ns}work")
        root.insert(0, work_elem)
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

    # 2. Staff Details & Octave Transposition (-1 octave for standard bass pitch)
    for part in root.findall(f"{ns}part"):
        for measure in part.findall(f"{ns}measure"):
            if measure.attrib.get('number') == '1':
                attributes = measure.find(f"{ns}attributes")
                if attributes is None:
                    attributes = ET.Element(f"{ns}attributes")
                    measure.insert(0, attributes)

                transpose = attributes.find(f"{ns}transpose")
                if transpose is None:
                    transpose = ET.SubElement(attributes, f"{ns}transpose")
                    ET.SubElement(transpose, f"{ns}diatonic").text = "0"
                    ET.SubElement(transpose, f"{ns}chromatic").text = "0"
                    ET.SubElement(transpose, f"{ns}octave-change").text = "-1"

                staff_details = attributes.find(f"{ns}staff-details")
                if staff_details is None:
                    staff_details = ET.SubElement(attributes, f"{ns}staff-details")
                
                # Standard pitch staff requires 5 lines
                staff_lines = staff_details.find(f"{ns}staff-lines")
                if staff_lines is None:
                    staff_lines = ET.SubElement(staff_details, f"{ns}staff-lines")
                staff_lines.text = "5"

                tunings = [
                    ("1", "G", "2"), ("2", "D", "2"), ("3", "A", "1"), ("4", "E", "1")
                ]
                if tuning_type == '5_string_low_b':
                    tunings.append(("5", "B", "0"))
                elif tuning_type == '4_string_drop_d':
                    tunings[3] = ("4", "D", "1")

                for line, step, octv in tunings:
                    st = ET.SubElement(staff_details, f"{ns}staff-tuning", attrib={"line": line})
                    ET.SubElement(st, f"{ns}tuning-step").text = step
                    ET.SubElement(st, f"{ns}tuning-octave").text = octv

    # 3. Clean Note Elements: Voice 1, Ghost Rest Stripping, Technical TAB Conversion
    for note_elem in root.iter(f"{ns}note"):
        if 'print-object' in note_elem.attrib:
            del note_elem.attrib['print-object']
        if 'print-spacing' in note_elem.attrib:
            del note_elem.attrib['print-spacing']

        voice_elem = note_elem.find(f"{ns}voice")
        if voice_elem is None:
            v_elem = ET.Element(f"{ns}voice")
            v_elem.text = "1"
            note_elem.insert(1, v_elem)

        lyric_elem = note_elem.find(f"{ns}lyric")
        if lyric_elem is not None:
            text_elem = lyric_elem.find(f"{ns}text")
            if text_elem is not None and text_elem.text and text_elem.text.startswith('S') and ':F' in text_elem.text:
                txt = text_elem.text
                try:
                    s_part, f_part = txt.split(':F')
                    string_num = s_part[1:]
                    fret_num = f_part

                    note_elem.remove(lyric_elem)

                    notations = note_elem.find(f"{ns}notations")
                    if notations is None:
                        notations = ET.SubElement(note_elem, f"{ns}notations")

                    technical = notations.find(f"{ns}technical")
                    if technical is None:
                        technical = ET.SubElement(notations, f"{ns}technical")

                    ET.SubElement(technical, f"{ns}string").text = string_num
                    ET.SubElement(technical, f"{ns}fret").text = fret_num
                except Exception:
                    pass

    # 4. Global Scrub: Ensure 'stems_' is completely eliminated from all XML text nodes
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
    # PHASE 1: Stable DSP Filtering (Second-Order Sections)
    # --------------------------------------------------------------------------
    print("[Phase 1/6] Ingesting Stems & Banding Filters (SOS)...")
    bass_y, _ = librosa.load(bass_path, sr=sr, mono=True)
    drums_y, _ = librosa.load(drums_path, sr=sr, mono=True)

    sos_low = signal.butter(4, [25 / (sr / 2), 500 / (sr / 2)], 'bandpass', output='sos')
    bass_low = signal.sosfiltfilt(sos_low, bass_y)

    sos_high = signal.butter(4, 2000 / (sr / 2), 'highpass', output='sos')
    bass_high = signal.sosfiltfilt(sos_high, bass_y)

    # --------------------------------------------------------------------------
    # PHASE 2: ML Inference & Fundamental Verification
    # --------------------------------------------------------------------------
    print("[Phase 2/6] Basic-Pitch Neural Inference & Sub-Bass Verification...")
    
    temp_low_path = os.path.join(output_dir, f'_temp_{base_name}_low.wav')
    sf.write(temp_low_path, bass_low, sr)

    with suppress_stdout_stderr():
        model_output, midi_data, note_events = predict(
            temp_low_path,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=0.5,
            frame_threshold=0.3
        )

    if os.path.exists(temp_low_path):
        os.remove(temp_low_path)

    bass_low = np.nan_to_num(bass_low, nan=0.0, posinf=0.0, neginf=0.0)

    hop_length = 512
    f0 = np.full(int(1 + len(bass_low) // hop_length), np.nan)
    
    if len(bass_low) >= 4096 and np.any(bass_low):
        try:
            f0, _, _ = librosa.pyin(
                bass_low,
                fmin=25,
                fmax=220,
                sr=sr,
                frame_length=4096,
                hop_length=hop_length
            )
        except Exception:
            pass

    corrected_notes = []
    for note_item in note_events:
        start, end, midi_pitch, amp = note_item[0], note_item[1], note_item[2], note_item[3]
        bends = note_item[4] if len(note_item) > 4 else None
        
        start_frame = librosa.time_to_frames(start, sr=sr, hop_length=hop_length)
        end_frame = librosa.time_to_frames(end, sr=sr, hop_length=hop_length)
        slice_end = min(end_frame, len(f0))
        
        if start_frame < slice_end:
            f0_slice = f0[start_frame:slice_end]
            valid_f0 = f0_slice[~np.isnan(f0_slice)]
            if len(valid_f0) > 0:
                median_hz = np.median(valid_f0)
                pyin_midi = float(librosa.hz_to_midi(median_hz))
                
                if (midi_pitch - pyin_midi) > 8.0:
                    midi_pitch -= 12

        clamped_pitch = max(21, min(100, int(round(midi_pitch))))
        corrected_notes.append((start, end, clamped_pitch, float(amp), bends))

    # --------------------------------------------------------------------------
    # PHASE 3: Artifact Rejection & Articulation Tagging
    # --------------------------------------------------------------------------
    print("[Phase 3/6] Demucs Bleed Rejection & Articulation Tagging...")
    
    drum_onsets = librosa.onset.onset_detect(y=drums_y, sr=sr, units='time')
    high_flux = librosa.onset.onset_strength(y=bass_high, sr=sr)
    flux_times = librosa.times_like(high_flux, sr=sr)
    
    rolloff = librosa.feature.spectral_rolloff(y=bass_high, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y=bass_y)[0]

    performance_layer = []

    for start, end, pitch_val, amp, bends in corrected_notes:
        duration = end - start
        closest_drum_dist = min([abs(start - d) for d in drum_onsets]) if len(drum_onsets) > 0 else 999.0
        is_drum_aligned = closest_drum_dist < 0.035

        frame_idx = np.argmin(np.abs(flux_times - start))
        local_flux = high_flux[frame_idx] if frame_idx < len(high_flux) else 0.0
        local_rolloff = rolloff[frame_idx] if frame_idx < len(rolloff) else 0.0
        local_zcr = zcr[frame_idx] if frame_idx < len(zcr) else 0.0

        if is_drum_aligned and local_flux < 0.25 and duration < 0.12:
            continue

        tag = "normal"
        if local_flux > 1.2 and local_rolloff > 4000.0 and not is_drum_aligned:
            tag = "pop"
        elif local_flux > 1.5 and is_drum_aligned:
            tag = "slap"
        elif duration < 0.10 and amp < 0.40 and local_zcr > 0.12:
            tag = "ghost"

        performance_layer.append((start, end, pitch_val, amp, bends, tag))

    performance_layer.sort(key=lambda x: x[0])

    for i in range(len(performance_layer) - 1):
        start, end, p, amp, bends, tag = performance_layer[i]
        next_start = performance_layer[i+1][0]
        if tag != "ghost":
            max_legato_end = max(end, next_start - 0.02)
            performance_layer[i] = (start, max_legato_end, p, amp, bends, tag)

    # --------------------------------------------------------------------------
    # PHASE 4: Tempo & Pocket Delta Alignment
    # --------------------------------------------------------------------------
    print("[Phase 4/6] Pocket & Groove Calculation...")
    tempo_val, beats = librosa.beat.beat_track(y=drums_y, sr=sr, units='time')
    bpm = float(np.atleast_1d(tempo_val)[0])
    bpm = 120.0 if bpm <= 0 else bpm

    bass_onsets = [n[0] for n in performance_layer]
    pocket_deltas = [b_onset - min(beats, key=lambda d: abs(d - b_onset))
                     for b_onset in bass_onsets if len(beats) > 0 and abs(b_onset - min(beats, key=lambda d: abs(d - b_onset))) < 0.08]

    pocket_delta = float(np.median(pocket_deltas)) if pocket_deltas else 0.0
    print(f"        Tempo: {bpm:.2f} BPM | Pocket Delta: {pocket_delta * 1000.0:+.2f} ms")

    # --------------------------------------------------------------------------
    # PHASE 5: Biomechanical Viterbi Fingering HMM
    # --------------------------------------------------------------------------
    print("[Phase 5/6] Biomechanical Fingering & Ergonomic Solver...")
    valid_pitches = [n[2] for n in performance_layer]
    lowest_pitch = min(valid_pitches) if valid_pitches else 40
    
    if lowest_pitch <= 25:
        tuning = '5_string_low_b'
    elif lowest_pitch <= 27:
        tuning = '4_string_drop_d'
    else:
        tuning = '4_string_standard'

    hmm = ErgonomicFretboardHMMSolver(tuning_type=tuning)
    fretboard_path, rakes = hmm.solve(performance_layer)

    # --------------------------------------------------------------------------
    # PHASE 6: Measure Assembly (MusicXML)
    # --------------------------------------------------------------------------
    print("[Phase 6/6] Generating Measure-Bound Notation & MIDI...")

    sec_per_quarter = 60.0 / bpm
    first_onset = performance_layer[0][0] if performance_layer else 0.0

    quantized_timeline = []
    current_q = fractions.Fraction(0, 1)

    for i, (start, end, pitch_val, amp, bends, tag) in enumerate(performance_layer):
        v_start = max(0.0, (start - first_onset) - pocket_delta)
        dur_s = max(0.05, end - start)

        start_q = fractions.Fraction(round((v_start / sec_per_quarter) * 4), 4)
        dur_q = max(fractions.Fraction(1, 4), fractions.Fraction(round((dur_s / sec_per_quarter) * 4), 4))

        if start_q < current_q:
            start_q = current_q

        if start_q > current_q:
            rest_len = start_q - current_q
            quantized_timeline.append(('rest', rest_len, None, None, None, None, None, False))
            current_q = start_q

        s_idx, f_val = fretboard_path[i] if i < len(fretboard_path) else (None, None)
        exact_midi = (hmm.strings[s_idx] + f_val) if (s_idx is not None and f_val is not None) else pitch_val
        is_rake = rakes[i] if i < len(rakes) else False

        quantized_timeline.append(('note', dur_q, exact_midi, amp, tag, s_idx, f_val, is_rake))
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
    m_capacity = fractions.Fraction(4, 1)

    curr_measure = stream.Measure(number=m_num)
    # Clef, Time Signature, and Tempo inserted ONLY in Measure 1
    curr_measure.insert(0.0, clef.BassClef())
    curr_measure.insert(0.0, meter.TimeSignature('4/4'))
    curr_measure.insert(0.0, tempo.MetronomeMark(number=round(bpm)))

    prev_note = None

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
                r = note.Rest(quarterLength=float(take_dur))
                r.voice = 1
                curr_measure.append(r)
                m_fill += take_dur
                rem_dur -= take_dur

        elif ev_type == 'note':
            _, dur_q, exact_midi, amp, tag, s_idx, f_val, is_rake = event
            rem_dur = dur_q
            is_first_piece = True

            while rem_dur > 0:
                space = m_capacity - m_fill
                if space <= 0:
                    m21_part.append(curr_measure)
                    m_num += 1
                    curr_measure = stream.Measure(number=m_num)
                    m_fill = fractions.Fraction(0, 1)
                    space = m_capacity

                take_dur = min(rem_dur, space)
                n = note.Note(pitch.Pitch(midi=exact_midi))
                n.quarterLength = float(take_dur)
                n.voice = 1

                vel = int(amp * 127)
                n.volume.velocity = min(127, max(30, vel))

                if is_first_piece:
                    if tag == "ghost":
                        n.notehead = 'cross'
                        n.articulations.append(articulations.Staccato())
                        n.articulations.append(articulations.FretHandMute())
                    elif tag == "slap":
                        n.articulations.append(articulations.StrongAccent())
                    elif tag == "pop":
                        n.articulations.append(articulations.Accent())

                    if s_idx is not None and f_val is not None:
                        n.addLyric(f"S{s_idx}:F{f_val}")

                    if is_rake and prev_note is not None:
                        try:
                            sl = spanner.Slur(prev_note, n)
                            m21_part.insert(0, sl)
                        except Exception:
                            pass

                    prev_note = n

                if rem_dur > take_dur:
                    n.tie = tie.Tie('start') if is_first_piece else tie.Tie('continue')
                else:
                    if not is_first_piece:
                        n.tie = tie.Tie('stop')

                curr_measure.append(n)
                m_fill += take_dur
                rem_dur -= take_dur
                is_first_piece = False

    # Prevent appending empty ghost trailing measure
    if len(curr_measure.notesAndRests) > 0:
        if m_fill < m_capacity and m_fill > 0:
            r = note.Rest(quarterLength=float(m_capacity - m_fill))
            r.voice = 1
            curr_measure.append(r)
        m21_part.append(curr_measure)

    m21_score.append(m21_part)

    xml_out = os.path.join(output_dir, f"{base_name}.musicxml")
    m21_score.write('musicxml', fp=xml_out)

    sanitize_and_inject_tablature(xml_out, artist_name, song_title, tuning)

    # --------------------------------------------------------------------------
    # MIDI Performance Layer
    # --------------------------------------------------------------------------
    pm = pretty_midi.PrettyMIDI()
    bass_prog = pretty_midi.instrument_name_to_program('Electric Bass (finger)')
    bass_inst = pretty_midi.Instrument(program=bass_prog, name=f"{artist_name} - {song_title}")

    for start, end, pitch_val, amp, bends, tag in performance_layer:
        vel = 40 if tag == "ghost" else min(127, max(25, int(amp * 127)))
        midi_note = pretty_midi.Note(velocity=vel, pitch=pitch_val, start=start, end=end)
        bass_inst.notes.append(midi_note)

        if bends is not None:
            bend_array = np.atleast_1d(bends)
            if bend_array.size > 0:
                t_steps = np.linspace(start, end, len(bend_array))
                for b_time, b_val in zip(t_steps, bend_array):
                    pb_val = int(np.clip(float(b_val) * 4096.0, -8192, 8191))
                    bass_inst.pitch_bends.append(pretty_midi.PitchBend(pitch=pb_val, time=float(b_time)))

    pm.instruments.append(bass_inst)
    midi_out = os.path.join(output_dir, f"{base_name}.mid")
    pm.write(midi_out)

    print(f"\nSUCCESS: Generated Humanized Transcription Assets inside {output_dir}")
    print(f" -> Notation (MusicXML): {xml_out}")
    print(f" -> Performance (MIDI):  {midi_out}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./humanbass_3.sh [--use-gpu] <path_to_stem_folder_1> ...")
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
    echo "Usage: ./humanbass_3.sh [--use-gpu] <path_to_stem_folder_1> ..."
    exit 1
fi

"$ENV_DIR/bin/python" run_pipeline.py "${ARGS[@]}"

echo "Pipeline execution finished successfully."
