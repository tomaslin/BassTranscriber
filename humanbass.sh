#!/bin/bash
# ==============================================================================
# Humanized Bass Transcription Pipeline (pYIN / MusicXML Pro Edition)
# Native Apple Silicon (M1/M2/M3) & Linux Support
# ==============================================================================
set -euo pipefail

OS_TYPE="$(uname -s)"
ARCH_TYPE="$(uname -m)"

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
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import scipy.signal as signal
from scipy.ndimage import median_filter
import librosa
import soundfile as sf
from music21 import stream, note, pitch, meter, tie, articulations, tempo, clef, instrument, metadata, key, spanner


def parse_metadata_from_path(folder_path):
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


def snap_pitch_to_scale(midi_val, key_obj, level=5):
    if key_obj is None or level >= 4:
        return midi_val

    scale_pcs = [p.pitchClass for p in key_obj.getPitches()]
    curr_pc = midi_val % 12

    if curr_pc in scale_pcs:
        return midi_val

    if level <= 1:
        distances = [((sp - curr_pc + 6) % 12 - 6, sp) for sp in scale_pcs]
        distances.sort(key=lambda x: abs(x[0]))
        return midi_val + distances[0][0]

    distances = [((sp - curr_pc + 6) % 12 - 6, sp) for sp in scale_pcs]
    distances.sort(key=lambda x: abs(x[0]))
    if abs(distances[0][0]) <= 1:
        return midi_val + distances[0][0]

    return midi_val


def get_key_aware_pitch(midi_val, key_obj):
    p = pitch.Pitch(midi=midi_val)
    if key_obj is None:
        return p

    sharps = key_obj.sharps
    if sharps > 0 and p.accidental and p.accidental.name == 'flat':
        p = p.getEnharmonic()
    elif sharps < 0 and p.accidental and p.accidental.name == 'sharp':
        p = p.getEnharmonic()

    if sharps == 4 and p.name == 'Eb': p = p.getEnharmonic()
    elif sharps == -1 and p.name == 'G#': p = p.getEnharmonic()
    elif sharps == 3 and p.name == 'Bb': p = p.getEnharmonic()
    return p


def purge_audio_artifacts(raw_notes, max_micro_rest=0.22, min_valid_duration=0.075):
    """
    FRONT-LOADED PURGE ENGINE:
    1. Removes sub-75ms transient pitch clicks/hallucations early.
    2. Merges adjacent split-note detections caused by pitch tracker flutter (<= 1 semitone).
    3. Stretches note durations over micro-rests (<220ms) directly to the next onset.
    """
    if not raw_notes:
        return []

    valid_notes = []
    for start, end, pitch_val, amp, bends in raw_notes:
        dur = end - start
        if dur < min_valid_duration and amp < 0.35:
            continue
        valid_notes.append([start, end, pitch_val, amp, bends])

    if not valid_notes:
        return []

    purged = []
    curr = valid_notes[0]

    for next_n in valid_notes[1:]:
        c_start, c_end, c_pitch, c_amp, c_bends = curr
        n_start, n_end, n_pitch, n_amp, n_bends = next_n
        gap = n_start - c_end

        # Pitch flutter (same or adjacent pitch with small gap) -> Fuse into one note
        if abs(c_pitch - n_pitch) <= 1 and gap <= max_micro_rest:
            curr[1] = n_end
            curr[2] = c_pitch
            curr[3] = max(c_amp, n_amp)

        # Micro-rest gap between distinct notes -> Extend previous note to eliminate gap
        elif 0 < gap <= max_micro_rest:
            curr[1] = n_start
            purged.append(tuple(curr))
            curr = next_n

        # Genuine musical rest (> 220ms)
        else:
            purged.append(tuple(curr))
            curr = next_n

    purged.append(tuple(curr))
    return purged


def idiomatic_rhythm_snap(raw_dur_q, level=5, is_compound=False):
    if level == 0:
        if raw_dur_q >= 3.0: return fractions.Fraction(4, 1)
        if raw_dur_q >= 1.5: return fractions.Fraction(2, 1)
        return fractions.Fraction(1, 1)
    
    if level == 1:
        if raw_dur_q >= 3.2: return fractions.Fraction(4, 1)
        if raw_dur_q >= 1.7: return fractions.Fraction(2, 1)
        if raw_dur_q >= 0.75: return fractions.Fraction(1, 1)
        return fractions.Fraction(1, 2)

    if is_compound:
        if 0.35 <= raw_dur_q <= 0.65: return fractions.Fraction(1, 2)
        elif 0.85 <= raw_dur_q <= 1.20: return fractions.Fraction(1, 1)
        elif raw_dur_q < 0.35: return fractions.Fraction(1, 4)
    else:
        if 0.38 <= raw_dur_q <= 0.65: return fractions.Fraction(1, 2)
        if 0.85 <= raw_dur_q <= 1.15: return fractions.Fraction(1, 1)
        if 0.70 <= raw_dur_q < 0.85: return fractions.Fraction(3, 4)
        if raw_dur_q < 0.38: return fractions.Fraction(1, 4)
    eighth_steps = max(1, int(round(raw_dur_q * 2)))
    return fractions.Fraction(eighth_steps, 2)


def decompose_duration_engraver_rules(dur_quarter, m_fill_offset, m_capacity, is_compound=False):
    units = []
    rem = fractions.Fraction(dur_quarter).limit_denominator(16)
    curr_offset = fractions.Fraction(m_fill_offset).limit_denominator(16)

    while rem > 0:
        space = m_capacity - curr_offset
        if space <= 0: break

        if not is_compound and m_capacity == fractions.Fraction(4, 1):
            if curr_offset < fractions.Fraction(2, 1) and (curr_offset + rem) > fractions.Fraction(2, 1):
                take = fractions.Fraction(2, 1) - curr_offset
            else:
                take = min(rem, space)
        else:
            take = min(rem, space)

        units.append(take)
        rem -= take
        curr_offset += take
        
    return units if units else [fractions.Fraction(1, 4)]


def pyin_predict_notes(audio_y, sr, conf_threshold=0.30):
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

    raw_notes, in_note, start_time, pitch_buf, conf_buf = [], False, 0.0, [], []

    for t, f, c in zip(times, f0, voiced_probs):
        if f > 0.0 and c >= conf_threshold:
            midi_p = librosa.hz_to_midi(f)
            if not in_note:
                in_note, start_time, pitch_buf, conf_buf = True, t, [midi_p], [c]
            else:
                if abs(midi_p - np.median(pitch_buf)) > 1.5:
                    if (t - start_time) >= 0.04:
                        raw_notes.append((start_time, t, int(round(np.median(pitch_buf))), float(np.mean(conf_buf)), None))
                    start_time, pitch_buf, conf_buf = t, [midi_p], [c]
                else:
                    pitch_buf.append(midi_p)
                    conf_buf.append(c)
        else:
            if in_note:
                if pitch_buf and (t - start_time) >= 0.04:
                    raw_notes.append((start_time, t, int(round(np.median(pitch_buf))), float(np.mean(conf_buf)), None))
                in_note, pitch_buf, conf_buf = False, [], []
    return raw_notes


class ErgonomicFretboardHMMSolver:
    def __init__(self, tuning_type='4_string_standard'):
        self.tuning_type = tuning_type
        if tuning_type == '5_string_low_b': self.strings = {1: 43, 2: 38, 3: 33, 4: 28, 5: 23}
        elif tuning_type == '4_string_drop_d': self.strings = {1: 43, 2: 38, 3: 33, 4: 26}
        else: self.strings = {1: 43, 2: 38, 3: 33, 4: 28}
        self.num_frets = 20

    def get_valid_positions(self, midi_pitch):
        return [(s, midi_pitch - open_p) for s, open_p in self.strings.items() if 0 <= midi_pitch - open_p <= self.num_frets]

    def solve(self, note_events):
        if not note_events: return [], [], []
        sequence_states = [
            self.get_valid_positions(n[2]) or
            self.get_valid_positions(n[2]+12) or
            self.get_valid_positions(n[2]-12) or
            [(list(self.strings.keys())[0], 0)]
            for n in note_events
        ]
        T, V, path = len(sequence_states), [{}], {}

        for state in sequence_states[0]:
            string_num, fret = state
            tag = note_events[0][5]
            box_cost = 0.0 if (1 <= fret <= 5 or fret == 0) else (fret - 5) * 1.5
            tech_cost = 10.0 if (tag == "pop" and string_num > 2) else 5.0 if (tag == "slap" and string_num < 3) else 0.0
            V[0][state], path[state] = -(box_cost + tech_cost), [state]

        for t in range(1, T):
            V.append({})
            new_path = {}
            tag = note_events[t][5]

            for c_state in sequence_states[t]:
                c_string, c_fret = c_state
                best_cost, best_prev = -float('inf'), None

                for p_state in sequence_states[t-1]:
                    if p_state not in V[t-1]: continue
                    p_string, p_fret = p_state
                    
                    fret_shift = 0.2 if (c_fret == 0 or p_fret == 0) else abs(c_fret - p_fret) * 1.2
                    string_shift = abs(c_string - p_string) * 1.5
                    high_fret_penalty = (c_fret - 5) * 2.0 if c_fret > 5 else 0.0
                    
                    tech_cost = 15.0 if tag == "pop" and c_string > 2 else 8.0 if tag == "slap" and c_string < 3 else 0.0
                    total_score = V[t-1][p_state] - (fret_shift + string_shift + high_fret_penalty + tech_cost)
                    
                    if total_score > best_cost:
                        best_cost, best_prev = total_score, p_state

                if best_prev is None and sequence_states[t-1]:
                    best_prev, best_cost = sequence_states[t-1][0], V[t-1].get(sequence_states[t-1][0], 0.0) - 10.0

                V[t][c_state], new_path[c_state] = best_cost, path.get(best_prev, [c_state]) + [c_state]
            path = new_path

        optimal_states = path.get(max(V[-1], key=V[-1].get), [sequence_states[-1][0]]) if V[-1] else [s[0] for s in sequence_states]
        rakes, legatos = [False] * T, [False] * T
        for i in range(1, T):
            dt = note_events[i][0] - note_events[i-1][1]
            if optimal_states[i][0] > optimal_states[i-1][0] and dt < 0.12: rakes[i] = True
            if optimal_states[i][0] == optimal_states[i-1][0] and abs(optimal_states[i][1] - optimal_states[i-1][1]) in [1, 2, 3] and dt < 0.04: legatos[i] = True
        return optimal_states, rakes, legatos


def consolidate_measure_notation(measure):
    elems = list(measure.notesAndRests)
    if not elems: return
    i = 0
    while i < len(elems) - 1:
        curr_el, next_el = elems[i], elems[i + 1]
        if isinstance(curr_el, note.Rest) and isinstance(next_el, note.Rest):
            curr_el.quarterLength += next_el.quarterLength
            measure.remove(next_el)
            elems.pop(i + 1)
            continue
        if (isinstance(curr_el, note.Note) and isinstance(next_el, note.Note) and curr_el.pitch.midi == next_el.pitch.midi and curr_el.tie and curr_el.tie.type in ['start', 'continue']):
            curr_el.quarterLength += next_el.quarterLength
            if next_el.tie and next_el.tie.type == 'stop': curr_el.tie = None
            elif next_el.tie and next_el.tie.type == 'continue': curr_el.tie.type = 'start'
            measure.remove(next_el)
            elems.pop(i + 1)
            continue
        i += 1


def sanitize_and_inject_tablature(xml_path, artist_name, song_title, tuning_type, level=5):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ""

    root.tag = f"{ns}score-partwise"
    part_list = root.find(f"{ns}part-list")
    if part_list is None: part_list, _ = ET.Element(f"{ns}part-list"), root.insert(0, ET.Element(f"{ns}part-list"))
    
    score_part = part_list.find(f"{ns}score-part")
    if score_part is None: score_part = ET.SubElement(part_list, f"{ns}score-part", attrib={"id": "P1"})
    else: score_part.attrib["id"] = "P1"

    part_name = score_part.find(f"{ns}part-name")
    if part_name is None: part_name = ET.SubElement(score_part, f"{ns}part-name")
    part_name.text = "Electric Bass"

    work_elem = root.find(f"{ns}work")
    if work_elem is None: work_elem = ET.Element(f"{ns}work"); root.insert(1, work_elem)
    work_title = work_elem.find(f"{ns}work-title")
    if work_title is None: work_title = ET.SubElement(work_elem, f"{ns}work-title")
    work_title.text = song_title

    ident = root.find(f"{ns}identification")
    if ident is None: ident = ET.SubElement(root, f"{ns}identification")
    creator_elem = ident.find(f"{ns}creator")
    if creator_elem is None: creator_elem = ET.SubElement(ident, f"{ns}creator", attrib={"type": "composer"})
    creator_elem.text = artist_name

    first_part = root.find(f"{ns}part")
    if first_part is not None and first_part.find(f"{ns}measure") is not None:
        first_measure = first_part.find(f"{ns}measure")
        attrs = first_measure.find(f"{ns}attributes") or ET.Element(f"{ns}attributes")
        if attrs not in first_measure: first_measure.insert(0, attrs)
        staff_details = ET.SubElement(attrs, f"{ns}staff-details")
        ET.SubElement(staff_details, f"{ns}staff-lines").text = str(5 if tuning_type == '5_string_low_b' else 4)
        
        tunings = {'5_string_low_b': [('G', 2), ('D', 2), ('A', 1), ('E', 1), ('B', 0)],
                   '4_string_drop_d': [('G', 2), ('D', 2), ('A', 1), ('D', 1)],
                   '4_string_standard': [('G', 2), ('D', 2), ('A', 1), ('E', 1)]}.get(tuning_type, [('G', 2), ('D', 2), ('A', 1), ('E', 1)])
        
        for s_idx, (step, oct_val) in enumerate(tunings, 1):
            s_tuning = ET.SubElement(staff_details, f"{ns}staff-tuning", attrib={"line": str(s_idx)})
            ET.SubElement(s_tuning, f"{ns}tuning-step").text, ET.SubElement(s_tuning, f"{ns}tuning-octave").text = step, str(oct_val)

    prev_fret_num, prev_string_num = None, None
    for part in root.findall(f"{ns}part"):
        part.attrib["id"] = "P1"
        for measure in part.findall(f"{ns}measure"):
            for dummy_rest in list(measure.findall(f"{ns}note")):
                if dummy_rest.attrib.get("print-object") == "no": measure.remove(dummy_rest)

            for note_elem in list(measure.findall(f"{ns}note")):
                if note_elem.find(f"{ns}stem") is not None: note_elem.remove(note_elem.find(f"{ns}stem"))
                for beam_el in list(note_elem.findall(f"{ns}beam")): note_elem.remove(beam_el)
                
                if 'dynamics' in note_elem.attrib: del note_elem.attrib['dynamics']

                lyric_elem = note_elem.find(f"{ns}lyric")
                string_num, fret_num = None, None
                if lyric_elem is not None:
                    text_elem = lyric_elem.find(f"{ns}text")
                    if text_elem is not None and text_elem.text and text_elem.text.startswith('S') and ':F' in text_elem.text:
                        try: string_num, fret_num = text_elem.text.split(':F')[0][1:], text_elem.text.split(':F')[1]
                        except: pass
                        note_elem.remove(lyric_elem)

                if level > 0 and string_num is not None and fret_num is not None:
                    notations = note_elem.find(f"{ns}notations") or ET.SubElement(note_elem, f"{ns}notations")
                    technical = notations.find(f"{ns}technical") or ET.SubElement(notations, f"{ns}technical")
                    
                    ET.SubElement(technical, f"{ns}string", attrib={"print-object": "no"}).text = str(string_num)
                    ET.SubElement(technical, f"{ns}fret").text = str(fret_num)

                    if level >= 3 and prev_string_num == string_num and prev_fret_num is not None and fret_num != prev_fret_num:
                        if int(fret_num) > int(prev_fret_num): ET.SubElement(technical, f"{ns}hammer-on", attrib={"type": "stop"}).text = "H"
                        elif int(fret_num) < int(prev_fret_num): ET.SubElement(technical, f"{ns}pull-off", attrib={"type": "stop"}).text = "P"
                    prev_string_num, prev_fret_num = string_num, fret_num

                notehead = note_elem.find(f"{ns}notehead")
                if notehead is not None and notehead.text == "cross":
                    if level <= 1:
                        note_elem.remove(notehead)
                    else:
                        notehead.text = "x"

    for elem in root.iter():
        if elem.text and 'stems_' in elem.text.lower(): elem.text = re.sub(r'(?i)stems_?', '', elem.text).strip()
    if ns: ET.register_namespace('', ns.strip('{}'))
    if hasattr(ET, 'indent'): ET.indent(tree, space="  ")
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)


def filter_performance_for_level(layer, level, beats, is_compound, bpm):
    if not layer: return []
    if level == 5: return list(layer)
        
    filtered = []
    beat_interval = 60.0 / bpm if bpm > 0 else 0.5
    measure_len = beat_interval * (6 if is_compound else 4)
    
    if len(beats) == 0: return list(layer)
        
    first_beat = beats[0]
    last_time = layer[-1][1]
    downbeats = np.arange(first_beat, last_time + measure_len, measure_len)
    
    if level == 0:
        for db in downbeats:
            window_notes = [n for n in layer if db - 0.20 <= n[0] < db + (beat_interval * 2)]
            if window_notes:
                root_note = min(window_notes, key=lambda x: x[2])
                extended_end = root_note[0] + (measure_len * 0.5)
                filtered.append((root_note[0], extended_end, root_note[2], root_note[3], root_note[4], "normal", root_note[6]))
        return filtered

    for start, end, pitch_val, amp, bends, tag, flux_val in layer:
        dur = end - start
        is_on_beat = min([abs(start - b) for b in beats]) < 0.15
        is_on_eighth = min([abs(start - (b + beat_interval/2)) for b in beats]) < 0.15
        
        if level == 1:
            if tag == "ghost": continue
            if dur < 0.14 and not is_on_beat: continue
            if not (is_on_beat or is_on_eighth): continue
            filtered.append((start, end, pitch_val, amp, bends, "normal", flux_val))
            
        elif level == 2:
            if tag == "ghost" and dur < 0.12: continue
            if dur < 0.08: continue
            filtered.append((start, end, pitch_val, amp, bends, "normal", flux_val))
            
        elif level == 3:
            if tag == "ghost" and dur < 0.06: continue
            filtered.append((start, end, pitch_val, amp, bends, tag, flux_val))
            
        elif level == 4:
            if tag == "ghost" and amp < 0.05: continue
            filtered.append((start, end, pitch_val, amp, bends, tag, flux_val))

    return filtered


def process_folder(stem_folder, generate_all_levels=False):
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

    print("[Phase 1/4] Filtering Audio & Estimating Key Signature...")
    bass_y, _ = librosa.load(bass_path, sr=sr, mono=True)
    drums_y, _ = librosa.load(drums_path, sr=sr, mono=True)
    bass_y, drums_y = np.asarray(bass_y, dtype=np.float32), np.asarray(drums_y, dtype=np.float32)

    detected_key = detect_key_signature(bass_y, sr)
    print(f"        Detected Key: {detected_key.name}")

    sos_low = signal.butter(4, [25 / (sr / 2), 280 / (sr / 2)], 'bandpass', output='sos')
    bass_low = np.asarray(signal.sosfiltfilt(sos_low, bass_y), dtype=np.float32)
    sos_high = signal.butter(4, 1800 / (sr / 2), 'highpass', output='sos')
    bass_high = np.asarray(signal.sosfiltfilt(sos_high, bass_y), dtype=np.float32)

    print("[Phase 2/4] Running pYIN Autocorrelation Pitch Tracking...")
    raw_pyin_notes = pyin_predict_notes(bass_low, sr, conf_threshold=0.30)
    if not raw_pyin_notes: raw_pyin_notes = pyin_predict_notes(bass_low, sr, conf_threshold=0.15)
    
    corrected_notes = []
    for start, end, midi_pitch, amp, bends in raw_pyin_notes:
        while midi_pitch > 67: midi_pitch -= 12
        while midi_pitch < 23: midi_pitch += 12
        corrected_notes.append((start, end, max(23, min(67, int(round(midi_pitch)))), float(amp), bends))

    print("[Phase 3/4] Purging Audio Artifacts & Micro-Rests...")
    purged_notes = purge_audio_artifacts(corrected_notes, max_micro_rest=0.22, min_valid_duration=0.075)

    drum_onsets = librosa.onset.onset_detect(y=drums_y, sr=sr, units='time')
    high_flux, flux_times = librosa.onset.onset_strength(y=bass_high, sr=sr), librosa.times_like(librosa.onset.onset_strength(y=bass_high, sr=sr), sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=bass_y)[0]

    performance_layer = []
    for start, end, pitch_val, amp, bends in purged_notes:
        dur = end - start
        closest_drum = min([abs(start - d) for d in drum_onsets]) if len(drum_onsets) > 0 else 999.0
        frame_idx = np.argmin(np.abs(flux_times - start))
        local_flux = high_flux[frame_idx] if frame_idx < len(high_flux) else 0.0
        local_zcr = zcr[frame_idx] if frame_idx < len(zcr) else 0.0

        if closest_drum < 0.035 and local_flux < 0.20 and dur < 0.10: continue
        tag = "pop" if local_flux > 1.4 and closest_drum >= 0.035 else "slap" if local_flux > 1.6 and closest_drum < 0.035 else "ghost" if dur < 0.10 and amp < 0.40 and local_zcr > 0.12 else "normal"
        performance_layer.append((start, end, pitch_val, amp, bends, tag, local_flux))
    
    performance_layer.sort(key=lambda x: x[0])

    tempo_val, beats = librosa.beat.beat_track(y=drums_y, sr=sr, units='time')
    bpm = 120.0 if float(np.atleast_1d(tempo_val)[0]) <= 0 else float(np.atleast_1d(tempo_val)[0])
    avg_interval = float(np.mean(np.diff(beats))) if len(beats) > 1 else 0.5
    is_compound = (avg_interval > 0.65) and (bpm < 95.0)

    time_sig_str = '12/8' if is_compound else '4/4'
    m_capacity = fractions.Fraction(6, 1) if is_compound else fractions.Fraction(4, 1)

    bass_onsets = [n[0] for n in performance_layer]
    pocket_deltas = [b_onset - min(beats, key=lambda d: abs(d - b_onset)) for b_onset in bass_onsets if len(beats) > 0 and abs(b_onset - min(beats, key=lambda d: abs(d - b_onset))) < 0.08]
    pocket_delta = float(np.median(pocket_deltas)) if pocket_deltas else 0.0

    valid_pitches = [n[2] for n in performance_layer]
    lowest_pitch = min(valid_pitches) if valid_pitches else 40
    tuning = '5_string_low_b' if lowest_pitch <= 25 else '4_string_drop_d' if lowest_pitch <= 27 else '4_string_standard'

    # Target levels: only level 5 by default, or 0..5 if --all-levels is set
    target_levels = range(6) if generate_all_levels else [5]

    for level in target_levels:
        if generate_all_levels:
            print(f"\n[Phase 4/4] Generating Notation Output (LEVEL {level})...")
            level_title = f"{song_title} (Level {level})"
            xml_out = os.path.join(output_dir, f"{base_name}_Level{level}.musicxml")
        else:
            print(f"\n[Phase 4/4] Generating Clean MusicXML Output...")
            level_title = song_title
            xml_out = os.path.join(output_dir, f"{base_name}.musicxml")

        level_layer = filter_performance_for_level(performance_layer, level, beats, is_compound, bpm)
        if not level_layer:
            print(f"        [Skipping] Level {level} yielded no notes.")
            continue

        hmm = ErgonomicFretboardHMMSolver(tuning_type=tuning)
        fretboard_path, rakes, legatos = hmm.solve(level_layer)
        
        sec_per_quarter = 60.0 / bpm
        first_onset = level_layer[0][0] if level_layer else 0.0
        
        quantized_timeline = []
        current_q = fractions.Fraction(0, 1)

        for i, (start, end, pitch_val, amp, bends, tag, flux_val) in enumerate(level_layer):
            v_start = max(0.0, (start - first_onset) - pocket_delta)
            dur_s = max(0.05, end - start)
            start_q = fractions.Fraction(int(round((v_start / sec_per_quarter) * 4)), 4)
            raw_dur_q = dur_s / sec_per_quarter

            dur_q = idiomatic_rhythm_snap(raw_dur_q, level=level, is_compound=is_compound)

            if start_q < current_q: start_q = current_q
            if start_q > current_q:
                rest_len = start_q - current_q
                quantized_timeline.append(('rest', rest_len, None, None, None, None, None, False, False))
                current_q = start_q

            s_idx, f_val = fretboard_path[i] if i < len(fretboard_path) else (None, None)
            clean_midi = snap_pitch_to_scale(pitch_val, detected_key, level=level)
            exact_midi = (hmm.strings[s_idx] + f_val) if (s_idx is not None and f_val is not None) else clean_midi

            is_rake = rakes[i] if (i < len(rakes) and level >= 3) else False
            is_legato = legatos[i] if (i < len(legatos) and level >= 2) else False

            quantized_timeline.append(('note', dur_q, exact_midi, amp, tag, s_idx, f_val, is_rake, is_legato))
            current_q += dur_q

        m21_score = stream.Score()
        m21_part = stream.Part()
        m21_part.partName = "Electric Bass"

        m21_score.metadata = metadata.Metadata()
        m21_score.metadata.title = level_title
        m21_score.metadata.composer = artist_name

        bass_inst = instrument.ElectricBass()
        bass_inst.partName = "Electric Bass"
        m21_part.insert(0.0, bass_inst)

        m_fill, m_num = fractions.Fraction(0, 1), 1
        curr_measure = stream.Measure(number=m_num)
        curr_measure.insert(0.0, clef.BassClef())
        curr_measure.insert(0.0, meter.TimeSignature(time_sig_str))
        curr_measure.insert(0.0, detected_key)
        curr_measure.insert(0.0, tempo.MetronomeMark(number=int(round(bpm))))

        prev_note_obj = None

        for event in quantized_timeline:
            ev_type = event[0]
            if ev_type == 'rest':
                rem_dur = event[1]
                while rem_dur > 0:
                    space = m_capacity - m_fill
                    if space <= 0:
                        consolidate_measure_notation(curr_measure)
                        m21_part.append(curr_measure)
                        m_num += 1
                        curr_measure, m_fill, space = stream.Measure(number=m_num), fractions.Fraction(0, 1), m_capacity

                    take_dur = min(rem_dur, space)
                    for sub_dur in decompose_duration_engraver_rules(take_dur, m_fill, m_capacity, is_compound):
                        r = note.Rest(quarterLength=float(sub_dur))
                        r.voice = 1
                        curr_measure.append(r)
                        
                    m_fill += take_dur
                    rem_dur -= take_dur

            elif ev_type == 'note':
                _, dur_q, exact_midi, amp, tag, s_idx, f_val, is_rake, is_legato = event
                rem_dur, is_first_piece = dur_q, True

                while rem_dur > 0:
                    space = m_capacity - m_fill
                    if space <= 0:
                        consolidate_measure_notation(curr_measure)
                        m21_part.append(curr_measure)
                        m_num += 1
                        curr_measure, m_fill, space = stream.Measure(number=m_num), fractions.Fraction(0, 1), m_capacity

                    take_dur = min(rem_dur, space)
                    dur_pieces = decompose_duration_engraver_rules(take_dur, m_fill, m_capacity, is_compound)

                    for p_idx, sub_dur in enumerate(dur_pieces):
                        key_pitch = get_key_aware_pitch(exact_midi, detected_key)
                        n = note.Note(key_pitch)
                        n.quarterLength = float(sub_dur)
                        n.voice = 1

                        if is_first_piece and p_idx == 0:
                            if level >= 2 and tag == "ghost":
                                n.notehead = 'cross'
                            elif level >= 3 and tag == "slap":
                                n.articulations.append(articulations.StrongAccent())
                            elif level >= 3 and tag == "pop":
                                n.articulations.append(articulations.Accent())
                                
                            if is_legato and prev_note_obj is not None and level >= 2:
                                m21_part.insert(0, spanner.Slur(prev_note_obj, n))
                            if level > 0 and s_idx is not None and f_val is not None:
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
                for sub_dur in decompose_duration_engraver_rules(m_capacity - m_fill, m_fill, m_capacity, is_compound):
                    r = note.Rest(quarterLength=float(sub_dur))
                    r.voice = 1
                    curr_measure.append(r)
            consolidate_measure_notation(curr_measure)
            m21_part.append(curr_measure)

        m21_part.makeBeams(inPlace=True)
        m21_score.append(m21_part)

        m21_score.write('musicxml', fp=xml_out)
        sanitize_and_inject_tablature(xml_out, artist_name, level_title, tuning, level=level)
        print(f"        -> Saved: {xml_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Humanized Bass Transcription Engine")
    parser.add_argument('folders', nargs='+', help="Path to stem folder(s)")
    parser.add_argument('-l', '--all-levels', action='store_true', help="Generate outputs for all 6 complexity/articulation levels (0-5). Default computes highest level only without level suffix.")

    args = parser.parse_args()

    for folder in args.folders:
        if os.path.isdir(folder):
            process_folder(folder, generate_all_levels=args.all_levels)
        else:
            print(f"Directory non-existent: {folder}")
EOF

# ------------------------------------------------------------------------------
# 4. Execution
# ------------------------------------------------------------------------------
if [ $# -eq 0 ]; then
    echo "Usage: ./humanbass.sh [-l|--all-levels] <path_to_stem_folder_1> ..."
    exit 1
fi

"$ENV_DIR/bin/python" run_pipeline.py "$@"

echo "Pipeline execution finished successfully."
