#!/bin/bash
# ==============================================================================
# Dual-Layer Bass Transcription Pipeline (M1 Apple Silicon Optimized)
# Full Production Architecture Implementing the Complete Master Specification.
# Stack: numpy==1.26.4 scipy==1.14.1 soundfile==0.12.1 soxr==0.3.7 librosa>=0.10.2
#        music21==9.1.0 pretty_midi==0.2.10 basic-pitch>=0.4.0 resampy==0.4.2
#        setuptools<82 tensorflow-macos<2.16.0 tensorflow-metal==1.1.0
# ==============================================================================
set -euo pipefail

## 1. System Architecture Verification
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "FATAL: This pipeline requires an Apple Silicon (M1/M2/M3) architecture."
    exit 1
fi
echo "[OK] Apple Silicon (arm64) architecture detected."

## 2. Homebrew & Python 3.11 Provisioning
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

if ! brew ls --versions python@3.11 &> /dev/null; then
    echo "Installing Python 3.11..."
    brew install python@3.11
fi

## 3. Environment Setup
ENV_DIR="${PWD}/.bass_pipeline_env"
PYTHON_BIN="/opt/homebrew/opt/python@3.11/bin/python3.11"

if [ ! -d "$ENV_DIR" ]; then
    echo "Provisioning Python 3.11 isolated environment..."
    "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

source "$ENV_DIR/bin/activate"

echo "Installing strict ML dependency stack..."
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
    "resampy==0.4.2" \
    "tensorflow-macos<2.16.0" \
    "tensorflow-metal==1.1.0"

## 4. Production Python Engine Generation
cat << 'EOF' > run_pipeline.py
import os
import sys
import math
import numpy as np
import scipy.signal as signal
import librosa
import soundfile as sf
import pretty_midi
from music21 import stream, note, pitch, meter, tie, articulations, spanner

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH


class FretboardViterbiHMMSolver:
    """
    Probabilistic Fretboard State Machine implementing the Viterbi Algorithm.
    Evaluates: P(S_{1:T} | O_{1:T}) \propto \prod P(O_t | S_t) * P(S_t | S_{t-1})
    """
    def __init__(self, tuning_type='4_string_standard'):
        if tuning_type == '5_string_low_b':
            # 5:B0(23), 4:E1(28), 3:A1(33), 2:D2(38), 1:G2(43)
            self.strings = {1: 43, 2: 38, 3: 33, 4: 28, 5: 23}
        elif tuning_type == '4_string_drop_d':
            # 4:D1(26), 3:A1(33), 2:D2(38), 1:G2(43)
            self.strings = {1: 43, 2: 38, 3: 33, 4: 26}
        else:
            # 4:E1(28), 3:A1(33), 2:D2(38), 1:G2(43)
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

        # Candidate state sequence generation
        sequence_states = []
        for n in note_events:
            p = n[2]
            valid = self.get_valid_positions(p)
            if not valid:
                # Transpose pitch if out of physical bounds
                valid = self.get_valid_positions(p + 12) or self.get_valid_positions(p - 12)
            if not valid:
                valid = [(list(self.strings.keys())[0], 0)]
            sequence_states.append(valid)

        T = len(sequence_states)
        V = [{}]
        path = {}

        # Initial state emission probabilities
        for state in sequence_states[0]:
            string_num, fret = state
            fret_penalty = 0.0 if fret == 0 else abs(fret - 5) * 0.2
            V[0][state] = -fret_penalty
            path[state] = [state]

        # Dynamic programming Viterbi recursion
        for t in range(1, T):
            V.append({})
            new_path = {}
            for curr_state in sequence_states[t]:
                (c_string, c_fret) = curr_state
                best_cost = -float('inf')
                best_prev = None

                for prev_state in sequence_states[t-1]:
                    (p_string, p_fret) = prev_state

                    fret_delta = abs(c_fret - p_fret) if (c_fret > 0 and p_fret > 0) else 0
                    string_delta = abs(c_string - p_string)

                    # Biomechanical transition penalties
                    hand_shift_penalty = fret_delta * 1.8
                    string_jump_penalty = string_delta * 0.9
                    high_fret_penalty = (c_fret / 12.0) ** 2

                    transition_score = -(hand_shift_penalty + string_jump_penalty + high_fret_penalty)
                    total_score = V[t-1][prev_state] + transition_score

                    if total_score > best_cost:
                        best_cost = total_score
                        best_prev = prev_state

                V[t][curr_state] = best_cost
                new_path[curr_state] = path[best_prev] + [curr_state]

            path = new_path

        best_final_state = max(V[-1], key=V[-1].get)
        optimal_states = path[best_final_state]

        # Detect Rakes: Consecutive notes crossing to physically higher string
        rakes = [False] * T
        for i in range(1, T):
            prev_string, prev_fret = optimal_states[i-1]
            curr_string, curr_fret = optimal_states[i]
            dt = note_events[i][0] - note_events[i-1][1]
            
            if curr_string > prev_string and dt < 0.15:
                rakes[i] = True

        return optimal_states, rakes


def process_folder(stem_folder):
    print(f"\n=======================================================")
    print(f"PROCESSING STEM SET: {os.path.abspath(stem_folder)}")
    print(f"=======================================================")

    bass_path = os.path.join(stem_folder, 'bass.wav')
    drums_path = os.path.join(stem_folder, 'drums.wav')
    others_path = os.path.join(stem_folder, 'others.wav')
    guitar_path = os.path.join(stem_folder, 'guitar.wav')
    piano_path = os.path.join(stem_folder, 'piano.wav')
    voice_path = os.path.join(stem_folder, 'voice.wav')

    if not os.path.exists(bass_path) or not os.path.exists(drums_path):
        print(f"SKIPPED: Missing bass.wav or drums.wav in {stem_folder}")
        return

    # Derive base filename by stripping 'stems_' prefix from the stem directory name
    folder_name = os.path.basename(os.path.normpath(stem_folder))
    if folder_name.startswith('stems_'):
        base_name = folder_name[6:]
    else:
        base_name = folder_name

    # Output directory setup (top-level output_bass folder relative to execution root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output_bass')
    os.makedirs(output_dir, exist_ok=True)

    sr = 22050

    # --------------------------------------------------------------------------
    # PHASE 1: Ingestion & Safeguarded Stem Separation (Audio Layer)
    # --------------------------------------------------------------------------
    print("[Phase 1/6] Ingesting Stems, Constructing MRC, & Banding Filters...")
    bass_y, _ = librosa.load(bass_path, sr=sr, mono=True)
    drums_y, _ = librosa.load(drums_path, sr=sr, mono=True)
    
    # Synthesize Master Reality Check (MRC) Mix
    mrc_y = np.copy(bass_y) + np.copy(drums_y)
    for extra_path in [others_path, guitar_path, piano_path, voice_path]:
        if os.path.exists(extra_path):
            ey, _ = librosa.load(extra_path, sr=sr, mono=True)
            min_l = min(len(mrc_y), len(ey))
            mrc_y[:min_l] += ey[:min_l]

    mrc_max = np.max(np.abs(mrc_y))
    if mrc_max > 0:
        mrc_y = mrc_y / mrc_max

    # 4th-Order Butterworth Banding
    b_low, a_low = signal.butter(4, [20 / (sr / 2), 500 / (sr / 2)], 'bandpass')
    bass_low = signal.filtfilt(b_low, a_low, bass_y)

    b_high, a_high = signal.butter(4, 2000 / (sr / 2), 'highpass')
    bass_high = signal.filtfilt(b_high, a_high, bass_y)

    # --------------------------------------------------------------------------
    # PHASE 2: Elastic Pitch & Event Detection (Performance Layer)
    # --------------------------------------------------------------------------
    print("[Phase 2/6] Spotify Basic-Pitch ML Inference & Sub-Bass YIN Verification...")
    
    temp_low_path = os.path.join(output_dir, f'_temp_{base_name}_low_bass.wav')
    sf.write(temp_low_path, bass_low, sr)

    model_output, midi_data, note_events = predict(
        temp_low_path,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=0.5,
        frame_threshold=0.3
    )
    if os.path.exists(temp_low_path):
        os.remove(temp_low_path)

    # Isolated Sub-Bass YIN Fundamental Verification (18Hz–200Hz)
    hop_length = 512
    f0 = librosa.yin(bass_low, fmin=18, fmax=200, sr=sr, frame_length=4096, hop_length=hop_length)

    corrected_notes = []
    for note_item in note_events:
        start, end, midi_pitch, amp = note_item[0], note_item[1], note_item[2], note_item[3]
        bends = note_item[4] if len(note_item) > 4 else None
        
        start_frame = librosa.time_to_frames(start, sr=sr, hop_length=hop_length)
        end_frame = librosa.time_to_frames(end, sr=sr, hop_length=hop_length)
        
        if start_frame < end_frame and end_frame < len(f0):
            f0_slice = f0[start_frame:end_frame]
            valid_f0 = f0_slice[~np.isnan(f0_slice)]
            if len(valid_f0) > 0:
                median_hz = np.median(valid_f0)
                yin_midi = float(librosa.hz_to_midi(median_hz))
                
                # Safeguard 2: Overriding Neural Network Octave Hallucination
                if (midi_pitch - yin_midi) > 8.0:
                    midi_pitch -= 12

        corrected_notes.append((start, end, int(round(midi_pitch)), float(amp), bends))

    # --------------------------------------------------------------------------
    # PHASE 3: Acoustic Artifact Rejection & Articulation Tagging
    # --------------------------------------------------------------------------
    print("[Phase 3/6] Cross-Correlating Stems for Demucs Artifact Rejection...")
    
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

        # Safeguard 3: Cross-Correlated Demucs Bleed Rejection
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

    # --------------------------------------------------------------------------
    # PHASE 4: The Pocket Analysis (Groove Relationship Mapping)
    # --------------------------------------------------------------------------
    print("[Phase 4/6] Drum Gravity Grid Mapping & Pocket Delta Calculation...")
    
    tempo, beats = librosa.beat.beat_track(y=drums_y, sr=sr, units='time')
    bpm = float(np.atleast_1d(tempo)[0])
    bpm = 120.0 if bpm <= 0 else bpm

    bass_onsets = [n[0] for n in performance_layer]
    pocket_deltas = []
    
    for b_onset in bass_onsets:
        if len(beats) > 0:
            closest_beat = min(beats, key=lambda d: abs(d - b_onset))
            delta = b_onset - closest_beat
            if abs(delta) < 0.08:
                pocket_deltas.append(delta)

    pocket_delta = float(np.median(pocket_deltas)) if pocket_deltas else 0.0
    print(f"        Detected Tempo: {bpm:.2f} BPM | Pocket Delta: {pocket_delta * 1000.0:+.2f} ms")

    # --------------------------------------------------------------------------
    # PHASE 5: Probabilistic Fretboard State Machine (Viterbi HMM)
    # --------------------------------------------------------------------------
    print("[Phase 5/6] Executing Ergonomic Viterbi HMM for Fingering & Rakes...")
    lowest_pitch = min([n[2] for n in performance_layer]) if performance_layer else 40
    
    if lowest_pitch <= 24:
        tuning = '5_string_low_b'
    elif lowest_pitch <= 26:
        tuning = '4_string_drop_d'
    else:
        tuning = '4_string_standard'

    hmm = FretboardViterbiHMMSolver(tuning_type=tuning)
    fretboard_path, rakes = hmm.solve(performance_layer)

    # --------------------------------------------------------------------------
    # PHASE 6: Dual-Layer Instantiation & Post-Generation Validation
    # --------------------------------------------------------------------------
    print("[Phase 6/6] Instantiating Parallel Performance & Notation Layers...")

    # --------------------------------------------------------------------------
    # PATH A: The Notation Layer (MusicXML for the Eyes)
    # --------------------------------------------------------------------------
    pruned_notes = []
    for i, item in enumerate(performance_layer):
        start, end, p, amp, bends, tag = item
        if i < len(performance_layer) - 1:
            next_start = performance_layer[i+1][0]
            next_pitch = performance_layer[i+1][2]
            
            if end > next_start:
                end = next_start
                
            if next_pitch == p and (next_start - end) < 0.040:
                end = performance_layer[i+1][1]
                
        pruned_notes.append((start, end, p, amp, bends, tag))

    m21_part = stream.Part()
    m21_part.append(meter.TimeSignature('4/4'))

    sec_per_quarter = 60.0 / bpm
    last_end_q = 0.0

    for i, (start, end, pitch_val, amp, bends, tag) in enumerate(pruned_notes):
        visual_start = max(0.0, start - pocket_delta)
        duration = max(0.05, end - start)

        start_q = round((visual_start / sec_per_quarter) * 4) / 4.0
        dur_q = max(0.25, round((duration / sec_per_quarter) * 4) / 4.0)

        gap = start_q - last_end_q
        if gap >= 0.25:
            r = note.Rest()
            r.quarterLength = gap
            m21_part.append(r)
            last_end_q = start_q

        if start_q < last_end_q:
            continue

        n = note.Note(pitch.Pitch(midi=pitch_val))
        n.quarterLength = dur_q

        if tag == "ghost":
            n.notehead = 'cross'
            n.articulations.append(articulations.FretHandMute())
        elif tag == "slap":
            n.articulations.append(articulations.StrongAccent())
        elif tag == "pop":
            n.articulations.append(articulations.SnapPizzicato())

        if i < len(fretboard_path):
            s_idx, f_val = fretboard_path[i]
            n.addLyric(f"S{s_idx}:F{f_val}")

        if i < len(rakes) and rakes[i]:
            slur = spanner.Slur()
            m21_part.append(slur)

        m21_part.append(n)
        last_end_q = start_q + dur_q

    m21_score = stream.Score()
    m21_score.append(m21_part)
    m21_validated = m21_score.makeNotation()

    xml_out = os.path.join(output_dir, f"{base_name}.musicxml")
    m21_validated.write('musicxml', fp=xml_out)

    # --------------------------------------------------------------------------
    # PATH B: The Performance Layer (MIDI for the Ears)
    # --------------------------------------------------------------------------
    pm = pretty_midi.PrettyMIDI()
    bass_program = pretty_midi.instrument_name_to_program('Electric Bass (finger)')
    bass_inst = pretty_midi.Instrument(program=bass_program)

    for start, end, pitch_val, amp, bends, tag in performance_layer:
        vel = 40 if tag == "ghost" else min(127, max(25, int(amp * 127)))
        midi_note = pretty_midi.Note(
            velocity=vel,
            pitch=pitch_val,
            start=start,
            end=end
        )
        bass_inst.notes.append(midi_note)

        if bends is not None:
            bend_array = np.atleast_1d(bends)
            if bend_array.size > 0:
                t_steps = np.linspace(start, end, len(bend_array))
                for b_time, b_val in zip(t_steps, bend_array):
                    pb_val = int(np.clip(float(b_val) * 4096.0, -8192, 8191))
                    pb = pretty_midi.PitchBend(pitch=pb_val, time=float(b_time))
                    bass_inst.pitch_bends.append(pb)

    pm.instruments.append(bass_inst)
    midi_out = os.path.join(output_dir, f"{base_name}.mid")
    pm.write(midi_out)

    print(f"\nSUCCESS: Generated dual-layer transcription assets inside {output_dir}")
    print(f" -> Notation Layer:    {xml_out}")
    print(f" -> Performance Layer: {midi_out}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./humanbass.sh <path_to_stem_folder_1> [<path_to_stem_folder_2> ...]")
        sys.exit(1)

    for folder in sys.argv[1:]:
        if os.path.isdir(folder):
            process_folder(folder)
        else:
            print(f"Directory non-existent: {folder}")
EOF

## 5. Execution
"$ENV_DIR/bin/python" run_pipeline.py "$@"

echo "Pipeline execution finished successfully."
