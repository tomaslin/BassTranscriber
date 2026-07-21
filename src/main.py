import os
import sys
import re
import json
import difflib
import argparse
import fractions
import numpy as np
import scipy.signal as signal
import librosa
from music21 import stream, note, meter, tempo, clef, instrument, metadata, spanner, tie, articulations
from src.audio_engine import (
    detect_key_signature,
    pyin_predict_notes,
    purge_audio_artifacts,
    snap_pitch_to_scale,
    get_key_aware_pitch
)
from src.fretboard_hmm import ErgonomicFretboardHMMSolver
from src.xml_formatter import (
    idiomatic_rhythm_snap,
    decompose_duration_engraver_rules,
    consolidate_measure_notation,
    sanitize_and_inject_tablature
)


def load_genre_configs():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "genres.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


GENRE_CONFIGS = load_genre_configs()


def resolve_genre(raw_genre):
    if not raw_genre:
        return "default", None
    
    clean_genre = raw_genre.strip().lower()
    if clean_genre in GENRE_CONFIGS:
        return clean_genre, GENRE_CONFIGS[clean_genre]

    for g in GENRE_CONFIGS:
        if g in clean_genre or clean_genre in g:
            return g, GENRE_CONFIGS[g]

    matches = difflib.get_close_matches(clean_genre, GENRE_CONFIGS.keys(), n=1, cutoff=0.3)
    if matches:
        return matches[0], GENRE_CONFIGS[matches[0]]

    return "default", None


def parse_metadata_from_path(folder_path):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    clean_name = re.sub(r'(?i)^stems_', '', folder_name).strip()
    
    parts = [p.strip() for p in clean_name.split('_') if p.strip()]
    
    genre_str, key_str, artist, title = None, None, "Unknown Artist", "Bass Track"
    
    if len(parts) >= 4:
        title = parts[-1].title()
        artist = parts[-2].title()
        key_str = parts[-3]
        genre_str = parts[-4]
    elif len(parts) == 3:
        title = parts[-1].title()
        artist = parts[-2].title()
        genre_str = parts[-3]
    elif len(parts) == 2:
        title = parts[-1].title()
        artist = parts[-2].title()
    elif len(parts) == 1:
        title = parts[0].title()

    resolved_genre_name, genre_config = resolve_genre(genre_str)

    return artist, title, clean_name, key_str, resolved_genre_name, genre_config


def filter_performance_for_level(layer, level, beats, is_compound, bpm, genre_config=None):
    if not layer: return []
    if level == 5: return list(layer)
        
    filtered = []
    beat_interval = 60.0 / bpm if bpm > 0 else 0.5
    measure_len = beat_interval * (6 if is_compound else 4)
    
    if len(beats) == 0: return list(layer)
        
    first_beat = beats[0]
    last_time = layer[-1][1]
    downbeats = np.arange(first_beat, last_time + measure_len, measure_len)
    
    ghost_enabled = genre_config["features"].get("ghost_notes", True) if genre_config else True

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
            if tag == "ghost" and (dur < 0.12 or not ghost_enabled): continue
            if dur < 0.08: continue
            filtered.append((start, end, pitch_val, amp, bends, "normal", flux_val))
            
        elif level == 3:
            if tag == "ghost" and (dur < 0.06 or not ghost_enabled): continue
            filtered.append((start, end, pitch_val, amp, bends, tag, flux_val))
            
        elif level == 4:
            if tag == "ghost" and (amp < 0.05 or not ghost_enabled): continue
            filtered.append((start, end, pitch_val, amp, bends, tag, flux_val))

    return filtered


def process_folder(stem_folder, generate_all_levels=False, custom_output_dir=None, profile='HIGH_FIDELITY', level=None):
    artist_name, song_title, base_name, parsed_key_str, resolved_genre, genre_config = parse_metadata_from_path(stem_folder)

    print(f"\n=======================================================")
    print(f"TRACK: {artist_name} - {song_title}")
    print(f"GENRE: {resolved_genre.title()} | PARSED KEY: {parsed_key_str or 'Auto-Detect'}")
    print(f"STEM FOLDER: {os.path.abspath(stem_folder)}")
    print(f"=======================================================")

    bass_path = os.path.join(stem_folder, 'bass.wav')
    drums_path = os.path.join(stem_folder, 'drums.wav')

    if not os.path.exists(bass_path):
        print(f"SKIPPED: Missing bass.wav")
        return

    if not os.path.exists(drums_path):
        print(f"[INFO] No drums.wav detected. Falling back to bass track for beat estimation.")
        drums_path = bass_path

    if custom_output_dir:
        output_dir = custom_output_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(script_dir), 'output_bass')
    os.makedirs(output_dir, exist_ok=True)

    sr = 22050

    print("[Phase 1/4] Filtering Audio & Estimating Key Signature...")
    bass_y, _ = librosa.load(bass_path, sr=sr, mono=True)
    drums_y, _ = librosa.load(drums_path, sr=sr, mono=True)
    bass_y, drums_y = np.asarray(bass_y, dtype=np.float32), np.asarray(drums_y, dtype=np.float32)

    detected_key, is_parsed = detect_key_signature(bass_y, sr, parsed_key=parsed_key_str)
    
    if is_parsed:
        print(f"        Using Parsed Key: {detected_key.name}")
    else:
        print(f"        Auto-Detected Key (DSP Chroma): {detected_key.name}")

    sos_low = signal.butter(4, [25 / (sr / 2), 280 / (sr / 2)], 'bandpass', output='sos')
    bass_low = np.asarray(signal.sosfiltfilt(sos_low, bass_y), dtype=np.float32)
    sos_high = signal.butter(4, 1800 / (sr / 2), 'highpass', output='sos')
    bass_high = np.asarray(signal.sosfiltfilt(sos_high, bass_y), dtype=np.float32)

    print("[Phase 2/4] Running pYIN Autocorrelation Pitch Tracking...")
    raw_pyin_notes = pyin_predict_notes(bass_low, sr, conf_threshold=0.30)
    if not raw_pyin_notes: raw_pyin_notes = pyin_predict_notes(bass_low, sr, conf_threshold=0.15)
    
    corrected_notes = []
    for start, end, midi_pitch, amp, bends in raw_pyin_notes:
        # Enforce folding to electric bass octave range (MIDI 23-67)
        while midi_pitch > 67: midi_pitch -= 12
        while midi_pitch < 23: midi_pitch += 12
        corrected_notes.append((start, end, max(23, min(67, int(round(midi_pitch)))), float(amp), bends))

    print("[Phase 3/4] Purging Audio Artifacts & Micro-Rests...")
    purged_notes = purge_audio_artifacts(corrected_notes, max_micro_rest=0.22, min_valid_duration=0.075)

    drum_onsets = librosa.onset.onset_detect(y=drums_y, sr=sr, units='time')
    high_flux = librosa.onset.onset_strength(y=bass_high, sr=sr)
    flux_times = librosa.times_like(high_flux, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=bass_y)[0]

    slap_pop_enabled = genre_config["features"].get("slap_pop", True) if genre_config else True

    performance_layer = []
    for start, end, pitch_val, amp, bends in purged_notes:
        dur = end - start
        closest_drum = min([abs(start - d) for d in drum_onsets]) if len(drum_onsets) > 0 else 999.0
        frame_idx = np.argmin(np.abs(flux_times - start))
        local_flux = high_flux[frame_idx] if frame_idx < len(high_flux) else 0.0
        local_zcr = zcr[frame_idx] if frame_idx < len(zcr) else 0.0

        if closest_drum < 0.035 and local_flux < 0.20 and dur < 0.10: continue

        if slap_pop_enabled:
            tag = "pop" if local_flux > 1.4 and closest_drum >= 0.035 else "slap" if local_flux > 1.6 and closest_drum < 0.035 else "ghost" if dur < 0.10 and amp < 0.40 and local_zcr > 0.12 else "normal"
        else:
            tag = "ghost" if dur < 0.10 and amp < 0.40 and local_zcr > 0.12 else "normal"

        performance_layer.append((start, end, pitch_val, amp, bends, tag, local_flux))
    
    performance_layer.sort(key=lambda x: x[0])

    tempo_val, beats = librosa.beat.beat_track(y=drums_y, sr=sr, units='time')
    raw_bpm = float(np.atleast_1d(tempo_val)[0])
    # Integer tempo locking to eliminate floating-point sub-BPM drift
    bpm = float(int(round(raw_bpm))) if raw_bpm > 0 else 120.0
    avg_interval = float(np.mean(np.diff(beats))) if len(beats) > 1 else 0.5

    is_compound = (avg_interval > 0.65) and (bpm < 95.0)
    if genre_config and "compound_meter" in genre_config["features"]:
        is_compound = is_compound or genre_config["features"]["compound_meter"]

    time_sig_str = '12/8' if is_compound else '4/4'
    m_capacity = fractions.Fraction(6, 1) if is_compound else fractions.Fraction(4, 1)

    bass_onsets = [n[0] for n in performance_layer]
    pocket_deltas = [b_onset - min(beats, key=lambda d: abs(d - b_onset)) for b_onset in bass_onsets if len(beats) > 0 and abs(b_onset - min(beats, key=lambda d: abs(d - b_onset))) < 0.08]
    pocket_delta = float(np.median(pocket_deltas)) if pocket_deltas else 0.0

    valid_pitches = [n[2] for n in performance_layer]
    lowest_pitch = min(valid_pitches) if valid_pitches else 40
    tuning = '5_string_low_b' if lowest_pitch <= 25 else '4_string_drop_d' if lowest_pitch <= 27 else '4_string_standard'

    if level is not None:
        selected_level = level
    elif profile == 'FAST_DRAFT':
        selected_level = 2
    elif profile == 'MIDI_TABS':
        selected_level = 4
    else:
        selected_level = 5

    target_levels = range(6) if generate_all_levels else [selected_level]

    for level in target_levels:
        if generate_all_levels:
            print(f"\n[Phase 4/4] Generating Notation Output (LEVEL {level})...")
            level_title = f"{song_title} (Level {level})"
            xml_out = os.path.join(output_dir, f"{base_name}_Level{level}.musicxml")
        else:
            print(f"\n[Phase 4/4] Generating Clean MusicXML Output...")
            level_title = song_title
            xml_out = os.path.join(output_dir, f"{base_name}.musicxml")

        level_layer = filter_performance_for_level(performance_layer, level, beats, is_compound, bpm, genre_config=genre_config)
        if not level_layer:
            print(f"        [Skipping] Level {level} yielded no notes.")
            continue

        # Pitch-snapping execution order fix: Snap pitches BEFORE running HMM solver
        snapped_layer = []
        for start, end, p_val, amp, bends, tag, flux in level_layer:
            s_pitch = snap_pitch_to_scale(p_val, detected_key, level=level)
            snapped_layer.append((start, end, s_pitch, amp, bends, tag, flux))

        hmm = ErgonomicFretboardHMMSolver(tuning_type=tuning)
        fretboard_path, rakes, legatos = hmm.solve(snapped_layer)
        
        sec_per_quarter = 60.0 / bpm
        first_onset = snapped_layer[0][0] if snapped_layer else 0.0
        
        quantized_timeline = []
        current_q = fractions.Fraction(0, 1)

        for i, (start, end, pitch_val, amp, bends, tag, flux_val) in enumerate(snapped_layer):
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
            exact_midi = (hmm.strings[s_idx] + f_val) if (s_idx is not None and f_val is not None) else pitch_val

            if s_idx is None or f_val is None:
                valid_pos = hmm.get_valid_positions(exact_midi)
                s_idx, f_val = valid_pos[0] if valid_pos else (4, 0)

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
                    # Centered whole-measure rest formatting rule
                    if take_dur == m_capacity and m_fill == 0:
                        r = note.Rest()
                        r.quarterLength = float(m_capacity)
                        r.fullMeasure = True
                        r.voice = 1
                        curr_measure.append(r)
                    else:
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
                        # Dynamic MIDI Velocity Injection (25 - 127)
                        n.volume.velocity = int(np.clip((amp if amp is not None else 0.8) * 127, 25, 127))

                        if s_idx is not None and f_val is not None:
                            n.addLyric(f"S{s_idx}:F{f_val}")

                        if is_first_piece and p_idx == 0:
                            if level >= 2 and tag == "ghost":
                                n.notehead = 'cross'
                            elif level >= 3 and tag == "slap":
                                n.articulations.append(articulations.StrongAccent())
                            elif level >= 3 and tag == "pop":
                                n.articulations.append(articulations.Accent())
                                
                            # Re-anchored slur insertion to top-level stream (m21_part)
                            if is_legato and prev_note_obj is not None and level >= 2:
                                m21_part.insert(0, spanner.Slur([prev_note_obj, n]))
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

        # Retain automated beam calculations; omit m21_part.quantize() to preserve exact engraver durations
        m21_part.makeBeams(inPlace=True)
        m21_score.append(m21_part)

        m21_score.write('musicxml', fp=xml_out)
        sanitize_and_inject_tablature(xml_out, artist_name, level_title, tuning, level=level)
        print(f"        -> Saved: {xml_out}")


def main():
    parser = argparse.ArgumentParser(description="Humanized Bass Transcription Engine")
    parser.add_argument('folders', nargs='+', help="Path to stem folder(s)")
    parser.add_argument('-a', '--all-levels', action='store_true', help="Generate outputs for all 6 complexity/articulation levels (0-5). Default computes highest level only without level suffix.")
    parser.add_argument('-o', '--output-dir', help="Custom output directory for generated files")
    parser.add_argument('-p', '--profile', default='HIGH_FIDELITY', help="Transcription profile (MIDI_TABS, HIGH_FIDELITY, FAST_DRAFT)")
    parser.add_argument('--level', type=int, help="Specific complexity level (0-5) to override profile default")

    args = parser.parse_args()

    for folder in args.folders:
        if os.path.isdir(folder):
            process_folder(
                folder,
                generate_all_levels=args.all_levels,
                custom_output_dir=args.output_dir,
                profile=args.profile,
                level=args.level
            )
        else:
            print(f"Directory non-existent: {folder}")


if __name__ == "__main__":
    main()
