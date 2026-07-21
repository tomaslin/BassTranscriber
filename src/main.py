import os
import re
import json
import difflib
import argparse
import fractions
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
import librosa
from music21 import stream, note, meter, tempo, clef, instrument, metadata, spanner, tie, articulations

from src.audio_engine import (
    estimate_master_tuning,
    detect_key_signature,
    pyin_predict_notes,
    cross_stem_bleed_filter,
    purge_audio_artifacts,
    snap_pitch_to_scale,
    get_key_aware_pitch,
    estimate_beat_grid
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

    clean = raw_genre.strip().lower()
    if clean in GENRE_CONFIGS:
        return clean, GENRE_CONFIGS[clean]

    for g in GENRE_CONFIGS:
        if g in clean or clean in g:
            return g, GENRE_CONFIGS[g]

    matches = difflib.get_close_matches(clean, GENRE_CONFIGS.keys(), n=1, cutoff=0.3)
    return (matches[0], GENRE_CONFIGS[matches[0]]) if matches else ("default", None)

def parse_metadata_from_path(folder_path):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    clean_name = re.sub(r'(?i)^stems_', '', folder_name).strip()

    artist, title, genre_str, key_str = "Unknown Artist", "Bass Track", None, None

    if ' - ' in clean_name:
        left, right = clean_name.rsplit(' - ', 1)
        title = right.replace('_', ' ').strip().title()
        parts = left.split('_')

        if parts and resolve_genre(parts[0])[1]:
            genre_str, parts = parts[0], parts[1:]
        if parts and re.match(r'^(?i)[a-g][#b\-]?(_?(minor|major|min|maj|m))?$', parts[0]):
            key_str, parts = parts[0], parts[1:]

        artist = " ".join(parts).strip().title() or "Unknown Artist"
    else:
        parts = [p.strip() for p in clean_name.split('_') if p.strip()]
        if len(parts) >= 2:
            artist, title = parts[0].title(), " ".join(parts[1:]).title()
        elif parts:
            title = parts[0].title()

    resolved_genre_name, genre_config = resolve_genre(genre_str)
    return artist, title, clean_name, key_str, resolved_genre_name, genre_config

def get_closest_value(target, array):
    if len(array) == 0:
        return None
    idx = np.searchsorted(array, target)
    if idx == 0: return float(array[0])
    if idx == len(array): return float(array[-1])
    left, right = float(array[idx - 1]), float(array[idx])
    return left if abs(target - left) < abs(target - right) else right

def filter_performance_for_level(layer, level, beats, is_compound, bpm, genre_config=None):
    if not layer: return []
    if level == 5: return list(layer)

    filtered = []
    beat_interval = 60.0 / bpm if bpm > 0 else 0.5
    measure_len = beat_interval * (6 if is_compound else 4)

    if len(beats) == 0: return list(layer)

    # Establish grids for alignment checking
    downbeats = np.arange(beats[0], layer[-1][1] + measure_len, measure_len) if layer[-1][1] >= beats[0] else np.array([])
    half_measure_beats = np.arange(beats[0] + (measure_len / 2), layer[-1][1] + measure_len, measure_len) if layer[-1][1] >= beats[0] else np.array([])
    eighth_beats = [b + beat_interval / 2 for b in beats]
    
    ghost_enabled = genre_config["features"].get("ghost_notes", True) if genre_config else True

    # Level 0/1: Absolute anchors (Beats 1 & 3)
    if level <= 1:
        target_beats = np.sort(np.concatenate((downbeats, half_measure_beats)))
        for tb in target_beats:
            window_notes = [n for n in layer if tb - 0.20 <= n[0] < tb + (beat_interval * 2)]
            if window_notes:
                root_note = min(window_notes, key=lambda x: x[2])
                end_time = root_note[0] + (measure_len * 0.5)
                filtered.append((root_note[0], end_time, root_note[2], root_note[3], root_note[4], "normal", root_note[6]))
        return filtered

    # Higher Levels: Iterate over the track and evaluate syncopation
    for start, end, pitch_val, amp, bends, tag, flux_val in layer:
        dur = end - start
        c_beat = get_closest_value(start, beats)
        is_on_beat = abs(start - c_beat) < 0.15 if c_beat is not None else False
        c_eighth = get_closest_value(start, eighth_beats)
        is_on_eighth = abs(start - c_eighth) < 0.15 if c_eighth is not None else False

        if level == 2:
            # Quarter notes and on-the-beat 8ths. Strip complex syncopation and ghost notes.
            if tag != "ghost" and (is_on_beat or (is_on_eighth and dur >= 0.20)):
                filtered.append((start, end, pitch_val, amp, bends, "normal", flux_val))
                
        elif level == 3:
            # Reintroduce syncopation and offbeats. Suppress ultra-fast runs and ghost notes.
            if tag != "ghost" and dur >= 0.12:
                filtered.append((start, end, pitch_val, amp, bends, tag, flux_val))
                
        elif level == 4:
            # Full recorded line. Keep fast runs and fills. Suppress micro-nuances if disabled.
            if not (tag == "ghost" and not ghost_enabled):
                filtered.append((start, end, pitch_val, amp, bends, tag, flux_val))

    return filtered

def process_folder(stem_folder, generate_all_levels=False, custom_output_dir=None, profile='HIGH_FIDELITY', level=5, use_gpu=False):
    artist_name, song_title, _, parsed_key_str, resolved_genre, genre_config = parse_metadata_from_path(stem_folder)
    clean_filename = re.sub(r'[\\/*?:"<>|]', "", f"{artist_name} - {song_title}").strip()

    print(f"[Processing] {artist_name} - {song_title}")

    bass_path = os.path.join(stem_folder, 'bass.wav')
    drums_path = os.path.join(stem_folder, 'drums.wav') if os.path.exists(os.path.join(stem_folder, 'drums.wav')) else bass_path

    if not os.path.exists(bass_path):
        print(f"Skipped: Missing bass.wav in {stem_folder}")
        return

    output_dir = custom_output_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output_bass')
    os.makedirs(output_dir, exist_ok=True)

    sr = 22050
    bass_y, _ = librosa.load(bass_path, sr=sr, mono=True)
    drums_y, _ = librosa.load(drums_path, sr=sr, mono=True)

    stem_dict = {'bass': bass_y, 'drums': drums_y}
    for name in ['guitar', 'piano', 'vocals', 'other']:
        p = os.path.join(stem_folder, f'{name}.wav')
        if os.path.exists(p):
            stem_dict[name], _ = librosa.load(p, sr=sr, mono=True)

    tuning_offset = estimate_master_tuning(bass_y, sr)
    detected_key, _ = detect_key_signature(bass_y, sr, parsed_key=parsed_key_str)

    sos_low = signal.butter(4, [25 / (sr / 2), 280 / (sr / 2)], 'bandpass', output='sos')
    bass_low = signal.sosfiltfilt(sos_low, bass_y)
    sos_high = signal.butter(4, 1800 / (sr / 2), 'highpass', output='sos')
    bass_high = signal.sosfiltfilt(sos_high, bass_y)

    raw_pyin_notes = pyin_predict_notes(bass_low, sr, conf_threshold=0.30, tuning_offset=tuning_offset)
    corrected_notes = [(s, e, max(28, min(67, int(round(p)))), a, b) for s, e, p, a, b in raw_pyin_notes]
    verified_notes = cross_stem_bleed_filter(corrected_notes, stem_dict, sr=sr)
    purged_notes = purge_audio_artifacts(verified_notes, bass_audio=bass_y, sr=sr)

    beat_times, instant_bpms = estimate_beat_grid(drums_y, sr)
    bpm = float(np.median(instant_bpms)) if len(instant_bpms) > 0 else 120.0
    is_compound = (genre_config and genre_config["features"].get("compound_meter")) or (bpm < 95.0)

    performance_layer = [(s, e, p, a, b, "normal", 0.0) for s, e, p, a, b in purged_notes]

    # Level constraint check: Default to 5 (most difficult)
    if isinstance(level, int) and 0 <= level <= 5:
        selected_level = level
    else:
        selected_level = 5

    target_levels = range(6) if generate_all_levels else [selected_level]
    time_to_beats_mapper = interp1d(beat_times, np.arange(len(beat_times)), kind='linear', fill_value="extrapolate") if len(beat_times) > 1 else lambda t: 0.0

    for target_level in target_levels:
        xml_out = os.path.join(output_dir, f"{clean_filename}_Level{target_level}.musicxml" if generate_all_levels else f"{clean_filename}.musicxml")
        level_layer = filter_performance_for_level(performance_layer, target_level, beat_times, is_compound, bpm, genre_config=genre_config)

        if not level_layer:
            continue

        snapped_layer = [(s, e, snap_pitch_to_scale(p, detected_key, level=target_level), a, b, t, fl) for s, e, p, a, b, t, fl in level_layer]
        hmm = ErgonomicFretboardHMMSolver(tuning_type='4_string_standard')
        fretboard_path, rakes, legatos, _ = hmm.solve(snapped_layer)

        m21_score = stream.Score()
        m21_part = stream.Part(id="P1")
        m21_part.insert(0.0, instrument.ElectricBass())

        m21_score.metadata = metadata.Metadata()
        m21_score.metadata.title, m21_score.metadata.composer = song_title, artist_name

        curr_measure = stream.Measure(number=1)
        curr_measure.append(clef.BassClef())
        curr_measure.append(detected_key)
        curr_measure.append(meter.TimeSignature('12/8' if is_compound else '4/4'))

        for i, (start, end, pitch_val, amp, _, _, _) in enumerate(snapped_layer):
            key_pitch = get_key_aware_pitch(pitch_val, detected_key)
            n = note.Note(key_pitch)
            n.quarterLength = idiomatic_rhythm_snap(end - start, level=target_level, is_compound=is_compound)
            s_idx, f_val = fretboard_path[i] if i < len(fretboard_path) else (4, 0)
            n.articulations.extend([articulations.StringIndication(s_idx), articulations.FretIndication(f_val)])
            curr_measure.append(n)

        consolidate_measure_notation(curr_measure)
        m21_part.append(curr_measure)
        m21_score.append(m21_part)
        m21_score.write('musicxml', fp=xml_out)

        sanitize_and_inject_tablature(xml_out, artist_name, song_title, '4_string_standard', level=target_level)
        print(f" -> Output saved: {xml_out}")

def main():
    parser = argparse.ArgumentParser(description="Bass Transcription Engine")
    parser.add_argument('folders', nargs='+', help="Path to stem folder(s)")
    parser.add_argument('-a', '--all-levels', action='store_true', help="Generate outputs for all levels")
    parser.add_argument('-o', '--output-dir', help="Custom output directory")
    parser.add_argument('--level', type=int, default=5, help="Complexity level (0-5) - Defaults to 5")
    parser.add_argument('-g', '--gpu', action='store_true', help="Use GPU stack")

    args = parser.parse_args()

    for folder in args.folders:
        if os.path.isdir(folder):
            process_folder(folder, generate_all_levels=args.all_levels, custom_output_dir=args.output_dir, level=args.level, use_gpu=args.gpu)
        else:
            print(f"Directory not found: {folder}")

if __name__ == "__main__":
    main()
