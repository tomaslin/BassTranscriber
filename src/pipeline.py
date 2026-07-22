import os
import re
import json
import difflib
import numpy as np
import scipy.signal as signal
import librosa

from note_event import NoteEvent
from pitch_theory import (
    detect_key_signature,
    snap_pitch_to_scale,
)
from audio_engine import (
    estimate_master_tuning,
    apply_bass_bandpass,
    pyin_predict_notes,
    cross_stem_bleed_filter,
    purge_audio_artifacts,
    estimate_beat_grid,
    snap_events_to_beat_grid,
)
from fretboard_hmm import ErgonomicFretboardHMMSolver
from score_builder import build_and_export_score


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


def filter_performance_for_level(
    layer: list[NoteEvent], level: int, beats, is_compound: bool, bpm: float, genre_config=None
) -> list[NoteEvent]:
    if not layer: return []
    if level == 5: return list(layer)

    filtered = []
    beat_interval = 60.0 / bpm if bpm > 0 else 0.5
    measure_len = beat_interval * (6 if is_compound else 4)

    if len(beats) == 0: return list(layer)

    downbeats = np.arange(beats[0], layer[-1].end + measure_len, measure_len) if layer[-1].end >= beats[0] else np.array([])
    half_measure_beats = np.arange(beats[0] + (measure_len / 2), layer[-1].end + measure_len, measure_len) if layer[-1].end >= beats[0] else np.array([])
    eighth_beats = [b + beat_interval / 2 for b in beats]

    ghost_enabled = genre_config["features"].get("ghost_notes", True) if genre_config else True

    if level <= 1:
        target_beats = np.sort(np.concatenate((downbeats, half_measure_beats)))
        for tb in target_beats:
            window_notes = [n for n in layer if tb - 0.20 <= n.start < tb + (beat_interval * 2)]
            if window_notes:
                root_note = min(window_notes, key=lambda x: x.pitch)
                filtered.append(
                    NoteEvent(
                        start=root_note.start,
                        end=root_note.start + (measure_len * 0.5),
                        pitch=root_note.pitch,
                        pitches=root_note.pitches,
                        amplitude=root_note.amplitude,
                        bends=root_note.bends,
                        microtone_cents=root_note.microtone_cents,
                        tag="normal",
                        duty_cycle=root_note.duty_cycle,
                        is_triplet=root_note.is_triplet,
                        is_accent=root_note.is_accent,
                        dynamic_mark=root_note.dynamic_mark,
                        is_pickup=root_note.is_pickup,
                    )
                )
        return filtered

    for note in layer:
        dur = note.duration
        c_beat = get_closest_value(note.start, beats)
        is_on_beat = abs(note.start - c_beat) < 0.15 if c_beat is not None else False
        c_eighth = get_closest_value(note.start, eighth_beats)
        is_on_eighth = abs(note.start - c_eighth) < 0.15 if c_eighth is not None else False

        if level == 2:
            if note.tag != "ghost" and (is_on_beat or (is_on_eighth and dur >= 0.20)):
                filtered.append(
                    NoteEvent(
                        start=note.start,
                        end=note.end,
                        pitch=note.pitch,
                        pitches=note.pitches,
                        amplitude=note.amplitude,
                        bends=note.bends,
                        microtone_cents=note.microtone_cents,
                        tag="normal",
                        duty_cycle=note.duty_cycle,
                        is_triplet=note.is_triplet,
                        is_accent=note.is_accent,
                        dynamic_mark=note.dynamic_mark,
                        is_pickup=note.is_pickup,
                    )
                )
        elif level == 3:
            if note.tag != "ghost" and dur >= 0.12:
                filtered.append(note)
        elif level == 4:
            if not (note.tag == "ghost" and not ghost_enabled):
                filtered.append(note)

    return filtered


class AudioTranscriptionPipeline:
    def __init__(self, output_dir=None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.output_dir = output_dir or os.path.join(project_root, 'output_bass')

    def run(self, stem_folder: str, generate_all_levels=False, level=5, use_gpu=False):
        artist_name, song_title, _, parsed_key_str, resolved_genre, genre_config = parse_metadata_from_path(stem_folder)
        clean_filename = re.sub(r'[\\/*?:"<>|]', "", f"{artist_name} - {song_title}").strip()

        print(f"[Processing] {artist_name} - {song_title}")

        bass_path = os.path.join(stem_folder, 'bass.wav')
        drums_path = os.path.join(stem_folder, 'drums.wav') if os.path.exists(os.path.join(stem_folder, 'drums.wav')) else bass_path

        if not os.path.exists(bass_path):
            print(f"Skipped: Missing bass.wav in {stem_folder}")
            return

        os.makedirs(self.output_dir, exist_ok=True)

        sr = 22050
        bass_y, _ = librosa.load(bass_path, sr=sr, mono=True)
        drums_y, _ = librosa.load(drums_path, sr=sr, mono=True)

        stem_dict = {'bass': bass_y, 'drums': drums_y}
        for name in ['guitar', 'piano', 'vocals', 'other']:
            p = os.path.join(stem_folder, f'{name}.wav')
            if os.path.exists(p):
                stem_dict[name], _ = librosa.load(p, sr=sr, mono=True)

        tuning_offset = estimate_master_tuning(bass_y, sr)
        detected_key, _ = detect_key_signature(
            bass_y, sr, parsed_key=parsed_key_str, bass_filter_fn=apply_bass_bandpass
        )

        sos_low = signal.butter(4, [25 / (sr / 2), 280 / (sr / 2)], 'bandpass', output='sos')
        bass_low = signal.sosfiltfilt(sos_low, bass_y)

        raw_pyin_notes = pyin_predict_notes(bass_low, sr, conf_threshold=0.30, tuning_offset=tuning_offset)

        tuning_type = genre_config.get("tuning", "4_string_standard") if genre_config else "4_string_standard"
        min_p = 23 if "5_string" in tuning_type or "6_string" in tuning_type else 28
        max_p = 72 if "6_string" in tuning_type else 67

        corrected_notes = []
        for n in raw_pyin_notes:
            n.pitch = max(min_p, min(max_p, int(round(n.pitch))))
            corrected_notes.append(n)

        verified_notes = cross_stem_bleed_filter(corrected_notes, stem_dict, sr=sr)
        purged_notes = purge_audio_artifacts(verified_notes, bass_audio=bass_y, sr=sr)

        beat_times, instant_bpms, time_sig_str = estimate_beat_grid(drums_y, sr)
        bpm = float(np.median(instant_bpms)) if len(instant_bpms) > 0 else 120.0
        is_compound = (genre_config and genre_config["features"].get("compound_meter")) or (time_sig_str in ["6/8", "12/8"])

        grid_aligned_notes = snap_events_to_beat_grid(purged_notes, beat_times, bpm, is_compound=is_compound)

        selected_level = level if (isinstance(level, int) and 0 <= level <= 5) else 5
        target_levels = range(6) if generate_all_levels else [selected_level]

        for target_level in target_levels:
            file_title = f"{clean_filename}_Level{target_level}" if generate_all_levels else clean_filename
            xml_out = os.path.join(self.output_dir, f"{file_title}.musicxml")

            level_layer = filter_performance_for_level(
                grid_aligned_notes, target_level, beat_times, is_compound, bpm, genre_config=genre_config
            )

            if not level_layer:
                continue

            snapped_layer = []
            for idx_l, note in enumerate(level_layer):
                next_p = level_layer[idx_l + 1].pitch if idx_l + 1 < len(level_layer) else None
                note.pitch = snap_pitch_to_scale(note.pitch, detected_key, level=target_level, next_midi=next_p)
                snapped_layer.append(note)

            hmm = ErgonomicFretboardHMMSolver(tuning_type=tuning_type)
            fretboard_path, rakes, legatos, slides = hmm.solve(snapped_layer, bpm=bpm)

            for idx_n, note_n in enumerate(snapped_layer):
                if idx_n < len(legatos):
                    note_n.is_legato = legatos[idx_n]
                if idx_n < len(slides):
                    note_n.is_slide = slides[idx_n]
                if idx_n < len(rakes):
                    note_n.is_rake = rakes[idx_n]

            expressive_data = {'rakes': rakes, 'legatos': legatos, 'slides': slides}
            build_and_export_score(
                snapped_layer,
                fretboard_path,
                detected_key,
                song_title,
                artist_name,
                bpm,
                is_compound,
                target_level,
                xml_out,
                beat_times=beat_times,
                instant_bpms=instant_bpms,
                expressive_data=expressive_data,
                time_sig_str=time_sig_str,
                tuning_type=tuning_type,
            )
            print(f" -> Output saved: {xml_out}")
