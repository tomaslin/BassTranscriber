import os
import sys
import re
import argparse
import numpy as np

# Inject local directory to sys.path to resolve 'models' correctly in all environments
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import NoteEvent, Genre, Level
from pitch_theory import snap_pitch_to_scale
from stems_to_notes import stems_to_note_events
from fretboard_hmm import ErgonomicFretboardHMMSolver
from score_builder import build_and_export_score
from utils import (
    load_genre_configs,
    resolve_genre,
    parse_metadata_from_path,
    get_closest_value,
)


def filter_performance_for_level(
    layer: list[NoteEvent], level: int | Level, beats, is_compound: bool, bpm: float, genre_config=None
) -> list[NoteEvent]:
    if not layer:
        return []

    if not isinstance(level, Level):
        level_obj = Level.from_id(level)
    else:
        level_obj = level

    if level_obj.level_id == 5:
        return list(layer)

    filtered = []
    beat_interval = 60.0 / bpm if bpm > 0 else 0.5
    beats_per_measure = 6 if is_compound else 4

    if len(beats) == 0:
        return list(layer)

    downbeat_indices = range(0, len(beats), beats_per_measure)
    downbeats = np.array([beats[bi] for bi in downbeat_indices])

    half_indices = [min(bi + beats_per_measure // 2, len(beats) - 1) for bi in downbeat_indices]
    half_measure_beats = np.array([beats[hi] for hi in half_indices])

    eighth_beats = []
    for i in range(len(beats) - 1):
        eighth_beats.append(beats[i])
        eighth_beats.append((beats[i] + beats[i+1]) / 2.0)

    ghost_enabled = True
    if genre_config:
        ghost_enabled = genre_config.get("features", {}).get("ghost_notes", True)

    if level_obj.downbeat_only:
        target_beats = np.sort(np.unique(np.concatenate((downbeats, half_measure_beats))))
        for tb in target_beats:
            window_notes = [n for n in layer if tb - 0.20 <= n.start < tb + (beat_interval * 1.5)]
            if window_notes:
                root_note = min(window_notes, key=lambda x: x.pitch)
                filtered.append(
                    NoteEvent(
                        start=root_note.start,
                        end=root_note.start + (beat_interval * 2.0),
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
                        category="groove_anchor",
                        is_anchor=True,
                    )
                )
        return filtered

    for note in layer:
        dur = note.duration
        c_beat = get_closest_value(note.start, beats)
        is_on_beat = abs(note.start - c_beat) < 0.15 if c_beat is not None else False
        c_eighth = get_closest_value(note.start, eighth_beats)
        is_on_eighth = abs(note.start - c_eighth) < 0.15 if c_eighth is not None else False

        if level_obj.level_id == 2:
            if note.tag != "ghost" and (is_on_beat or (is_on_eighth and dur >= level_obj.min_duration)):
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
                        category=note.category,
                        anchor_pattern=note.anchor_pattern,
                        anchor_fret=note.anchor_fret,
                        is_anchor=note.is_anchor,
                    )
                )
        elif level_obj.level_id == 3:
            if note.tag != "ghost" and dur >= level_obj.min_duration:
                filtered.append(note)
        elif level_obj.level_id == 4:
            is_ghost_allowed = ghost_enabled and level_obj.ghost_notes_allowed
            if not (note.tag == "ghost" and not is_ghost_allowed):
                filtered.append(note)

    return filtered


class AudioTranscriptionPipeline:
    def __init__(self, output_dir=None, genre_config_path=None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.output_dir = output_dir or os.path.join(project_root, 'output_bass')
        self.genre_config_path = genre_config_path

    def run(self, stem_folder: str, generate_all_levels=False, level=5, use_gpu=False, genre_override=None):
        artist_name, song_title, _, parsed_key_str, resolved_genre, genre_config = parse_metadata_from_path(
            stem_folder, custom_genre=genre_override, config_path=self.genre_config_path
        )
        clean_filename = re.sub(r'[\\/*?:"<>|]', "", f"{artist_name} - {song_title}").strip()

        print(f"[Processing] {artist_name} - {song_title} (Genre: {resolved_genre})")

        os.makedirs(self.output_dir, exist_ok=True)

        try:
            (
                grid_aligned_notes,
                detected_key,
                beat_times,
                instant_bpms,
                time_sig_str,
                bpm,
                is_compound,
                tuning_type,
            ) = stems_to_note_events(stem_folder, genre_config, parsed_key_str=parsed_key_str)
        except FileNotFoundError as e:
            print(f"Skipped: {e}")
            return

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
            for idx_l, note_item in enumerate(level_layer):
                next_p = level_layer[idx_l + 1].pitch if idx_l + 1 < len(level_layer) else None
                snapped_p = snap_pitch_to_scale(note_item.pitch, detected_key, level=target_level, next_midi=next_p)
                note_item.update_pitch(snapped_p)
                snapped_layer.append(note_item)

            # Pass genre configuration into the HMM solver
            hmm = ErgonomicFretboardHMMSolver(tuning_type=tuning_type, genre_config=genre_config)
            fretboard_path, rakes, legatos, slides = hmm.solve(snapped_layer, bpm=bpm)

            for idx_n, note_n in enumerate(snapped_layer):
                if idx_n < len(fretboard_path):
                    note_n.fret_position = fretboard_path[idx_n]
                if idx_n < len(legatos):
                    note_n.is_legato = legatos[idx_n]
                if idx_n < len(slides):
                    note_n.is_slide = slides[idx_n]
                if idx_n < len(rakes):
                    note_n.is_rake = rakes[idx_n]
                note_n.determine_category()

            expressive_data = {
                'rakes': rakes,
                'legatos': legatos,
                'slides': slides,
                'categories': [n.category for n in snapped_layer],
                'anchor_patterns': [n.anchor_pattern for n in snapped_layer],
                'anchor_frets': [n.anchor_fret for n in snapped_layer],
            }

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


def main():
    parser = argparse.ArgumentParser(description="Bass Transcription Engine")
    parser.add_argument('folders', nargs='+', help="Path to stem folder(s)")
    parser.add_argument('-a', '--all-levels', action='store_true', help="Generate outputs for all levels")
    parser.add_argument('-o', '--output-dir', help="Custom output directory")
    parser.add_argument('--level', type=int, default=5, help="Complexity level (0-5) - Defaults to 5")
    parser.add_argument('-g', '--gpu', action='store_true', help="Use GPU stack")

    args = parser.parse_args()

    pipeline = AudioTranscriptionPipeline(output_dir=args.output_dir)

    for folder in args.folders:
        if os.path.isdir(folder):
            pipeline.run(folder, generate_all_levels=args.all_levels, level=args.level, use_gpu=args.gpu)
        else:
            print(f"Directory not found: {folder}")


if __name__ == "__main__":
    main()
