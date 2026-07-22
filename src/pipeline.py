import os
import re
import json
import difflib
from copy import deepcopy
import numpy as np

from models import NoteEvent, Genre, Level
from pitch_theory import snap_pitch_to_scale
from transcriber import stems_to_note_events
from fretboard_hmm import ErgonomicFretboardHMMSolver
from score_builder import build_and_export_score


def load_genre_configs(config_path=None):
    """Loads genre, category, and pattern configurations dynamically."""
    if not config_path:
        config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
        genres_path = os.path.join(config_dir, "genres.json")
    else:
        if os.path.isdir(config_path):
            config_dir = config_path
            genres_path = os.path.join(config_dir, "genres.json")
        else:
            config_dir = os.path.dirname(config_path)
            genres_path = config_path

    categories_path = os.path.join(config_dir, "categories.json")
    patterns_path = os.path.join(config_dir, "patterns.json")

    genres = {}
    if os.path.exists(genres_path):
        with open(genres_path, "r", encoding="utf-8") as f:
            genres = json.load(f)
            # Handle if the input was still a unified file containing 'genres' key
            if isinstance(genres, dict) and "genres" in genres:
                categories_fb = genres.get("categories", {})
                patterns_fb = genres.get("patterns", {})
                genres = genres["genres"]
                return {
                    "genres": genres,
                    "categories": categories_fb,
                    "patterns": patterns_fb
                }

    categories = {}
    if os.path.exists(categories_path):
        with open(categories_path, "r", encoding="utf-8") as f:
            categories = json.load(f)

    patterns = {}
    if os.path.exists(patterns_path):
        with open(patterns_path, "r", encoding="utf-8") as f:
            patterns = json.load(f)

    return {
        "genres": genres,
        "categories": categories,
        "patterns": patterns
    }


def resolve_genre(raw_genre, genre_configs=None):
    """Resolves genre string against dynamically loaded JSON configs with full recursive inheritance support."""
    if genre_configs is None:
        genre_configs = load_genre_configs()

    # Unwrap dictionaries
    genres_dict = genre_configs.get("genres", {})
    categories_dict = genre_configs.get("categories", {})
    patterns_dict = genre_configs.get("patterns", {})

    # Match genre key
    if not raw_genre:
        matched_key = "default"
    else:
        clean = raw_genre.strip().lower()
        matched_key = None
        if clean in genres_dict:
            matched_key = clean
        else:
            for g in genres_dict:
                if g in clean or clean in g:
                    matched_key = g
                    break
            if not matched_key:
                valid_keys = list(genres_dict.keys())
                matches = difflib.get_close_matches(clean, valid_keys, n=1, cutoff=0.3)
                matched_key = matches[0] if matches else "default"

    raw_cfg = genres_dict.get(matched_key, genres_dict.get("default", {}))

    def merge_configs(parent, child):
        res = deepcopy(parent)
        for k, v in child.items():
            if isinstance(v, dict) and k in res and isinstance(res[k], dict):
                res[k] = merge_configs(res[k], v)
            else:
                res[k] = deepcopy(v)
        return res

    def resolve_inheritance(cfg):
        if isinstance(cfg, dict) and "extends" in cfg:
            parent_key = cfg["extends"]
            parent_cfg = categories_dict.get(parent_key, genres_dict.get(parent_key, {}))
            if parent_cfg:
                resolved_parent = resolve_inheritance(parent_cfg)
                return merge_configs(resolved_parent, cfg)
        return deepcopy(cfg)

    resolved_cfg = resolve_inheritance(raw_cfg)

    # Automatically resolve the pattern string in "rhythmic_anchor" if present
    if isinstance(resolved_cfg, dict) and "rhythmic_anchor" in resolved_cfg:
        anchor = resolved_cfg["rhythmic_anchor"]
        if isinstance(anchor, dict) and "pattern" in anchor:
            pat_name = anchor["pattern"]
            if isinstance(pat_name, str) and pat_name in patterns_dict:
                anchor["pattern_name"] = pat_name
                anchor["pattern"] = patterns_dict[pat_name].get("accents", [])

    genre_obj = Genre.from_dict(matched_key, resolved_cfg)
    return matched_key, genre_obj


def parse_metadata_from_path(folder_path, custom_genre=None, config_path=None):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    clean_name = re.sub(r'(?i)^stems_', '', folder_name).strip()

    genre_configs = load_genre_configs(config_path)
    artist, title, genre_str, key_str = "Unknown Artist", "Bass Track", custom_genre, None

    if ' - ' in clean_name:
        left, right = clean_name.rsplit(' - ', 1)
        title = right.replace('_', ' ').strip().title()
        parts = left.split('_')

        if not genre_str and parts:
            resolved_g, resolved_cfg = resolve_genre(parts[0], genre_configs)
            if resolved_g != "default":
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

    resolved_genre_name, genre_config = resolve_genre(genre_str, genre_configs)
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
    layer: list[NoteEvent], level: int | Level, beats, is_compound: bool, bpm: float, genre_config=None
) -> list[NoteEvent]:
    if not layer: return []

    if not isinstance(level, Level):
        level_obj = Level.from_id(level)
    else:
        level_obj = level

    if level_obj.level_id == 5: return list(layer)

    filtered = []
    beat_interval = 60.0 / bpm if bpm > 0 else 0.5
    beats_per_measure = 6 if is_compound else 4

    if len(beats) == 0: return list(layer)

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
