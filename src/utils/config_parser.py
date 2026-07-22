import os
import sys
import re
import json
import difflib
from copy import deepcopy

# Inject parent directory of utils to sys.path to resolve 'models' correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import Genre


def load_genre_configs(config_path=None):
    """Loads genre, category, and pattern configurations dynamically."""
    if not config_path:
        config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "config")
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
