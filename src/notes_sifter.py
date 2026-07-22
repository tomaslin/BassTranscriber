import os
import sys
import numpy as np
import librosa
from scipy.ndimage import median_filter

# Inject local directory to sys.path to resolve 'models' correctly in all environments
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import NoteEvent


def _pad_audio_for_fft(y, min_len=2048):
    """Pads audio array to min_len to avoid librosa STFT n_fft warnings on short segments."""
    if y is None or len(y) == 0:
        return np.zeros(min_len, dtype=np.float32)
    if len(y) < min_len:
        return np.pad(y, (0, min_len - len(y)))
    return y


def detect_key_signature(audio_y, sr) -> tuple[set[int], int, str]:
    """
    Detects global key signature using Krumhansl-Schmuckler pitch class profiles.
    Returns: (scale_pitch_classes_set, root_midi_class, mode_string)
    """
    if audio_y is None or len(audio_y) == 0:
        return set(range(12)), 0, "major"

    chroma = librosa.feature.chroma_stft(y=_pad_audio_for_fft(audio_y, 4096), sr=sr)
    chroma_avg = np.mean(chroma, axis=1)

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.34, 3.17, 3.28])

    best_score = -np.inf
    best_root = 0
    best_mode = "major"

    for root in range(12):
        maj_rot = np.roll(major_profile, root)
        min_rot = np.roll(minor_profile, root)
        score_maj = float(np.corrcoef(chroma_avg, maj_rot)[0, 1])
        score_min = float(np.corrcoef(chroma_avg, min_rot)[0, 1])

        if score_maj > best_score:
            best_score, best_root, best_mode = score_maj, root, "major"
        if score_min > best_score:
            best_score, best_root, best_mode = score_min, root, "minor"

    scale_steps = [0, 2, 4, 5, 7, 9, 11] if best_mode == "major" else [0, 2, 3, 5, 7, 8, 10]
    scale_pc = set((best_root + step) % 12 for step in scale_steps)
    return scale_pc, best_root, best_mode


def apply_scale_hysteresis(notes: list[NoteEvent], scale_pc: set[int], min_chromatic_duration=0.12) -> list[NoteEvent]:
    """
    Snaps out-of-scale transient notes to nearest in-key pitch unless they have sufficient
    duration/intentionality.
    """
    for n in notes:
        pc = n.pitch % 12
        if pc not in scale_pc and n.duration < min_chromatic_duration:
            diffs = [(abs((pc - spc + 6) % 12 - 6), spc) for spc in scale_pc]
            diffs.sort(key=lambda x: x[0])
            nearest_spc = diffs[0][1]
            shift = (nearest_spc - pc + 6) % 12 - 6
            n.update_pitch(n.pitch + shift)
    return notes


def collapse_gestures(notes: list[NoteEvent], max_gesture_duration=0.16) -> list[NoteEvent]:
    """
    Abstracts fast pitch trajectories (slides, hammer-ons, pull-offs) into single destination
    notes with symbolic articulation tags.
    """
    if len(notes) < 2:
        return notes

    abstracted = []
    i = 0
    while i < len(notes):
        curr = notes[i]
        if i + 1 < len(notes):
            next_n = notes[i + 1]
            dur = next_n.start - curr.start
            pitch_delta = next_n.pitch - curr.pitch

            if dur <= max_gesture_duration and 1 <= abs(pitch_delta) <= 4:
                next_n.slide_from = curr.pitch
                next_n.start = curr.start
                if abs(pitch_delta) <= 2:
                    next_n.tag = "hammer_on" if pitch_delta > 0 else "pull_off"
                else:
                    next_n.tag = "slide"
                next_n.category = "expressive"
                i += 1
                continue

        abstracted.append(curr)
        i += 1

    return abstracted


def smooth_macro_dynamics(notes: list[NoteEvent], window_size_sec=2.5, hysteresis_threshold=0.25) -> list[NoteEvent]:
    """
    Calculates dynamic markings using a macro time-window with hysteresis to prevent notation clutter.
    """
    if not notes:
        return notes

    dyn_levels = [("p", 0.0, 0.30), ("mp", 0.30, 0.50), ("mf", 0.50, 0.72), ("f", 0.72, 1.01)]
    dyn_map = {"p": 0.15, "mp": 0.40, "mf": 0.60, "f": 0.85}
    current_dynamic = "mf"

    for note in notes:
        w_start, w_end = note.start - (window_size_sec / 2.0), note.start + (window_size_sec / 2.0)
        window_amps = [n.amplitude for n in notes if w_start <= n.start <= w_end]
        avg_amp = float(np.mean(window_amps)) if window_amps else note.amplitude

        target_dynamic = "mf"
        for d_name, d_low, d_high in dyn_levels:
            if d_low <= avg_amp < d_high:
                target_dynamic = d_name
                break

        if abs(dyn_map[target_dynamic] - dyn_map[current_dynamic]) >= hysteresis_threshold:
            current_dynamic = target_dynamic

        note.dynamic_mark = current_dynamic

    return notes


def snap_events_to_beat_grid(
    raw_notes: list[NoteEvent], beat_times, bpm, is_compound=False, subdivisions=4, genre_config=None
) -> list[NoteEvent]:
    """Aligns notes relative to beat anchors using Onset-to-Onset duration snapping with genre-aware grid rules."""
    if not raw_notes:
        return []

    features = genre_config.get("features", {}) if genre_config else {}
    rhythmic_grid = genre_config.get("rhythmic_grid", "") if genre_config else ""
    rhythmic_anchor = genre_config.get("rhythmic_anchor", {}) if genre_config else {}
    anchor_pattern = rhythmic_anchor.get("pattern", [])

    is_quantized_straight = (rhythmic_grid == "quantized_straight") or features.get("synth_emulation", False)

    grid_notes = []
    avg_amp = float(np.mean([n.amplitude for n in raw_notes])) if raw_notes else 0.5
    first_downbeat = beat_times[0] if len(beat_times) > 0 else 0.0
    sec_per_beat = 60.0 / bpm if bpm > 0 else 0.5
    rest_threshold_sec = sec_per_beat * 0.5

    def get_local_beat_dur(time_val):
        if len(beat_times) > 0:
            b_idx = int(np.argmin(np.abs(beat_times - time_val)))
            ref = float(beat_times[b_idx])
            if b_idx < len(beat_times) - 1:
                return float(beat_times[b_idx + 1] - ref), ref
            elif b_idx > 0:
                return float(ref - beat_times[b_idx - 1]), ref
        return sec_per_beat, 0.0

    def quantize_time(t_val):
        local_beat_dur, ref_beat = get_local_beat_dur(t_val)
        subdiv_sec_binary = local_beat_dur / (3 if is_compound else subdivisions)
        subdiv_sec_triplet = local_beat_dur / 3.0

        rel_t = t_val - ref_beat
        err_binary = abs(rel_t - round(rel_t / subdiv_sec_binary) * subdiv_sec_binary)
        err_triplet = abs(rel_t - round(rel_t / subdiv_sec_triplet) * subdiv_sec_triplet)

        use_triplet = (err_triplet < (err_binary * 0.55)) and not is_compound and not is_quantized_straight
        subdiv_sec = subdiv_sec_triplet if use_triplet else subdiv_sec_binary

        snapped = ref_beat + (round(rel_t / subdiv_sec) * subdiv_sec)
        return max(0.0, snapped), subdiv_sec, use_triplet

    num_notes = len(raw_notes)
    for i in range(num_notes):
        note = raw_notes[i]
        raw_dur = note.duration

        is_pickup = (i == 0 and note.start < (first_downbeat - 0.15))
        snapped_s, subdiv_sec, use_triplet = quantize_time(note.start)

        if i + 1 < num_notes:
            next_note = raw_notes[i + 1]
            snapped_next_s, _, _ = quantize_time(next_note.start)
            nominal_grid_dur = max(subdiv_sec, snapped_next_s - snapped_s)
            raw_gap = next_note.start - note.end
        else:
            nominal_grid_dur = max(subdiv_sec, round(raw_dur / subdiv_sec) * subdiv_sec)
            raw_gap = 1.0

        duty_cycle = raw_dur / nominal_grid_dur if nominal_grid_dur > 0 else 1.0

        if raw_gap < rest_threshold_sec or duty_cycle >= 0.50 or i == num_notes - 1:
            snapped_e = snapped_s + nominal_grid_dur
            is_staccato = (duty_cycle < 0.45 and note.tag == "normal")
        elif raw_gap <= 0.35 and note.tag not in ["ghost", "palm_mute"]:
            snapped_e = snapped_s + nominal_grid_dur
            is_staccato = True
        else:
            snapped_e_raw, _, _ = quantize_time(note.end)
            snapped_e = max(snapped_s + subdiv_sec, min(snapped_e_raw, snapped_s + nominal_grid_dur))
            is_staccato = False

        grid_dur = max(0.01, snapped_e - snapped_s)
        effective_duty = raw_dur / grid_dur

        is_accent = note.amplitude > (avg_amp * 1.45)
        tag_out = "staccato" if is_staccato else note.tag

        is_downbeat = (i == 0) or (is_pickup) or (abs(snapped_s - first_downbeat) < 0.05) or (sec_per_beat > 0 and abs((snapped_s - first_downbeat) % sec_per_beat) < 0.05)

        is_pattern_anchor = False
        if anchor_pattern and len(anchor_pattern) > 0:
            rel_beat_pos = ((snapped_s - first_downbeat) / sec_per_beat) % 4.0
            is_pattern_anchor = any(abs(rel_beat_pos - pat_pos) < 0.125 for pat_pos in anchor_pattern)

        is_anchor_evt = is_downbeat or is_accent or is_pattern_anchor
        anchor_pat = "downbeat_anchor" if is_downbeat else ("pattern_anchor" if is_pattern_anchor else ("beat_anchor" if is_anchor_evt else "subdivision"))

        if tag_out in ["ghost", "palm_mute", "slap", "pop", "staccato"]:
            cat = "percussive"
        elif tag_out in ["hammer_on", "pull_off", "slide"] or note.is_harmonic or len(note.bends or []) > 0:
            cat = "expressive"
        elif is_downbeat or is_pattern_anchor:
            cat = "groove_anchor"
        else:
            cat = "melodic"

        grid_notes.append(
            NoteEvent(
                start=snapped_s,
                end=snapped_e,
                pitch=note.pitch,
                pitches=note.pitches,
                amplitude=note.amplitude,
                bends=note.bends,
                microtone_cents=note.microtone_cents,
                tag=tag_out,
                duty_cycle=effective_duty,
                is_triplet=use_triplet,
                is_accent=is_accent,
                dynamic_mark=note.dynamic_mark,
                is_pickup=is_pickup,
                is_harmonic=note.is_harmonic,
                slide_from=note.slide_from,
                category=cat,
                anchor_pattern=anchor_pat,
                is_anchor=is_anchor_evt,
            )
        )

    return grid_notes


def apply_lossy_abstraction(
    raw_notes: list[NoteEvent],
    audio_y,
    sr,
    beat_times,
    bpm,
    abstraction_level: int = 3,
    is_compound: bool = False,
    stem_dict: dict = None,
    genre_config: dict = None,
) -> list[NoteEvent]:
    """Applies multi-level lossy abstraction pipeline based on requested complexity mode (Levels 1 to 5)."""
    if not raw_notes:
        return []

    # Import from audio_engine locally to avoid circular dependencies
    from audio_engine import cross_stem_bleed_filter, purge_audio_artifacts

    if stem_dict is not None:
        raw_notes = cross_stem_bleed_filter(raw_notes, stem_dict, sr)

    purged = purge_audio_artifacts(raw_notes, bass_audio=audio_y, sr=sr, genre_config=genre_config)

    if abstraction_level <= 3 and audio_y is not None:
        scale_pc, key_root, mode = detect_key_signature(audio_y, sr)
        min_chrom_dur = 0.18 if abstraction_level == 1 else (0.14 if abstraction_level == 2 else 0.10)
        purged = apply_scale_hysteresis(purged, scale_pc, min_chromatic_duration=min_chrom_dur)

    if abstraction_level <= 4:
        max_gest_dur = 0.18 if abstraction_level <= 2 else 0.14
        purged = collapse_gestures(purged, max_gesture_duration=max_gest_dur)

    subdivs = 2 if abstraction_level == 1 else 4
    grid_notes = snap_events_to_beat_grid(
        purged, beat_times=beat_times, bpm=bpm, is_compound=is_compound, subdivisions=subdivs, genre_config=genre_config
    )

    if abstraction_level <= 4:
        win_size = 3.5 if abstraction_level <= 2 else 2.5
        grid_notes = smooth_macro_dynamics(grid_notes, window_size_sec=win_size, hysteresis_threshold=0.25)

    if abstraction_level == 1:
        for n in grid_notes:
            n.microtone_cents = 0.0
            n.bends = []

    return grid_notes
