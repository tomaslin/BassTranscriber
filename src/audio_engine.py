# audio_engine.py
import re
import numpy as np
import librosa
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfilt
from music21 import key, pitch


def _apply_bass_bandpass(audio_y, sr, lowcut=25.0, highcut=400.0):
    """
    Applies a Butterworth bandpass filter to isolate bass fundamental frequencies
    and suppress sub-bass DC rumble and high-frequency string clicks.
    """
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = min(highcut / nyquist, 0.99)
    sos = butter(2, [low, high], btype='band', output='sos')
    return sosfilt(sos, audio_y)


def normalize_key_str(raw_key):
    if not raw_key:
        return None

    k = raw_key.strip()
    k = re.sub(r'(?i)sharp', '#', k)
    k = re.sub(r'(?i)flat', '-', k)
    k = re.sub(r'([A-Ga-g])b(?![a-zA-Z])', r'\1-', k)

    if k.lower().endswith("min") or k.lower().endswith("minor") or (len(k) > 1 and k.endswith("m") and not k.endswith("-m")):
        clean_root = re.sub(r'(?i)(min|minor|m)$', '', k).strip()
        return clean_root.lower()

    if k.lower().endswith("maj") or k.lower().endswith("major"):
        clean_root = re.sub(r'(?i)(maj|major)$', '', k).strip()
        return clean_root.capitalize()

    return k


def detect_key_signature(audio_y, sr, parsed_key=None):
    if parsed_key:
        normalized = normalize_key_str(parsed_key)
        try:
            return key.Key(normalized), True
        except Exception:
            pass

    # Research Improvement: Pre-filter audio to avoid upper-register noise bias
    filtered_y = _apply_bass_bandpass(audio_y, sr, lowcut=30.0, highcut=600.0)

    try:
        chroma = librosa.feature.chroma_cqt(
            y=filtered_y, sr=sr, fmin=librosa.note_to_hz('C1'), n_octaves=4
        )
    except Exception:
        chroma = librosa.feature.chroma_cqt(y=filtered_y, sr=sr)

    # Apply logarithmic energy scaling to favor lower fundamentals
    chroma = np.log1p(chroma * 10)
    chroma_sum = np.sum(chroma, axis=1)

    if np.sum(chroma_sum) == 0:
        return key.Key('C'), False

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
        return key.Key(best_key_str), False
    except Exception:
        return key.Key('C'), False


def snap_pitch_to_scale(midi_val, key_obj, level=5):
    while midi_val > 67:
        midi_val -= 12
    while midi_val < 23:
        midi_val += 12

    if key_obj is None or level >= 4:
        return midi_val

    scale_pcs = [p.pitchClass for p in key_obj.getPitches()]
    curr_pc = midi_val % 12

    if curr_pc in scale_pcs:
        return midi_val

    distances = [((sp - curr_pc + 6) % 12 - 6, sp) for sp in scale_pcs]
    distances.sort(key=lambda x: abs(x[0]))

    if level <= 1:
        return midi_val + distances[0][0]

    if abs(distances[0][0]) <= 1:
        return midi_val + distances[0][0]

    return midi_val


def get_key_aware_pitch(midi_val, key_obj):
    if key_obj is None:
        return pitch.Pitch(midi=midi_val)
    try:
        return key_obj.getPitchFromMidi(midi_val)
    except Exception:
        return pitch.Pitch(midi=midi_val)


def purge_audio_artifacts(raw_notes, max_micro_rest=0.22, min_valid_duration=0.075):
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

        if abs(c_pitch - n_pitch) <= 1 and gap <= max_micro_rest:
            curr[1] = n_end
            curr[3] = max(c_amp, n_amp)
            if c_bends or n_bends:
                curr[4] = (c_bends or []) + (n_bends or [])
        elif 0 < gap <= max_micro_rest:
            curr[1] = n_start
            purged.append(tuple(curr))
            curr = next_n
        else:
            purged.append(tuple(curr))
            curr = next_n

    purged.append(tuple(curr))
    return purged


def pyin_predict_notes(audio_y, sr, conf_threshold=0.30):
    hop_length = 512
    frame_length = 4096

    # Research Improvement: Bandpass audio before fundamental estimation
    filtered_audio = _apply_bass_bandpass(audio_y, sr, lowcut=25.0, highcut=400.0)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        filtered_audio, fmin=25.0, fmax=350.0, sr=sr,
        frame_length=frame_length, hop_length=hop_length
    )
    
    f0 = np.nan_to_num(f0)
    voiced_probs = np.nan_to_num(voiced_probs)

    # Research Improvement: Smooth f0 directly to suppress octave jitter spikes
    nonzero_mask = f0 > 0
    if np.any(nonzero_mask):
        f0[nonzero_mask] = median_filter(f0[nonzero_mask], size=3)

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


def estimate_beat_grid(drums_y, sr):
    """
    Returns an array of beat timestamps (in seconds) and an initial instantaneous BPM array.
    """
    tempo_val, beat_times = librosa.beat.beat_track(y=drums_y, sr=sr, units='time')
    
    if len(beat_times) < 2:
        # Fallback for ultra-short clips
        return np.array([0.0, 0.5, 1.0, 1.5]), np.array([120.0, 120.0, 120.0, 120.0])

    # Prepend backwards to handle pickup notes occurring before the first drum transient
    if beat_times[0] > 0.1:
        first_interval = beat_times[1] - beat_times[0] if len(beat_times) > 1 else 0.5
        pre_beats = np.arange(beat_times[0] - first_interval, 0.0, -first_interval)[::-1]
        if len(pre_beats) == 0: pre_beats = np.array([0.0])
        beat_times = np.concatenate((pre_beats, beat_times))

    beat_durations = np.diff(beat_times)
    beat_durations = np.clip(beat_durations, 0.15, 2.5)
    instant_bpms = 60.0 / beat_durations
    
    # Smooth BPMs to avoid tracking jitter over-cluttering the score tempo maps
    smoothed_bpms = median_filter(instant_bpms, size=5)
    smoothed_bpms = np.append(smoothed_bpms, smoothed_bpms[-1])

    return beat_times, smoothed_bpms
