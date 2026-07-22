import re
import numpy as np
import librosa
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfilt
from music21 import key, pitch


def _apply_bass_bandpass(audio_y, sr, lowcut=25.0, highcut=400.0):
    """Applies Butterworth bandpass filter to isolate bass fundamental frequencies."""
    nyquist = 0.5 * sr
    low, high = lowcut / nyquist, min(highcut / nyquist, 0.99)
    sos = butter(2, [low, high], btype='band', output='sos')
    return sosfilt(sos, audio_y)


def estimate_master_tuning(audio_y, sr):
    """Estimates global master pitch tuning deviation from A440 (in semitones)."""
    try:
        filtered_y = _apply_bass_bandpass(audio_y, sr, lowcut=30.0, highcut=500.0)
        return float(librosa.estimate_tuning(y=filtered_y, sr=sr))
    except Exception:
        return 0.0


def normalize_key_str(raw_key):
    """Normalizes key strings into music21 pitch notation."""
    if not raw_key:
        return None

    k = re.sub(r'(?i)sharp', '#', raw_key.strip().replace('_', ' '))
    k = re.sub(r'(?i)flat', '-', k)
    k = re.sub(r'([A-Ga-g])b(?![a-zA-Z])', r'\1-', k)

    if any(k.lower().endswith(ext) for ext in ["min", "minor"]) or (len(k) > 1 and k.endswith("m") and not k.endswith("-m")):
        clean_root = re.sub(r'(?i)[\s\-_]*(min|minor|m)$', '', k).strip()
        return clean_root.lower()

    if any(k.lower().endswith(ext) for ext in ["maj", "major"]):
        clean_root = re.sub(r'(?i)[\s\-_]*(maj|major)$', '', k).strip()
        return clean_root.capitalize()

    return k


def detect_key_signature(audio_y, sr, parsed_key=None):
    """Detects musical key signature using chroma profiles or parsed key metadata."""
    if parsed_key:
        normalized = normalize_key_str(parsed_key)
        try:
            return key.Key(normalized), True
        except Exception:
            pass

    filtered_y = _apply_bass_bandpass(audio_y, sr, lowcut=30.0, highcut=600.0)

    try:
        chroma = librosa.feature.chroma_cqt(y=filtered_y, sr=sr, fmin=librosa.note_to_hz('C1'), n_octaves=4)
    except Exception:
        chroma = librosa.feature.chroma_cqt(y=filtered_y, sr=sr)

    chroma_sum = np.sum(np.log1p(chroma * 10), axis=1)
    if np.sum(chroma_sum) == 0:
        return key.Key('C'), False

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.98, 2.69, 3.34, 3.17])
    pitch_names = ['C', 'C#', 'D', 'E-', 'E', 'F', 'F#', 'G', 'A-', 'A', 'B-', 'B']

    best_score, best_key_str = -float('inf'), 'C'

    for i in range(12):
        rot_chroma = np.roll(chroma_sum, -i)
        maj_corr = np.nan_to_num(np.corrcoef(rot_chroma, major_profile)[0, 1])
        min_corr = np.nan_to_num(np.corrcoef(rot_chroma, minor_profile)[0, 1])

        if maj_corr > best_score:
            best_score, best_key_str = maj_corr, pitch_names[i]
        if min_corr > best_score:
            best_score, best_key_str = min_corr, pitch_names[i].lower()

    try:
        return key.Key(best_key_str), False
    except Exception:
        return key.Key('C'), False


def snap_pitch_to_scale(midi_val, key_obj, level=5):
    """Folds pitch into 4-string bass range and snaps to scale depending on level."""
    while midi_val > 67: midi_val -= 12
    while midi_val < 28: midi_val += 12

    if key_obj is None or level >= 4:
        return midi_val

    scale_pcs = [p.pitchClass for p in key_obj.getPitches()]
    curr_pc = midi_val % 12

    if curr_pc in scale_pcs:
        return midi_val

    distances = sorted([((sp - curr_pc + 6) % 12 - 6, sp) for sp in scale_pcs], key=lambda x: abs(x[0]))

    if level <= 1 or abs(distances[0][0]) <= 1:
        return midi_val + distances[0][0]

    return midi_val


def get_key_aware_pitch(midi_val, key_obj):
    """Returns key-aware music21 Pitch object."""
    if key_obj is None:
        return pitch.Pitch(midi=midi_val)
    try:
        return key_obj.getPitchFromMidi(midi_val)
    except Exception:
        return pitch.Pitch(midi=midi_val)


def cross_stem_bleed_filter(raw_notes, stem_dict, sr, threshold_ratio=0.85):
    """Filters out cross-stem spectral bleed ghost notes and vocal plosive low-end transients."""
    if not raw_notes or not stem_dict:
        return raw_notes

    stft_dict = {
        name: np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        for name, audio in stem_dict.items() if audio is not None and len(audio) > 0
    }

    if 'bass' not in stft_dict or stft_dict['bass'].shape[1] == 0:
        return raw_notes

    bass_stft = stft_dict['bass']
    n_frames = bass_stft.shape[1]
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    verified_notes = []
    for start, end, pitch_val, amp, bends in raw_notes:
        target_hz = librosa.midi_to_hz(pitch_val)
        bin_idx = np.argmin(np.abs(fft_freqs - target_hz))

        start_frame = min(max(0, librosa.time_to_frames(start, sr=sr, hop_length=512)), n_frames - 1)
        end_frame = min(max(start_frame + 1, librosa.time_to_frames(end, sr=sr, hop_length=512)), n_frames)

        if start_frame >= end_frame:
            verified_notes.append((start, end, pitch_val, amp, bends))
            continue

        bass_e = np.mean(bass_stft[bin_idx, start_frame:end_frame])

        def get_stem_energy(name, mult=1.0):
            if name in stft_dict and stft_dict[name].shape[1] > start_frame:
                e_frame = min(end_frame, stft_dict[name].shape[1])
                return float(np.mean(stft_dict[name][bin_idx, start_frame:e_frame])) * mult
            return 0.0

        bleed_e = get_stem_energy('guitar') + get_stem_energy('piano') + get_stem_energy('other', 0.7)
        vocal_e = get_stem_energy('vocals')

        if bleed_e > 0 and (bass_e / (bleed_e + 1e-6)) < threshold_ratio:
            continue

        if vocal_e > 0 and pitch_val < 36 and 'vocals' in stft_dict and stft_dict['vocals'].shape[1] > start_frame:
            e_frame = min(end_frame, stft_dict['vocals'].shape[1])
            vocal_sub_e = np.mean(stft_dict['vocals'][:5, start_frame:e_frame])
            if vocal_sub_e > (bass_e * 1.5):
                continue

        verified_notes.append((start, end, pitch_val, amp, bends))

    return verified_notes


def purge_audio_artifacts(raw_notes, bass_audio=None, sr=22050, max_micro_rest=0.25, min_valid_duration=0.075, max_single_note_dur=4.0):
    """Purges short transients and merges micro-rests and legato overlaps while capping infinite tie tails."""
    if not raw_notes:
        return []

    # Cap individual raw note durations to avoid stuck MIDI notes from audio pitch decay
    capped_notes = []
    for s, e, p, a, b in raw_notes:
        dur = e - s
        if dur > max_single_note_dur:
            e = s + max_single_note_dur
        capped_notes.append([s, e, p, a, b])

    valid_notes = [
        note_item for note_item in capped_notes
        if not (note_item[1] - note_item[0] < min_valid_duration and note_item[3] < 0.35)
    ]

    if not valid_notes:
        return []

    purged, curr = [], valid_notes[0]
    n_samples = len(bass_audio) if bass_audio is not None else 0

    for next_n in valid_notes[1:]:
        c_start, c_end, c_pitch, c_amp, c_bends = curr
        n_start, n_end, n_pitch, n_amp, n_bends = next_n
        gap = n_start - c_end

        has_palm_mute = False
        if n_samples > 0:
            tail_s, tail_e = int(c_end * sr), min(int(n_start * sr), n_samples)
            body_s, body_e = max(0, int(c_start * sr)), min(tail_s, n_samples)

            if tail_e > tail_s and body_e > body_s:
                tail_rms = np.sqrt(np.mean(bass_audio[tail_s:tail_e] ** 2))
                body_rms = np.sqrt(np.mean(bass_audio[body_s:body_e] ** 2))
                if body_rms > 0 and (tail_rms / body_rms) < 0.15:
                    has_palm_mute = True

        if not has_palm_mute and 0 < gap <= 0.30:
            c_end = curr[1] = n_start

        if abs(c_pitch - n_pitch) <= 1 and gap <= max_micro_rest and (n_end - c_start) <= (max_single_note_dur * 1.5):
            curr[1], curr[3] = n_end, max(c_amp, n_amp)
            if c_bends or n_bends:
                curr[4] = (c_bends or []) + (n_bends or [])
        elif 0 < gap <= (max_micro_rest * 0.75):
            curr[1] = n_start
            purged.append(tuple(curr))
            curr = next_n
        else:
            purged.append(tuple(curr))
            curr = next_n

    purged.append(tuple(curr))
    return purged


def pyin_predict_notes(audio_y, sr, conf_threshold=0.30, tuning_offset=0.0):
    """Runs pYIN pitch detection calibrated by master tuning offset."""
    hop_length, frame_length = 512, 4096
    filtered_audio = _apply_bass_bandpass(audio_y, sr, lowcut=25.0, highcut=400.0)

    if len(filtered_audio) < frame_length:
        filtered_audio = np.pad(filtered_audio, (0, frame_length - len(filtered_audio)))

    f0, _, voiced_probs = librosa.pyin(
        filtered_audio, fmin=25.0, fmax=350.0, sr=sr, frame_length=frame_length, hop_length=hop_length
    )

    f0, voiced_probs = np.nan_to_num(f0), np.nan_to_num(voiced_probs)
    nonzero_mask = f0 > 0
    if np.any(nonzero_mask):
        f0[nonzero_mask] = median_filter(f0[nonzero_mask], size=3)

    voiced_probs = median_filter(voiced_probs, size=3)
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    raw_notes, in_note, start_time, pitch_buf, conf_buf = [], False, 0.0, [], []

    for t, f, c in zip(times, f0, voiced_probs):
        if f > 0.0 and c >= conf_threshold:
            midi_p = librosa.hz_to_midi(f) - tuning_offset
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

    if in_note and pitch_buf and (times[-1] - start_time) >= 0.04:
        raw_notes.append((start_time, times[-1], int(round(np.median(pitch_buf))), float(np.mean(conf_buf)), None))

    return raw_notes


def estimate_beat_grid(drums_y, sr):
    """Estimates beat grid timestamps and instantaneous BPM array."""
    tempo_val, beat_times = librosa.beat.beat_track(y=drums_y, sr=sr, units='time')

    if len(beat_times) < 2:
        return np.array([0.0, 0.5, 1.0, 1.5]), np.array([120.0] * 4)

    if beat_times[0] > 0.1:
        first_interval = beat_times[1] - beat_times[0] if len(beat_times) > 1 else 0.5
        if first_interval <= 0.05:
            first_interval = 0.5
        pre_beats, curr_t = [], beat_times[0] - first_interval
        while curr_t >= 0.0:
            pre_beats.append(curr_t)
            curr_t -= first_interval
        beat_times = np.concatenate((np.array(pre_beats[::-1]) if pre_beats else np.array([0.0]), beat_times))

    beat_durations = np.clip(np.diff(beat_times), 0.15, 2.5)
    instant_bpms = median_filter(60.0 / beat_durations, size=5)
    return beat_times, np.append(instant_bpms, instant_bpms[-1])
