import numpy as np
import librosa
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfilt

from note_event import NoteEvent


def apply_bass_bandpass(audio_y, sr, lowcut=25.0, highcut=400.0):
    """Applies Butterworth bandpass filter to isolate bass fundamental frequencies."""
    nyquist = 0.5 * sr
    low, high = lowcut / nyquist, min(highcut / nyquist, 0.99)
    sos = butter(2, [low, high], btype='band', output='sos')
    return sosfilt(sos, audio_y)


def estimate_master_tuning(audio_y, sr):
    """Estimates global master pitch tuning deviation from A440 (in semitones)."""
    try:
        filtered_y = apply_bass_bandpass(audio_y, sr, lowcut=30.0, highcut=500.0)
        return float(librosa.estimate_tuning(y=filtered_y, sr=sr))
    except Exception:
        return 0.0


def pyin_predict_notes(audio_y, sr, conf_threshold=0.30, tuning_offset=0.0) -> list[NoteEvent]:
    """
    Runs pYIN pitch detection calibrated by master tuning offset.
    Calculates true audio RMS amplitude and continuous pitch bend contours.
    """
    hop_length, frame_length = 512, 4096
    filtered_audio = apply_bass_bandpass(audio_y, sr, lowcut=25.0, highcut=400.0)

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

    # Compute RMS envelope across audio
    rms_env = librosa.feature.rms(y=filtered_audio, frame_length=frame_length, hop_length=hop_length)[0]
    max_rms = np.max(rms_env) if np.max(rms_env) > 0 else 1.0
    norm_rms = rms_env / max_rms

    # Spectral High-Frequency Ratio for Slap/Pop detection
    stft_mag = np.abs(librosa.stft(audio_y, n_fft=2048, hop_length=hop_length))
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hf_mask = fft_freqs > 2500.0
    hf_energy = np.sum(stft_mag[hf_mask, :], axis=0)
    total_energy = np.sum(stft_mag, axis=0) + 1e-6
    hf_ratio = hf_energy / total_energy

    raw_notes, in_note, start_time, pitch_buf, bend_buf, idx_buf = [], False, 0.0, [], [], []

    for idx, (t, f, c) in enumerate(zip(times, f0, voiced_probs)):
        if f > 0.0 and c >= conf_threshold:
            midi_p = librosa.hz_to_midi(f) - tuning_offset
            if not in_note:
                in_note, start_time, pitch_buf, bend_buf, idx_buf = True, t, [midi_p], [midi_p], [idx]
            else:
                if abs(midi_p - np.median(pitch_buf)) > 1.5:
                    if (t - start_time) >= 0.04:
                        med_pitch = int(round(np.median(pitch_buf)))
                        bend_contour = [round(b - med_pitch, 2) for b in bend_buf]
                        avg_rms = float(np.mean([norm_rms[i] for i in idx_buf if i < len(norm_rms)]))
                        avg_hf = float(np.mean([hf_ratio[i] for i in idx_buf if i < len(hf_ratio)]))

                        # Extended Technique Tagging
                        tag = "normal"
                        if avg_hf > 0.35 and avg_rms > 0.60:
                            tag = "pop" if med_pitch >= 43 else "slap"

                        raw_notes.append(
                            NoteEvent(
                                start=start_time,
                                end=t,
                                pitch=med_pitch,
                                amplitude=avg_rms,
                                bends=bend_contour,
                                tag=tag,
                            )
                        )
                    start_time, pitch_buf, bend_buf, idx_buf = t, [midi_p], [midi_p], [idx]
                else:
                    pitch_buf.append(midi_p)
                    bend_buf.append(midi_p)
                    idx_buf.append(idx)
        else:
            if in_note:
                if pitch_buf and (t - start_time) >= 0.04:
                    med_pitch = int(round(np.median(pitch_buf)))
                    bend_contour = [round(b - med_pitch, 2) for b in bend_buf]
                    avg_rms = float(np.mean([norm_rms[i] for i in idx_buf if i < len(norm_rms)]))
                    avg_hf = float(np.mean([hf_ratio[i] for i in idx_buf if i < len(hf_ratio)]))

                    tag = "normal"
                    if avg_hf > 0.35 and avg_rms > 0.60:
                        tag = "pop" if med_pitch >= 43 else "slap"

                    raw_notes.append(
                        NoteEvent(
                            start=start_time,
                            end=t,
                            pitch=med_pitch,
                            amplitude=avg_rms,
                            bends=bend_contour,
                            tag=tag,
                        )
                    )
                in_note, pitch_buf, bend_buf, idx_buf = False, [], [], []

    return raw_notes


def cross_stem_bleed_filter(raw_notes: list[NoteEvent], stem_dict, sr, threshold_ratio=0.85) -> list[NoteEvent]:
    """Filters out cross-stem spectral bleed ghost notes and vocal plosive low-end transients."""
    if not raw_notes or not stem_dict:
        return raw_notes

    stft_dict = {
        name: np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        for name, audio in stem_dict.items()
        if audio is not None and len(audio) > 0
    }

    if 'bass' not in stft_dict or stft_dict['bass'].shape[1] == 0:
        return raw_notes

    bass_stft = stft_dict['bass']
    n_frames = bass_stft.shape[1]
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    verified_notes = []
    for note in raw_notes:
        target_hz = librosa.midi_to_hz(note.pitch)
        bin_idx = np.argmin(np.abs(fft_freqs - target_hz))

        start_frame = min(max(0, librosa.time_to_frames(note.start, sr=sr, hop_length=512)), n_frames - 1)
        end_frame = min(max(start_frame + 1, librosa.time_to_frames(note.end, sr=sr, hop_length=512)), n_frames)

        if start_frame >= end_frame:
            verified_notes.append(note)
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

        if vocal_e > 0 and note.pitch < 36 and 'vocals' in stft_dict and stft_dict['vocals'].shape[1] > start_frame:
            e_frame = min(end_frame, stft_dict['vocals'].shape[1])
            vocal_sub_e = np.mean(stft_dict['vocals'][:5, start_frame:e_frame])
            if vocal_sub_e > (bass_e * 1.5):
                continue

        verified_notes.append(note)

    return verified_notes


def purge_audio_artifacts(
    raw_notes: list[NoteEvent],
    bass_audio=None,
    sr=22050,
    max_micro_rest=0.25,
    min_valid_duration=0.075,
    max_single_note_dur=4.0,
) -> list[NoteEvent]:
    """Purges short transients, tags palm mutes and ghost notes, and merges legato overlaps."""
    if not raw_notes:
        return []

    capped_notes = []
    for n in raw_notes:
        e = n.start + max_single_note_dur if n.duration > max_single_note_dur else n.end
        
        # Detect Ghost Notes (Low amplitude percussive transients)
        tag = n.tag
        if n.amplitude < 0.18 and n.duration <= 0.15:
            tag = "ghost"

        capped_notes.append(
            NoteEvent(
                start=n.start,
                end=e,
                pitch=n.pitch,
                amplitude=n.amplitude,
                bends=n.bends,
                tag=tag,
                duty_cycle=n.duty_cycle,
            )
        )

    valid_notes = [n for n in capped_notes if not (n.duration < min_valid_duration and n.amplitude < 0.12)]

    if not valid_notes:
        return []

    purged = []
    curr = valid_notes[0]
    n_samples = len(bass_audio) if bass_audio is not None else 0

    for next_n in valid_notes[1:]:
        gap = next_n.start - curr.end

        has_palm_mute = False
        if n_samples > 0:
            tail_s, tail_e = int(curr.end * sr), min(int(next_n.start * sr), n_samples)
            body_s, body_e = max(0, int(curr.start * sr)), min(tail_s, n_samples)

            if tail_e > tail_s and body_e > body_s:
                tail_rms = np.sqrt(np.mean(bass_audio[tail_s:tail_e] ** 2))
                body_rms = np.sqrt(np.mean(bass_audio[body_s:body_e] ** 2))
                if body_rms > 0 and (tail_rms / body_rms) < 0.15:
                    has_palm_mute = True

        if has_palm_mute:
            curr.tag = "palm_mute"
        elif 0 < gap <= 0.30:
            curr.end = next_n.start

        if abs(curr.pitch - next_n.pitch) <= 1 and gap <= max_micro_rest and (next_n.end - curr.start) <= (max_single_note_dur * 1.5):
            curr.end = next_n.end
            curr.amplitude = max(curr.amplitude, next_n.amplitude)
            if curr.bends or next_n.bends:
                curr.bends = (curr.bends or []) + (next_n.bends or [])
        elif 0 < gap <= (max_micro_rest * 0.75):
            curr.end = next_n.start
            purged.append(curr)
            curr = next_n
        else:
            purged.append(curr)
            curr = next_n

    purged.append(curr)
    return purged


def estimate_beat_grid(drums_y, sr):
    """Estimates beat grid timestamps and instantaneous BPM array for dynamic tempo mapping."""
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


def snap_events_to_beat_grid(
    raw_notes: list[NoteEvent], beat_times, bpm, is_compound=False, subdivisions=4
) -> list[NoteEvent]:
    """
    Aligns notes to beat grid subdivisions, evaluating both binary (16th) and triplet (1/3 beat) grids.
    Assigns dynamics (p, mp, mf, f) and accents based on audio RMS amplitude levels.
    """
    if not raw_notes:
        return []

    sec_per_beat = 60.0 / bpm if bpm > 0 else 0.5
    subdiv_sec_binary = sec_per_beat / (3 if is_compound else subdivisions)
    subdiv_sec_triplet = sec_per_beat / 3.0

    # Calculate local RMS amplitude average for dynamic accent detection
    amps = [n.amplitude for n in raw_notes]
    avg_amp = float(np.mean(amps)) if amps else 0.5

    grid_notes = []
    for i, note in enumerate(raw_notes):
        raw_dur = note.duration

        # 1. Binary vs Triplet Grid Error Evaluation
        err_binary = abs(note.start - round(note.start / subdiv_sec_binary) * subdiv_sec_binary)
        err_triplet = abs(note.start - round(note.start / subdiv_sec_triplet) * subdiv_sec_triplet)

        use_triplet = (err_triplet < (err_binary * 0.55)) and not is_compound
        subdiv_sec = subdiv_sec_triplet if use_triplet else subdiv_sec_binary

        snapped_s = round(note.start / subdiv_sec) * subdiv_sec
        snapped_e = round(note.end / subdiv_sec) * subdiv_sec

        # Bridge micro-gaps between consecutive notes
        if i + 1 < len(raw_notes):
            next_start_snapped = round(raw_notes[i + 1].start / subdiv_sec) * subdiv_sec
            if 0 < (next_start_snapped - snapped_e) <= subdiv_sec:
                snapped_e = next_start_snapped

        if snapped_e <= snapped_s:
            snapped_e = snapped_s + subdiv_sec

        grid_dur = snapped_e - snapped_s
        duty_cycle = raw_dur / grid_dur if grid_dur > 0 else 1.0

        is_staccato = duty_cycle < 0.65 and note.tag not in ["ghost", "palm_mute"]
        is_accent = note.amplitude > (avg_amp * 1.45)

        # Dynamic level assignment
        if note.amplitude < 0.25:
            dynamic_mark = "p"
        elif note.amplitude < 0.45:
            dynamic_mark = "mp"
        elif note.amplitude < 0.70:
            dynamic_mark = "mf"
        else:
            dynamic_mark = "f"

        grid_notes.append(
            NoteEvent(
                start=snapped_s,
                end=snapped_e,
                pitch=note.pitch,
                amplitude=note.amplitude,
                bends=note.bends,
                tag="staccato" if is_staccato else note.tag,
                duty_cycle=duty_cycle,
                is_triplet=use_triplet,
                is_accent=is_accent,
                dynamic_mark=dynamic_mark,
            )
        )

    return grid_notes
