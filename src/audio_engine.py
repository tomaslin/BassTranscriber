import numpy as np
import librosa
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

from note_event import NoteEvent


def _pad_audio_for_fft(y, min_len=2048):
    """Pads audio array to min_len to avoid librosa STFT n_fft warnings on short segments."""
    if y is None or len(y) == 0:
        return np.zeros(min_len, dtype=np.float32)
    if len(y) < min_len:
        return np.pad(y, (0, min_len - len(y)))
    return y


def apply_bass_bandpass(audio_y, sr, lowcut=25.0, highcut=400.0):
    """Applies zero-phase Butterworth bandpass filter to isolate bass frequencies without phase distortion."""
    nyquist = 0.5 * sr
    low, high = lowcut / nyquist, min(highcut / nyquist, 0.99)
    sos = butter(2, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, audio_y)


def estimate_master_tuning(audio_y, sr):
    """Estimates global master pitch tuning deviation from A440 (in semitones)."""
    try:
        filtered_y = apply_bass_bandpass(audio_y, sr, lowcut=30.0, highcut=500.0)
        return float(librosa.estimate_tuning(y=filtered_y, sr=sr))
    except Exception:
        return 0.0


def detect_polyphonic_harmonies(audio_y, sr, note_event: NoteEvent, hop_length=512):
    """Detects double-stops or chord intervals (e.g., 10ths, 7ths, 5ths) during the note event."""
    start_sample = int(note_event.start * sr)
    end_sample = int(note_event.end * sr)
    if end_sample - start_sample < 1024:
        return [note_event.pitch]

    segment = audio_y[start_sample:end_sample]
    segment = _pad_audio_for_fft(segment, min_len=4096)
    stft_mag = np.abs(librosa.stft(segment, n_fft=4096, hop_length=hop_length))
    avg_spectrum = np.mean(stft_mag, axis=1)
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    root_hz = librosa.midi_to_hz(note_event.pitch)
    if root_hz < 20:
        return [note_event.pitch]

    min_sec_hz = root_hz * (2 ** (3 / 12))
    max_sec_hz = root_hz * (2 ** (28 / 12))

    valid_mask = (fft_freqs >= min_sec_hz) & (fft_freqs <= max_sec_hz)
    if not np.any(valid_mask):
        return [note_event.pitch]

    root_bin = np.argmin(np.abs(fft_freqs - root_hz))
    root_energy = avg_spectrum[root_bin]

    if root_energy <= 1e-5:
        return [note_event.pitch]

    sub_spectrum = avg_spectrum.copy()
    for mult in [2, 3, 4]:
        h_bin = np.argmin(np.abs(fft_freqs - (root_hz * mult)))
        b_start = max(0, h_bin - 2)
        b_end = min(len(sub_spectrum), h_bin + 3)
        sub_spectrum[b_start:b_end] = 0.0

    sub_spectrum[~valid_mask] = 0.0
    peak_bin = np.argmax(sub_spectrum)
    peak_energy = sub_spectrum[peak_bin]

    if peak_energy / root_energy > 0.35:
        sec_hz = fft_freqs[peak_bin]
        sec_midi = int(round(librosa.hz_to_midi(sec_hz)))
        if sec_midi > note_event.pitch and (sec_midi - note_event.pitch) >= 3:
            return [note_event.pitch, sec_midi]

    return [note_event.pitch]


def pyin_predict_notes(audio_y, sr, conf_threshold=0.30, tuning_offset=0.0) -> list[NoteEvent]:
    """Runs pYIN pitch detection calibrated by master tuning offset with polyphonic interval detection."""
    hop_length, frame_length = 512, 4096
    filtered_audio = apply_bass_bandpass(audio_y, sr, lowcut=25.0, highcut=400.0)

    filtered_audio = _pad_audio_for_fft(filtered_audio, min_len=frame_length)

    f0, _, voiced_probs = librosa.pyin(
        filtered_audio, fmin=25.0, fmax=450.0, sr=sr, frame_length=frame_length, hop_length=hop_length
    )

    f0 = np.nan_to_num(f0)
    voiced_probs = np.nan_to_num(voiced_probs)

    f0 = median_filter(f0, size=3)
    voiced_probs = median_filter(voiced_probs, size=3)
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    rms_env = librosa.feature.rms(y=filtered_audio, frame_length=frame_length, hop_length=hop_length)[0]
    max_rms = np.max(rms_env) if np.max(rms_env) > 0 else 1.0
    norm_rms = rms_env / max_rms

    audio_y_padded = _pad_audio_for_fft(audio_y, min_len=2048)
    stft_mag = np.abs(librosa.stft(audio_y_padded, n_fft=2048, hop_length=hop_length))
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hf_mask = fft_freqs > 2500.0
    hf_energy = np.sum(stft_mag[hf_mask, :], axis=0)
    total_energy = np.sum(stft_mag, axis=0) + 1e-6
    hf_ratio = hf_energy / total_energy

    raw_notes, in_note, start_time, pitch_buf, bend_buf, idx_buf = [], False, 0.0, [], [], []

    def emit_note_event(e_time):
        if not pitch_buf:
            return
        med_midi = np.median(pitch_buf)
        med_pitch = int(round(med_midi))
        microtone_cents = round((med_midi - med_pitch) * 100.0, 1)
        bend_contour = [round(b - med_pitch, 2) for b in bend_buf]
        avg_rms = float(np.mean([norm_rms[i] for i in idx_buf if i < len(norm_rms)])) if idx_buf else 0.5
        avg_hf = float(np.mean([hf_ratio[i] for i in idx_buf if i < len(hf_ratio)])) if idx_buf else 0.0

        tag = "normal"
        if avg_hf > 0.35 and avg_rms > 0.60:
            tag = "pop" if med_pitch >= 43 else "slap"

        ne = NoteEvent(
            start=start_time,
            end=e_time,
            pitch=med_pitch,
            pitches=[med_pitch],
            amplitude=avg_rms,
            bends=bend_contour,
            microtone_cents=microtone_cents,
            tag=tag,
        )

        ne.pitches = detect_polyphonic_harmonies(audio_y, sr, ne, hop_length=hop_length)
        raw_notes.append(ne)

    for idx, (t, f, c) in enumerate(zip(times, f0, voiced_probs)):
        if f > 0.0 and c >= conf_threshold:
            midi_p = librosa.hz_to_midi(f) - tuning_offset
            if not in_note:
                in_note, start_time, pitch_buf, bend_buf, idx_buf = True, t, [midi_p], [midi_p], [idx]
            else:
                if abs(midi_p - np.median(pitch_buf)) > 1.5:
                    if (t - start_time) >= 0.04:
                        emit_note_event(t)
                        start_time, pitch_buf, bend_buf, idx_buf = t, [midi_p], [midi_p], [idx]
                    else:
                        pitch_buf.append(midi_p)
                        bend_buf.append(midi_p)
                        idx_buf.append(idx)
                else:
                    pitch_buf.append(midi_p)
                    bend_buf.append(midi_p)
                    idx_buf.append(idx)
        else:
            if in_note:
                if (t - start_time) >= 0.03:
                    emit_note_event(t)
                in_note, pitch_buf, bend_buf, idx_buf = False, [], [], []

    return raw_notes


def cross_stem_bleed_filter(raw_notes: list[NoteEvent], stem_dict, sr, threshold_ratio=0.85) -> list[NoteEvent]:
    """Filters out cross-stem spectral bleed using harmonic energy summation across fundamental + 3 harmonics."""
    if not raw_notes or not stem_dict:
        return raw_notes

    n_fft = 4096
    hop_length = 512
    stft_dict = {
        name: np.abs(librosa.stft(_pad_audio_for_fft(audio, min_len=n_fft), n_fft=n_fft, hop_length=hop_length))
        for name, audio in stem_dict.items()
        if audio is not None and len(audio) > 0
    }

    if 'bass' not in stft_dict or stft_dict['bass'].shape[1] == 0:
        return raw_notes

    bass_stft = stft_dict['bass']
    n_frames = bass_stft.shape[1]
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    verified_notes = []
    for note in raw_notes:
        f0_hz = librosa.midi_to_hz(note.pitch)
        target_hz_list = [f0_hz * m for m in [1.0, 2.0, 3.0] if (f0_hz * m) < (sr / 2)]

        bin_indices = [np.argmin(np.abs(fft_freqs - hz)) for hz in target_hz_list]

        start_frame = min(max(0, librosa.time_to_frames(note.start, sr=sr, hop_length=hop_length)), n_frames - 1)
        end_frame = min(max(start_frame + 1, librosa.time_to_frames(note.end, sr=sr, hop_length=hop_length)), n_frames)

        if start_frame >= end_frame:
            verified_notes.append(note)
            continue

        bass_e = np.sum([np.mean(bass_stft[b_idx, start_frame:end_frame]) for b_idx in bin_indices])

        def get_stem_harmonic_energy(name, mult=1.0):
            if name in stft_dict and stft_dict[name].shape[1] > start_frame:
                e_frame = min(end_frame, stft_dict[name].shape[1])
                return float(np.sum([np.mean(stft_dict[name][b_idx, start_frame:e_frame]) for b_idx in bin_indices])) * mult
            return 0.0

        bleed_e = get_stem_harmonic_energy('guitar') + get_stem_harmonic_energy('piano') + get_stem_harmonic_energy('other', 0.7)
        vocal_e = get_stem_harmonic_energy('vocals')

        if bleed_e > 0 and (bass_e / (bleed_e + 1e-6)) < threshold_ratio:
            continue

        if vocal_e > 0 and note.pitch < 36 and 'vocals' in stft_dict and stft_dict['vocals'].shape[1] > start_frame:
            e_frame = min(end_frame, stft_dict['vocals'].shape[1])
            vocal_sub_e = np.mean(stft_dict['vocals'][:10, start_frame:e_frame])
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
    """
    Purges artifacts, detects palm mutes, ghost notes, and harmonics while safely padding audio arrays for FFT features.
    """
    if not raw_notes:
        return []

    n_samples = len(bass_audio) if bass_audio is not None else 0

    capped_notes = []
    for n in raw_notes:
        e = n.start + max_single_note_dur if n.duration > max_single_note_dur else n.end
        tag = n.tag

        if n_samples > 0:
            s_idx = max(0, int(n.start * sr))
            e_idx = min(int(n.end * sr), n_samples)
            if e_idx - s_idx > 256:
                note_seg = bass_audio[s_idx:e_idx]

                # Palm mute detection
                note_seg_stft = _pad_audio_for_fft(note_seg, min_len=1024)
                stft_seg = np.abs(librosa.stft(note_seg_stft, n_fft=1024, hop_length=256))
                if stft_seg.shape[1] >= 2:
                    hf_decay = np.sum(stft_seg[15:, -1]) / (np.sum(stft_seg[15:, 0]) + 1e-6)
                    total_decay = np.sqrt(np.mean(note_seg[len(note_seg)//2:]**2)) / (np.sqrt(np.mean(note_seg[:len(note_seg)//2]**2)) + 1e-6)
                    if hf_decay < 0.15 and total_decay < 0.25 and tag == "normal":
                        tag = "palm_mute"

                # Spectral feature analysis with padded audio to avoid n_fft=2048 warnings
                note_seg_spectral = _pad_audio_for_fft(note_seg, min_len=2048)
                flatness = np.mean(librosa.feature.spectral_flatness(y=note_seg_spectral))
                if (n.amplitude < 0.22 and flatness > 0.08) or (n.amplitude < 0.15 and n.duration <= 0.15):
                    tag = "ghost"

                centroid = np.mean(librosa.feature.spectral_centroid(y=note_seg_spectral, sr=sr))
                expected_f0 = librosa.midi_to_hz(n.pitch)
                if centroid > (expected_f0 * 3.5) and flatness < 0.02 and n.pitch >= 43 and tag == "normal":
                    tag = "harmonic"

        capped_notes.append(
            NoteEvent(
                start=n.start,
                end=e,
                pitch=n.pitch,
                pitches=n.pitches,
                amplitude=n.amplitude,
                bends=n.bends,
                microtone_cents=n.microtone_cents,
                tag=tag,
                duty_cycle=n.duty_cycle,
                is_harmonic=(tag == "harmonic"),
            )
        )

    valid_notes = [n for n in capped_notes if not (n.duration < min_valid_duration and n.amplitude < 0.12)]
    if not valid_notes:
        return []

    purged = []
    curr = valid_notes[0]

    # FIX: Micro-gap merging now strictly checks for IDENTICAL pitches (curr.pitch == next_n.pitch)
    for next_n in valid_notes[1:]:
        gap = next_n.start - curr.end

        if curr.pitch == next_n.pitch and gap <= max_micro_rest and (next_n.end - curr.start) <= (max_single_note_dur * 1.5):
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
    """Estimates beat grid timestamps, instantaneous BPM array, and meter profile based on onset pulse periodicity."""
    tempo_val, beat_times = librosa.beat.beat_track(y=drums_y, sr=sr, units='time')

    if len(beat_times) < 2:
        return np.array([0.0, 0.5, 1.0, 1.5]), np.array([120.0] * 4), "4/4"

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

    onset_env = librosa.onset.onset_strength(y=drums_y, sr=sr)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    beat_frames = beat_frames[beat_frames < len(onset_env)]

    time_sig = "4/4"
    if len(beat_frames) >= 8:
        beat_energies = onset_env[beat_frames]
        acc = [np.corrcoef(beat_energies[:-lag], beat_energies[lag:])[0, 1] for lag in range(1, 8)]
        acc = np.nan_to_num(acc)

        avg_bpm = np.median(instant_bpms)
        if acc[2] > max(acc[1], acc[3]) and acc[2] > 0.2:
            time_sig = "12/8" if avg_bpm < 90.0 else "3/4"
        elif acc[4] > 0.2:
            time_sig = "5/4"
        elif acc[6] > 0.2:
            time_sig = "7/8"
        elif avg_bpm < 90.0 and acc[2] > 0.15:
            time_sig = "6/8"

    return beat_times, np.append(instant_bpms, instant_bpms[-1]), time_sig


def snap_events_to_beat_grid(
    raw_notes: list[NoteEvent], beat_times, bpm, is_compound=False, subdivisions=4
) -> list[NoteEvent]:
    """
    Aligns notes relative to beat anchors using an Onset-to-Onset (Legato-First) framework.
    Prevents rest clutter by extending note lengths to subsequent onsets and applying staccato
    dots for short-sounding notes.
    """
    if not raw_notes:
        return []

    grid_notes = []
    avg_amp = float(np.mean([n.amplitude for n in raw_notes])) if raw_notes else 0.5
    first_downbeat = beat_times[0] if len(beat_times) > 0 else 0.0

    def get_local_beat_dur(time_val):
        if len(beat_times) > 0:
            b_idx = int(np.argmin(np.abs(beat_times - time_val)))
            ref = float(beat_times[b_idx])
            if b_idx < len(beat_times) - 1:
                return float(beat_times[b_idx + 1] - ref), ref
            elif b_idx > 0:
                return float(ref - beat_times[b_idx - 1]), ref
        return (60.0 / bpm if bpm > 0 else 0.5), 0.0

    def quantize_time(t_val):
        local_beat_dur, ref_beat = get_local_beat_dur(t_val)
        subdiv_sec_binary = local_beat_dur / (3 if is_compound else subdivisions)
        subdiv_sec_triplet = local_beat_dur / 3.0

        rel_t = t_val - ref_beat
        err_binary = abs(rel_t - round(rel_t / subdiv_sec_binary) * subdiv_sec_binary)
        err_triplet = abs(rel_t - round(rel_t / subdiv_sec_triplet) * subdiv_sec_triplet)

        use_triplet = (err_triplet < (err_binary * 0.55)) and not is_compound
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

        # Legato-First heuristic decision
        if duty_cycle >= 0.50 or raw_gap <= 0.15 or i == num_notes - 1:
            snapped_e = snapped_s + nominal_grid_dur
            is_staccato = False
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

        if note.amplitude < 0.25:
            dynamic_mark = "p"
        elif note.amplitude < 0.45:
            dynamic_mark = "mp"
        elif note.amplitude < 0.70:
            dynamic_mark = "mf"
        else:
            dynamic_mark = "f"

        tag_out = "staccato" if is_staccato else note.tag

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
                dynamic_mark=dynamic_mark,
                is_pickup=is_pickup,
                is_harmonic=note.is_harmonic,
            )
        )

    return grid_notes
