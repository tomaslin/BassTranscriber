import numpy as np
import librosa
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

from models import NoteEvent


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
    """Detects double-stops or chord intervals during the note event."""
    start_sample, end_sample = int(note_event.start * sr), int(note_event.end * sr)
    if end_sample - start_sample < 1024:
        return [note_event.pitch]

    segment = _pad_audio_for_fft(audio_y[start_sample:end_sample], min_len=4096)
    stft_mag = np.abs(librosa.stft(segment, n_fft=4096, hop_length=hop_length))
    avg_spectrum = np.mean(stft_mag, axis=1)
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    root_hz = librosa.midi_to_hz(note_event.pitch)
    if root_hz < 20:
        return [note_event.pitch]

    min_sec_hz, max_sec_hz = root_hz * (2 ** (3 / 12)), root_hz * (2 ** (28 / 12))
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
        sub_spectrum[max(0, h_bin - 2):min(len(sub_spectrum), h_bin + 3)] = 0.0

    sub_spectrum[~valid_mask] = 0.0
    peak_bin = np.argmax(sub_spectrum)
    peak_energy = sub_spectrum[peak_bin]

    if peak_energy / root_energy > 0.35:
        sec_hz = fft_freqs[peak_bin]
        sec_midi = int(round(librosa.hz_to_midi(sec_hz)))
        if sec_midi > note_event.pitch and (sec_midi - note_event.pitch) >= 3:
            return [note_event.pitch, sec_midi]

    return [note_event.pitch]


def hps_refine_pitch(frame_spec, fft_freqs, fmin=18.0, fmax=110.0, num_harmonics=4):
    """
    Computes Harmonic Product Spectrum (HPS) for a single spectral frame to infer
    missing fundamental registers (HFRE) in drop-tuned/death metal genres.
    """
    hps = frame_spec.copy()
    for r in range(2, num_harmonics + 1):
        downsampled = np.interp(
            np.arange(0, len(frame_spec)) * r,
            np.arange(0, len(frame_spec)),
            frame_spec
        )
        hps *= downsampled
    valid_mask = (fft_freqs >= fmin) & (fft_freqs <= fmax)
    hps[~valid_mask] = 0.0
    if np.max(hps) > 1e-6:
        best_bin = np.argmax(hps)
        return fft_freqs[best_bin], hps[best_bin]
    return 0.0, 0.0


def extract_csim_context(stem_dict, sr, genre_config=None):
    """
    Computes Multi-Modal Cross-Stem Interaction Matrix (CSIM) features.
    Treats Demucs stems as interdependent physical signals.
    """
    kick_onsets = np.array([])
    double_kick_onsets = np.array([])
    guitar_chroma = None
    guitar_times = None

    if not stem_dict:
        return {
            'kick_onsets': kick_onsets,
            'double_kick_onsets': double_kick_onsets,
            'guitar_chroma': guitar_chroma,
            'guitar_times': guitar_times
        }

    # 1. Drum Stem Analysis (Kick drum & double kick tracking)
    if 'drums' in stem_dict and stem_dict['drums'] is not None and len(stem_dict['drums']) > 0:
        drums_y = stem_dict['drums']
        nyquist = 0.5 * sr
        low, high = 20.0 / nyquist, 90.0 / nyquist
        try:
            sos = butter(2, [low, high], btype='band', output='sos')
            kick_y = sosfiltfilt(sos, drums_y)
            kick_onset_env = librosa.onset.onset_strength(y=kick_y, sr=sr)
            kick_frames = librosa.onset.onset_detect(onset_envelope=kick_onset_env, sr=sr, wait=5)
            kick_onsets = librosa.frames_to_time(kick_frames, sr=sr)
            if len(kick_onsets) > 1:
                diffs = np.diff(kick_onsets)
                double_kick_mask = np.concatenate(([False], diffs < 0.18))
                double_kick_onsets = kick_onsets[double_kick_mask]
        except Exception as e:
            print(f"[CSIM Warn] Error extracting kick features: {e}")

    # 2. Guitar Stem Analysis (Guitar Unison & Bleed tracking)
    if 'guitar' in stem_dict and stem_dict['guitar'] is not None and len(stem_dict['guitar']) > 0:
        guitar_y = stem_dict['guitar']
        try:
            sos_g = butter(2, [80.0 / (0.5 * sr), 1000.0 / (0.5 * sr)], btype='band', output='sos')
            guitar_filtered = sosfiltfilt(sos_g, guitar_y)
            guitar_chroma = librosa.feature.chroma_stft(y=_pad_audio_for_fft(guitar_filtered, 2048), sr=sr, n_fft=2048, hop_length=512)
            guitar_times = librosa.times_like(guitar_chroma, sr=sr, hop_length=512)
        except Exception as e:
            print(f"[CSIM Warn] Error extracting guitar features: {e}")

    return {
        'kick_onsets': kick_onsets,
        'double_kick_onsets': double_kick_onsets,
        'guitar_chroma': guitar_chroma,
        'guitar_times': guitar_times
    }


def pyin_predict_notes(
    audio_y, sr, conf_threshold=0.30, tuning_offset=0.0, fmin=25.0, fmax=450.0, stem_dict=None, genre_config=None
) -> list[NoteEvent]:
    """Runs pYIN pitch detection calibrated by master tuning offset with smoothed filtering, CSIM models, and dynamic fmin bounds."""
    hop_length, frame_length = 512, 4096
    filtered_audio = apply_bass_bandpass(audio_y, sr, lowcut=max(15.0, fmin), highcut=fmax)
    filtered_audio = _pad_audio_for_fft(filtered_audio, min_len=frame_length)

    f0, _, voiced_probs = librosa.pyin(
        filtered_audio, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop_length
    )

    f0 = np.nan_to_num(f0)
    voiced_probs = np.nan_to_num(voiced_probs)

    f0 = median_filter(f0, size=7)
    voiced_probs = median_filter(voiced_probs, size=5)
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

    genre_name = ""
    if genre_config:
        if hasattr(genre_config, "name"):
            genre_name = genre_config.name.lower()
        elif isinstance(genre_config, dict):
            genre_name = genre_config.get("name", "").lower()
        elif isinstance(genre_config, str):
            genre_name = genre_config.lower()

    is_metal = "metal" in genre_name or "rock" in genre_name or "prog" in genre_name or (genre_config and "drop" in getattr(genre_config, "tuning", ""))
    is_synth = "synth" in genre_name or "electronic" in genre_name or "dance" in genre_name or (genre_config and getattr(genre_config, "technique", "") == "synth_emulation")

    # Extract CSIM context if available
    csim = extract_csim_context(stem_dict, sr, genre_config) if stem_dict else None

    # Genre-conditioned pitch array enhancement (KEDI, HFRE, GUCE)
    f0 = f0.copy()
    voiced_probs = voiced_probs.copy()

    for idx in range(len(times)):
        t = times[idx]

        # A. KEDI (Kick Envelope Ducking Recovery) for Synthwave/EDM
        if is_synth and csim and len(csim.get('kick_onsets', [])) > 0:
            kick_onsets = csim['kick_onsets']
            in_ducking = np.any((t >= kick_onsets) & (t <= kick_onsets + 0.22))
            if in_ducking and idx > 0 and f0[idx - 1] > 0.0:
                if f0[idx] == 0.0 or voiced_probs[idx] < conf_threshold:
                    f0[idx] = f0[idx - 1]
                    voiced_probs[idx] = voiced_probs[idx - 1]

        # B. HFRE (Missing Fundamental Recovery) for Metal/Rock
        if is_metal and (f0[idx] == 0.0 or voiced_probs[idx] < conf_threshold) and idx < stft_mag.shape[1]:
            frame_spec = stft_mag[:, idx]
            hps_f, hps_val = hps_refine_pitch(frame_spec, fft_freqs, fmin=18.0, fmax=110.0, num_harmonics=4)
            if hps_f > 0.0:
                f0[idx] = hps_f
                voiced_probs[idx] = conf_threshold + 0.10

        # C. GUCE (Guitar Unison Consensus Engine) for Metal/Rock
        if is_metal and csim and csim.get('guitar_chroma') is not None and idx < csim['guitar_chroma'].shape[1]:
            g_chroma = csim['guitar_chroma']
            if 0.15 <= voiced_probs[idx] < conf_threshold and f0[idx] > 0.0:
                guitar_col = g_chroma[:, idx]
                guitar_pc = np.argmax(guitar_col)
                bass_midi = int(round(librosa.hz_to_midi(f0[idx]) - tuning_offset))
                if (bass_midi % 12) == guitar_pc:
                    voiced_probs[idx] = conf_threshold + 0.05

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

        # Continuous pitch slide / portamento detection for synthwave
        if is_synth and len(bend_buf) >= 5:
            slopes = np.diff(bend_buf)
            if np.all(slopes >= -0.05) or np.all(slopes <= 0.05):
                total_range = abs(bend_buf[-1] - bend_buf[0])
                if total_range >= 1.5:
                    tag = "slide"

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
        ne.determine_category()
        ne.pitches = detect_polyphonic_harmonies(audio_y, sr, ne, hop_length=hop_length)
        raw_notes.append(ne)

    for idx, (t, f, c) in enumerate(zip(times, f0, voiced_probs)):
        if f > 0.0 and c >= conf_threshold:
            midi_p = librosa.hz_to_midi(f) - tuning_offset
            if not in_note:
                in_note, start_time, pitch_buf, bend_buf, idx_buf = True, t, [midi_p], [midi_p], [idx]
            else:
                if abs(midi_p - np.median(pitch_buf)) > 2.1:
                    if (t - start_time) >= 0.08:
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
                if (t - start_time) >= 0.04:
                    emit_note_event(t)
                in_note, pitch_buf, bend_buf, idx_buf = False, [], [], []

    # Filter double kick bleed if metal and csim is active
    if is_metal and csim and len(csim.get('double_kick_onsets', [])) > 0:
        double_kicks = csim['double_kick_onsets']
        cleaned = []
        for note in raw_notes:
            matches_kick = np.any(np.abs(note.start - double_kicks) < 0.04)
            if matches_kick and note.duration < 0.15 and note.amplitude < 0.25:
                continue
            cleaned.append(note)
        raw_notes = cleaned

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
    max_micro_rest=0.18,
    min_valid_duration=0.08,
    max_single_note_dur=4.0,
    genre_config=None,
) -> list[NoteEvent]:
    if not raw_notes:
        return []

    n_samples = len(bass_audio) if bass_audio is not None else 0
    capped_notes = []

    genre_name = ""
    if genre_config:
        if hasattr(genre_config, "name"):
            genre_name = genre_config.name.lower()
        elif isinstance(genre_config, dict):
            genre_name = genre_config.get("name", "").lower()
        elif isinstance(genre_config, str):
            genre_name = genre_config.lower()

    is_funk = "funk" in genre_name or "disco" in genre_name or "jazz" in genre_name or (genre_config and getattr(genre_config, "technique", "") == "slap_pop")

    for n in raw_notes:
        e = n.start + max_single_note_dur if n.duration > max_single_note_dur else n.end
        tag = n.tag

        if n_samples > 0:
            s_idx = max(0, int(n.start * sr))
            e_idx = min(int(n.end * sr), n_samples)
            if e_idx - s_idx > 256:
                note_seg = bass_audio[s_idx:e_idx]
                note_seg_stft = _pad_audio_for_fft(note_seg, min_len=1024)
                stft_seg = np.abs(librosa.stft(note_seg_stft, n_fft=1024, hop_length=256))
                
                # Compute envelope rise time
                peak_idx = np.argmax(np.abs(note_seg))
                rise_time = peak_idx / sr

                # Compute high frequency ratio
                freqs_seg = librosa.fft_frequencies(sr=sr, n_fft=1024)
                hf_mask_seg = freqs_seg > 2500.0
                hf_energy_seg = np.sum(stft_seg[hf_mask_seg, :]) if np.any(hf_mask_seg) and hf_mask_seg.shape[0] <= stft_seg.shape[0] else 0.0
                total_energy_seg = np.sum(stft_seg) + 1e-6
                hf_ratio_seg = hf_energy_seg / total_energy_seg

                if stft_seg.shape[1] >= 2:
                    hf_decay = np.sum(stft_seg[15:, -1]) / (np.sum(stft_seg[15:, 0]) + 1e-6)
                    total_decay = np.sqrt(np.mean(note_seg[len(note_seg)//2:]**2)) / (np.sqrt(np.mean(note_seg[:len(note_seg)//2]**2)) + 1e-6)
                    if hf_decay < 0.15 and total_decay < 0.25 and tag == "normal":
                        tag = "palm_mute"

                note_seg_spectral = _pad_audio_for_fft(note_seg, min_len=2048)
                flatness = np.mean(librosa.feature.spectral_flatness(y=note_seg_spectral))

                if is_funk:
                    # 2D Transient Classification Matrix for Funk/Jazz
                    if hf_ratio_seg > 0.35 and rise_time < 0.03 and n.pitch >= 43:
                        tag = "pop"
                    elif flatness > 0.06 and hf_ratio_seg <= 0.35:
                        tag = "slap"
                    elif n.amplitude < 0.18 and flatness > 0.08:
                        tag = "ghost"
                else:
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
                category=n.category,
                anchor_pattern=n.anchor_pattern,
                anchor_fret=n.anchor_fret,
                is_anchor=n.is_anchor,
            )
        )

    valid_notes = [n for n in capped_notes if not (n.duration < min_valid_duration and n.amplitude < 0.18)]
    if not valid_notes:
        return []

    purged = []
    curr = valid_notes[0]

    for next_n in valid_notes[1:]:
        gap = next_n.start - curr.end
        pitch_diff = abs(curr.pitch - next_n.pitch)
        is_pitch_wobble = (pitch_diff <= 1) or (pitch_diff == 12)

        if is_pitch_wobble and gap <= max_micro_rest and (next_n.end - curr.start) <= (max_single_note_dur * 1.5):
            curr.end = next_n.end
            curr.amplitude = max(curr.amplitude, next_n.amplitude)
            if curr.bends or next_n.bends:
                curr.bends = (curr.bends or []) + (next_n.bends or [])
            if pitch_diff == 12:
                curr.update_pitch(min(curr.pitch, next_n.pitch))
        elif 0 < gap <= 0.12:
            curr.end = next_n.start
            purged.append(curr)
            curr = next_n
        else:
            purged.append(curr)
            curr = next_n

    purged.append(curr)
    return purged


def estimate_beat_grid(drums_y, sr):
    """Estimates beat grid timestamps, instantaneous BPM array, and meter profile."""
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


def snap_events_to_beat_grid(*args, **kwargs):
    from notes_sifter import snap_events_to_beat_grid as _snap
    return _snap(*args, **kwargs)


def apply_lossy_abstraction(*args, **kwargs):
    from notes_sifter import apply_lossy_abstraction as _apply
    return _apply(*args, **kwargs)


def transcribe_audio(
    bass_y,
    sr=22050,
    drums_y=None,
    stem_dict=None,
    abstraction_level: int = 3,
    genre_config=None,
) -> tuple[list[NoteEvent], float, str]:
    """
    High-level entry point function that runs the complete audio processing pipeline:
    Tuning Estimation -> Pitch Detection -> Beat Grid -> Lossy Abstraction.
    """
    tuning_type = genre_config.get("tuning", "4_string_standard") if genre_config else "4_string_standard"
    fmin_hz = 18.0 if ("5_string" in tuning_type or "6_string" in tuning_type or "drop" in tuning_type) else 25.0

    tuning_offset = estimate_master_tuning(bass_y, sr)
    raw_notes = pyin_predict_notes(bass_y, sr, conf_threshold=0.30, tuning_offset=tuning_offset, fmin=fmin_hz)

    drums_signal = drums_y if drums_y is not None else bass_y
    beat_times, bpms, time_sig = estimate_beat_grid(drums_signal, sr)
    avg_bpm = float(np.median(bpms))
    is_compound = time_sig in ["6/8", "12/8", "7/8"]

    final_notes = apply_lossy_abstraction(
        raw_notes=raw_notes,
        audio_y=bass_y,
        sr=sr,
        beat_times=beat_times,
        bpm=avg_bpm,
        abstraction_level=abstraction_level,
        is_compound=is_compound,
        stem_dict=stem_dict,
        genre_config=genre_config,
    )

    return final_notes, avg_bpm, time_sig
