import fractions
from music21 import stream, note, meter, instrument, metadata, tie, articulations, dynamics, tempo, duration

from note_event import NoteEvent
from pitch_theory import get_key_aware_pitch
from xml_formatter import (
    idiomatic_rhythm_snap,
    decompose_duration_engraver_rules,
    consolidate_measure_notation,
    sanitize_and_inject_tablature,
)


def build_and_export_score(
    note_layer: list[NoteEvent],
    fretboard_path: list,
    detected_key,
    song_title: str,
    artist_name: str,
    bpm: float,
    is_compound: bool,
    target_level: int,
    output_xml_path: str,
    beat_times=None,
    instant_bpms=None,
    expressive_data=None,
):
    """
    Builds music21 score stream, applies engraving layout rules, dynamics, tuplets,
    expressive articulations, dynamic tempo maps, and exports MusicXML.
    """
    sec_per_quarter = (60.0 / bpm) if bpm > 0 else 0.5
    measure_capacity = fractions.Fraction(6, 1) if is_compound else fractions.Fraction(4, 1)
    time_sig_str = '12/8' if is_compound else '4/4'

    m21_score = stream.Score()
    m21_part = stream.Part(id="P1")
    m21_part.insert(0.0, instrument.ElectricBass())

    m21_score.metadata = metadata.Metadata()
    m21_score.metadata.title = song_title
    m21_score.metadata.composer = artist_name

    curr_measure_num = 1
    curr_measure = stream.Measure(number=curr_measure_num)
    curr_measure.append(detected_key)
    curr_measure.append(meter.TimeSignature(time_sig_str))
    curr_measure.append(tempo.MetronomeMark(number=round(bpm)))

    curr_m_fill = fractions.Fraction(0, 1)
    current_time_q = fractions.Fraction(0, 1)
    last_dynamic = None

    for i, note_evt in enumerate(note_layer):
        start_q = fractions.Fraction(round((note_evt.start / sec_per_quarter) * 4), 4)
        raw_dur_q = max(0.25, note_evt.duration / sec_per_quarter)
        dur_q = idiomatic_rhythm_snap(raw_dur_q, level=target_level, is_compound=is_compound)

        # 1. Fill Rest Gaps Cleanly
        if start_q > current_time_q:
            rest_q = start_q - current_time_q
            rest_chunks = decompose_duration_engraver_rules(rest_q, curr_m_fill, measure_capacity, is_compound)
            for r_dur in rest_chunks:
                r = note.Rest()
                r.quarterLength = float(r_dur)
                curr_measure.append(r)
                curr_m_fill += r_dur
                current_time_q += r_dur
                if curr_m_fill >= measure_capacity:
                    consolidate_measure_notation(curr_measure, measure_capacity, is_compound)
                    m21_part.append(curr_measure)
                    curr_measure_num += 1
                    curr_measure = stream.Measure(number=curr_measure_num)
                    curr_m_fill = fractions.Fraction(0, 1)

        # 2. Dynamic Tempo Map Updates
        if instant_bpms is not None and beat_times is not None and i < len(beat_times):
            curr_bpm = round(instant_bpms[min(i, len(instant_bpms) - 1)])
            if abs(curr_bpm - bpm) > 6.0 and curr_m_fill == 0:
                bpm = curr_bpm
                curr_measure.append(tempo.MetronomeMark(number=bpm))

        # 3. Append Note Events & Expressive Articulations
        s_idx, f_val = fretboard_path[i] if i < len(fretboard_path) else (4, 0)
        key_pitch = get_key_aware_pitch(note_evt.pitch, detected_key)

        note_chunks = decompose_duration_engraver_rules(dur_q, curr_m_fill, measure_capacity, is_compound)
        num_chunks = len(note_chunks)

        # Dynamic Volume Markings
        if note_evt.dynamic_mark and note_evt.dynamic_mark != last_dynamic:
            curr_measure.append(dynamics.Dynamic(note_evt.dynamic_mark))
            last_dynamic = note_evt.dynamic_mark

        for k, chunk_dur in enumerate(note_chunks):
            n_sub = note.Note(key_pitch)
            
            # Handle Triplets
            if note_evt.is_triplet and k == 0:
                n_sub.duration = duration.Duration(float(chunk_dur))
                n_sub.duration.appendTuplet(duration.Tuplet(3, 2))
            else:
                n_sub.quarterLength = float(chunk_dur)

            n_sub.articulations.extend([articulations.StringIndication(s_idx), articulations.FretIndication(f_val)])

            # Ghost Notes & Articulations
            if note_evt.tag == "ghost":
                n_sub.notehead = 'x'
            elif note_evt.tag == "staccato" and k == 0:
                n_sub.articulations.append(articulations.Staccato())

            if note_evt.is_accent and k == 0:
                n_sub.articulations.append(articulations.Accent())

            # Tied Note Chunks
            if num_chunks > 1:
                if k == 0:
                    n_sub.tie = tie.Tie('start')
                elif k == num_chunks - 1:
                    n_sub.tie = tie.Tie('stop')
                else:
                    n_sub.tie = tie.Tie('continue')

            curr_measure.append(n_sub)
            curr_m_fill += chunk_dur
            current_time_q += chunk_dur

            if curr_m_fill >= measure_capacity:
                consolidate_measure_notation(curr_measure, measure_capacity, is_compound)
                m21_part.append(curr_measure)
                curr_measure_num += 1
                curr_measure = stream.Measure(number=curr_measure_num)
                curr_m_fill = fractions.Fraction(0, 1)

    # 4. Fill remaining space in the final measure if incomplete
    if curr_m_fill > 0 and curr_m_fill < measure_capacity:
        remaining_q = measure_capacity - curr_m_fill
        rest_chunks = decompose_duration_engraver_rules(remaining_q, curr_m_fill, measure_capacity, is_compound)
        for r_dur in rest_chunks:
            r = note.Rest()
            r.quarterLength = float(r_dur)
            curr_measure.append(r)
            curr_m_fill += r_dur

    if len(curr_measure.notesAndRests) > 0:
        consolidate_measure_notation(curr_measure, measure_capacity, is_compound)
        m21_part.append(curr_measure)

    m21_score.append(m21_part)

    m21_score.write('musicxml', fp=output_xml_path)

    # 5. MusicXML Technical & Tablature Injection
    sanitize_and_inject_tablature(
        output_xml_path,
        artist_name,
        song_title,
        '4_string_standard',
        level=target_level,
        snapped_layer=note_layer,
        expressive_data=expressive_data,
    )
