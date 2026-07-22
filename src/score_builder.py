import fractions
from music21 import stream, note, chord, meter, instrument, metadata, tie, articulations, dynamics, tempo, duration

from note_event import NoteEvent
from pitch_theory import get_directional_enharmonic_pitch
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
    time_sig_str: str = "4/4",
    tuning_type: str = "4_string_standard",
):
    """
    Builds music21 score stream, eliminates temporal drift, handles anacrusis,
    attaches fingerings/harmonics, defines Electric Bass MIDI Program 33, and exports MusicXML.
    """
    sec_per_quarter = (60.0 / bpm) if bpm > 0 else 0.5

    time_sig_map = {
        '4/4': fractions.Fraction(4, 1),
        '3/4': fractions.Fraction(3, 1),
        '5/4': fractions.Fraction(5, 1),
        '6/8': fractions.Fraction(3, 1),
        '7/8': fractions.Fraction(7, 2),
        '12/8': fractions.Fraction(6, 1),
    }
    measure_capacity = time_sig_map.get(time_sig_str, fractions.Fraction(6, 1) if is_compound else fractions.Fraction(4, 1))

    m21_score = stream.Score()
    m21_part = stream.Part(id="P1")

    # MIDI Instrument Definition: Electric Bass (Program 33)
    bass_inst = instrument.ElectricBass()
    bass_inst.midiProgram = 33
    m21_part.insert(0.0, bass_inst)

    m21_score.metadata = metadata.Metadata()
    m21_score.metadata.title = song_title
    m21_score.metadata.composer = artist_name

    curr_measure_num = 1
    curr_measure = stream.Measure(number=curr_measure_num)
    curr_measure.append(detected_key)
    curr_measure.append(meter.TimeSignature(time_sig_str))
    curr_measure.append(tempo.MetronomeMark(number=round(bpm)))

    if note_layer and note_layer[0].is_pickup:
        curr_measure.number = 0
        curr_measure.padAsAnacrusis()

    curr_m_fill = fractions.Fraction(0, 1)
    current_time_q = fractions.Fraction(0, 1)
    last_dynamic = None
    prev_midi = None

    for i, note_evt in enumerate(note_layer):
        start_q = fractions.Fraction(round((note_evt.start / sec_per_quarter) * 4), 4)
        raw_dur_q = max(0.25, note_evt.duration / sec_per_quarter)
        dur_q = idiomatic_rhythm_snap(raw_dur_q, level=target_level, is_compound=is_compound)

        if start_q < current_time_q:
            start_q = current_time_q

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

        if instant_bpms is not None and beat_times is not None and i < len(beat_times):
            curr_bpm = round(instant_bpms[min(i, len(instant_bpms) - 1)])
            if abs(curr_bpm - bpm) > 6.0 and curr_m_fill == 0:
                bpm = curr_bpm
                curr_measure.append(tempo.MetronomeMark(number=bpm))

        fret_pos = fretboard_path[i] if i < len(fretboard_path) else (4, 0, 0)

        pitches = note_evt.pitches if note_evt.pitches else [note_evt.pitch]
        m21_pitches = [get_directional_enharmonic_pitch(p, detected_key, prev_midi) for p in pitches]
        prev_midi = pitches[0]

        note_chunks = decompose_duration_engraver_rules(dur_q, curr_m_fill, measure_capacity, is_compound)
        num_chunks = len(note_chunks)

        if note_evt.dynamic_mark and note_evt.dynamic_mark != last_dynamic:
            curr_measure.append(dynamics.Dynamic(note_evt.dynamic_mark))
            last_dynamic = note_evt.dynamic_mark

        for k, chunk_dur in enumerate(note_chunks):
            if len(m21_pitches) == 1:
                elem_sub = note.Note(m21_pitches[0])
                s_idx, f_val, fing_val = 4, 0, 0
                if isinstance(fret_pos, tuple):
                    s_idx = fret_pos[0]
                    f_val = fret_pos[1]
                    fing_val = fret_pos[2] if len(fret_pos) > 2 else 0

                elem_sub.articulations.extend([
                    articulations.StringIndication(s_idx),
                    articulations.FretIndication(f_val)
                ])
                if fing_val > 0:
                    elem_sub.articulations.append(articulations.Fingering(fing_val))
            else:
                elem_sub = chord.Chord(m21_pitches)
                if isinstance(fret_pos, tuple):
                    s_idx, f_val, fing_val = fret_pos[0], fret_pos[1], (fret_pos[2] if len(fret_pos) > 2 else 0)
                    for chord_note in elem_sub.notes:
                        chord_note.articulations.extend([
                            articulations.StringIndication(s_idx),
                            articulations.FretIndication(f_val)
                        ])
                        if fing_val > 0:
                            chord_note.articulations.append(articulations.Fingering(fing_val))

            if note_evt.is_harmonic:
                elem_sub.articulations.append(articulations.Harmonic())

            # Safely handle triplet tuplet attachments
            if note_evt.is_triplet and k == 0 and float(chunk_dur) >= 0.125:
                elem_sub.duration = duration.Duration(float(chunk_dur))
                elem_sub.duration.appendTuplet(duration.Tuplet(3, 2))
            else:
                elem_sub.quarterLength = max(0.0625, float(chunk_dur))

            if note_evt.tag == "ghost":
                elem_sub.notehead = 'x'
            elif note_evt.tag == "staccato" and k == 0:
                elem_sub.articulations.append(articulations.Staccato())

            if note_evt.is_accent and k == 0:
                elem_sub.articulations.append(articulations.Accent())

            if num_chunks > 1:
                if k == 0:
                    elem_sub.tie = tie.Tie('start')
                elif k == num_chunks - 1:
                    elem_sub.tie = tie.Tie('stop')
                else:
                    elem_sub.tie = tie.Tie('continue')

            curr_measure.append(elem_sub)
            curr_m_fill += chunk_dur
            current_time_q += chunk_dur

            if curr_m_fill >= measure_capacity:
                consolidate_measure_notation(curr_measure, measure_capacity, is_compound)
                m21_part.append(curr_measure)
                curr_measure_num += 1
                curr_measure = stream.Measure(number=curr_measure_num)
                curr_m_fill = fractions.Fraction(0, 1)

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

    sanitize_and_inject_tablature(
        output_xml_path,
        artist_name,
        song_title,
        tuning_type,
        level=target_level,
        snapped_layer=note_layer,
        expressive_data=expressive_data,
        time_sig_str=time_sig_str,
    )
