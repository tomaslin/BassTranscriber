import fractions
import xml.etree.ElementTree as ET
from music21 import note

VALID_SINGLE_DURATIONS = [
    fractions.Fraction(4, 1),   # Whole
    fractions.Fraction(3, 1),   # Dotted Half
    fractions.Fraction(2, 1),   # Half
    fractions.Fraction(3, 2),   # Dotted Quarter (1.5)
    fractions.Fraction(1, 1),   # Quarter
    fractions.Fraction(3, 4),   # Dotted 8th (0.75)
    fractions.Fraction(1, 2),   # 8th (0.5)
    fractions.Fraction(1, 4),   # 16th (0.25)
]


def idiomatic_rhythm_snap(raw_dur_q, level=5, is_compound=False):
    """Quantizes note durations into musically idiomatic fractional quarter lengths."""
    if level == 0:
        if raw_dur_q >= 3.0: return fractions.Fraction(4, 1)
        if raw_dur_q >= 1.5: return fractions.Fraction(2, 1)
        return fractions.Fraction(1, 1)

    if level == 1:
        if raw_dur_q >= 3.2: return fractions.Fraction(4, 1)
        if raw_dur_q >= 1.7: return fractions.Fraction(2, 1)
        if raw_dur_q >= 0.75: return fractions.Fraction(1, 1)
        return fractions.Fraction(1, 2)

    if is_compound:
        if 0.35 <= raw_dur_q <= 0.65: return fractions.Fraction(1, 2)
        if 0.85 <= raw_dur_q <= 1.20: return fractions.Fraction(1, 1)
        if raw_dur_q < 0.35: return fractions.Fraction(1, 4)
    else:
        if 0.38 <= raw_dur_q <= 0.65: return fractions.Fraction(1, 2)
        if 0.85 <= raw_dur_q <= 1.15: return fractions.Fraction(1, 1)
        if 0.70 <= raw_dur_q < 0.85: return fractions.Fraction(3, 4)
        if raw_dur_q < 0.38: return fractions.Fraction(1, 4)

    eighth_steps = max(1, int(round(raw_dur_q * 2)))
    return fractions.Fraction(eighth_steps, 2)


def _split_into_valid_units(dur_q):
    """Decomposes any quarter duration into a list of standard MusicXML duration units."""
    units = []
    rem = fractions.Fraction(dur_q).limit_denominator(16)
    for v in VALID_SINGLE_DURATIONS:
        while rem >= v:
            units.append(v)
            rem -= v
    if rem > 0:
        units.append(rem)
    return units or [fractions.Fraction(1, 4)]


def decompose_duration_engraver_rules(dur_quarter, m_fill_offset, m_capacity, is_compound=False):
    """Decomposes duration values across measure boundaries and midpoints."""
    units = []
    rem = fractions.Fraction(dur_quarter).limit_denominator(16)
    curr_offset = fractions.Fraction(m_fill_offset).limit_denominator(16)

    while rem > 0:
        space = m_capacity - curr_offset
        if space <= 0:
            space = m_capacity
            curr_offset = fractions.Fraction(0, 1)

        if curr_offset == 0 and rem >= space:
            max_take = space
        elif not is_compound and m_capacity == fractions.Fraction(4, 1):
            midpoint = fractions.Fraction(2, 1)
            if curr_offset < midpoint < (curr_offset + rem):
                max_take = midpoint - curr_offset
            elif curr_offset % fractions.Fraction(1, 1) != 0:
                next_beat = fractions.Fraction(int(curr_offset) + 1, 1)
                max_take = min(next_beat - curr_offset, rem)
            else:
                max_take = min(rem, space)
        elif is_compound:
            pulse = fractions.Fraction(3, 2)
            next_pulse = ((curr_offset // pulse) + 1) * pulse
            if curr_offset < next_pulse < (curr_offset + rem):
                max_take = next_pulse - curr_offset
            else:
                max_take = min(rem, space)
        else:
            half_capacity = m_capacity / 2
            if curr_offset < half_capacity < (curr_offset + rem):
                max_take = half_capacity - curr_offset
            elif curr_offset % fractions.Fraction(1, 1) != 0:
                next_beat = fractions.Fraction(int(curr_offset) + 1, 1)
                max_take = min(next_beat - curr_offset, rem)
            else:
                max_take = min(rem, space)

        sub_chunks = _split_into_valid_units(max_take)
        units.extend(sub_chunks)

        rem -= max_take
        curr_offset = (curr_offset + max_take) % m_capacity

    return units or [fractions.Fraction(1, 4)]


def consolidate_measure_notation(measure, m_capacity=fractions.Fraction(4, 1), is_compound=False):
    """Consolidates rests and tied notes across measure beat boundaries."""
    elems = list(measure.notesAndRests)
    if not elems:
        return

    total_q = sum(fractions.Fraction(e.quarterLength).limit_denominator(16) for e in elems)
    all_rests = all(isinstance(e, note.Rest) for e in elems)
    if all_rests and total_q == m_capacity:
        full_rest = note.Rest()
        full_rest.quarterLength = float(m_capacity)
        full_rest.fullMeasure = True
        measure.elements = (full_rest,)
        return

    consolidated = []
    curr_offset = fractions.Fraction(0, 1)
    i = 0

    while i < len(elems):
        curr = elems[i]
        curr_q = fractions.Fraction(curr.quarterLength).limit_denominator(16)

        if isinstance(curr, note.Rest) and (i + 1 < len(elems)) and isinstance(elems[i + 1], note.Rest):
            next_rest = elems[i + 1]
            next_q = fractions.Fraction(next_rest.quarterLength).limit_denominator(16)
            combined_q = curr_q + next_q

            midpoint = m_capacity / 2
            crosses_midpoint = (curr_offset < midpoint) and ((curr_offset + combined_q) > midpoint)

            if not crosses_midpoint and combined_q in VALID_SINGLE_DURATIONS:
                curr.quarterLength = float(combined_q)
                curr_offset += combined_q
                consolidated.append(curr)
                i += 2
                continue

        consolidated.append(curr)
        curr_offset += curr_q
        i += 1

    merged = []
    idx = 0
    while idx < len(consolidated):
        elem = consolidated[idx]
        if (
            isinstance(elem, note.Note)
            and elem.tie
            and elem.tie.type in ['start', 'continue']
            and idx + 1 < len(consolidated)
        ):
            next_elem = consolidated[idx + 1]
            if isinstance(next_elem, note.Note) and next_elem.pitch == elem.pitch:
                comb_q = fractions.Fraction(elem.quarterLength + next_elem.quarterLength).limit_denominator(16)
                if comb_q in VALID_SINGLE_DURATIONS:
                    elem.quarterLength = float(comb_q)
                    if next_elem.tie and next_elem.tie.type == 'stop':
                        elem.tie = None
                    else:
                        elem.tie = next_elem.tie
                    idx += 1
        merged.append(elem)
        idx += 1

    measure.elements = tuple(merged)


def sanitize_and_inject_tablature(
    xml_path, artist_name, song_title, tuning_type, level=5, snapped_layer=None, expressive_data=None
):
    """
    Sanitizes MusicXML structure and formats valid dual-staff layout (Standard Notation on Staff 1 + TAB on Staff 2)
    using proper MusicXML <backup> temporal positioning and MIDI program definitions.
    """
    try:
        with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        s_idx, e_idx = content.find('<?xml'), content.rfind('</score-partwise>')
        if s_idx != -1 and e_idx != -1:
            clean_xml = content[s_idx : e_idx + len('</score-partwise>')]
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(clean_xml)
    except Exception:
        pass

    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ""

    root.tag = f"{ns}score-partwise"

    def set_or_create(parent, tag_name, text_val, attrib=None):
        elem = parent.find(f"{ns}{tag_name}")
        if elem is None:
            elem = ET.SubElement(parent, f"{ns}{tag_name}", attrib=attrib or {})
        elem.text = text_val
        return elem

    set_or_create(root, "movement-title", song_title)

    work_elem = root.find(f"{ns}work")
    if work_elem is None:
        work_elem = ET.Element(f"{ns}work")
        root.insert(0, work_elem)
    set_or_create(work_elem, "work-title", song_title)

    ident = root.find(f"{ns}identification")
    if ident is None:
        ident = ET.SubElement(root, f"{ns}identification")

    creators = ident.findall(f"{ns}creator")
    if creators:
        primary_creator = creators[0]
        primary_creator.attrib["type"] = "composer"
        primary_creator.text = artist_name
        for extra in creators[1:]:
            ident.remove(extra)
    else:
        creator = ET.SubElement(ident, f"{ns}creator", attrib={"type": "composer"})
        creator.text = artist_name

    score_part_elem = root.find(f".//{ns}score-part")
    score_part_id = score_part_elem.attrib.get("id", "P1") if score_part_elem is not None else "P1"

    # Ensure score-part contains valid midi-instrument for Electric Bass (Program 33)
    if score_part_elem is not None:
        midi_inst = score_part_elem.find(f"{ns}midi-instrument")
        if midi_inst is None:
            midi_inst = ET.SubElement(score_part_elem, f"{ns}midi-instrument", attrib={"id": f"{score_part_id}-I1"})
        set_or_create(midi_inst, "midi-channel", "1")
        set_or_create(midi_inst, "midi-program", "33")  # Electric Bass - Finger

    first_part = root.find(f"{ns}part")
    if first_part is None:
        return

    first_part.attrib["id"] = score_part_id
    measures = list(first_part.findall(f"{ns}measure"))

    for m_idx, measure in enumerate(measures, 1):
        if m_idx > 1 and (m_idx - 1) % 4 == 0:
            for existing_p in measure.findall(f"{ns}print"):
                measure.remove(existing_p)

            print_elem = ET.Element(f"{ns}print")
            if (m_idx - 1) % 16 == 0:
                print_elem.attrib["new-page"] = "yes"
            else:
                print_elem.attrib["new-system"] = "yes"
            measure.insert(0, print_elem)

        if m_idx == 1:
            attrs = measure.find(f"{ns}attributes")
            if attrs is None:
                attrs = ET.Element(f"{ns}attributes")
                measure.insert(0, attrs)

            set_or_create(attrs, "staves", "2")

            for existing_clef in list(attrs.findall(f"{ns}clef")):
                attrs.remove(existing_clef)
            for existing_sd in list(attrs.findall(f"{ns}staff-details")):
                attrs.remove(existing_sd)

            clef1 = ET.SubElement(attrs, f"{ns}clef", attrib={"number": "1"})
            ET.SubElement(clef1, f"{ns}sign").text = "F"
            ET.SubElement(clef1, f"{ns}line").text = "4"

            clef2 = ET.SubElement(attrs, f"{ns}clef", attrib={"number": "2"})
            ET.SubElement(clef2, f"{ns}sign").text = "TAB"
            ET.SubElement(clef2, f"{ns}line").text = "5"

            staff_details = ET.SubElement(attrs, f"{ns}staff-details", attrib={"number": "2"})
            ET.SubElement(staff_details, f"{ns}staff-lines").text = "4"

            tunings = [('G', 2), ('D', 2), ('A', 1), ('E', 1)]
            if tuning_type == '5_string_standard':
                tunings = [('G', 2), ('D', 2), ('A', 1), ('E', 1), ('B', 0)]
                staff_details.find(f"{ns}staff-lines").text = "5"
            elif tuning_type == '6_string_standard':
                tunings = [('C', 3), ('G', 2), ('D', 2), ('A', 1), ('E', 1), ('B', 0)]
                staff_details.find(f"{ns}staff-lines").text = "6"

            for idx_t, (step, oct_val) in enumerate(tunings, 1):
                s_tuning = ET.SubElement(staff_details, f"{ns}staff-tuning", attrib={"line": str(idx_t)})
                ET.SubElement(s_tuning, f"{ns}tuning-step").text = step
                ET.SubElement(s_tuning, f"{ns}tuning-octave").text = str(oct_val)

    note_global_idx = 0
    active_spanner_tag = None

    for part in root.findall(f"{ns}part"):
        part.attrib["id"] = score_part_id
        for measure in part.findall(f"{ns}measure"):
            m_children = list(measure)

            measure_duration = 0
            tab_elements = []

            for elem in m_children:
                if elem.tag == f"{ns}note":
                    set_or_create(elem, "staff", "1")

                    pitch_elem = elem.find(f"{ns}pitch")
                    if pitch_elem is not None:
                        step_val = pitch_elem.findtext(f"{ns}step") or "D"
                        oct_val = int(pitch_elem.findtext(f"{ns}octave") or "3")
                        if oct_val < 3 or (oct_val == 3 and step_val in ["C", "D"]):
                            set_or_create(elem, "stem", "up")
                        else:
                            set_or_create(elem, "stem", "down")

                    is_chord = elem.find(f"{ns}chord") is not None
                    dur_text = elem.findtext(f"{ns}duration")
                    if not is_chord and dur_text:
                        measure_duration += int(dur_text)

                    if elem.find(f"{ns}rest") is None and snapped_layer and note_global_idx < len(snapped_layer):
                        evt = snapped_layer[note_global_idx]
                        notations = elem.find(f"{ns}notations")
                        if notations is None:
                            notations = ET.SubElement(elem, f"{ns}notations")

                        technical = notations.find(f"{ns}technical")
                        if technical is None:
                            technical = ET.SubElement(notations, f"{ns}technical")

                        if evt.tag == "slap" and technical.find(f"{ns}slap") is None:
                            ET.SubElement(technical, f"{ns}slap")
                        elif evt.tag == "pop" and technical.find(f"{ns}pop") is None:
                            ET.SubElement(technical, f"{ns}pop")

                        if evt.tag in ["palm_mute", "let_ring"]:
                            if active_spanner_tag != evt.tag:
                                active_spanner_tag = evt.tag
                                ET.SubElement(notations, f"{ns}dashes", attrib={"type": "start", "number": "1"})

                                words_dir = ET.Element(f"{ns}direction")
                                dt = ET.SubElement(words_dir, f"{ns}direction-type")
                                ET.SubElement(dt, f"{ns}words").text = "P.M." if evt.tag == "palm_mute" else "let ring"

                                insert_pos = 0
                                for idx_e, child in enumerate(list(measure)):
                                    if child.tag.endswith(("print", "attributes")):
                                        insert_pos = idx_e + 1
                                measure.insert(insert_pos, words_dir)
                        else:
                            if active_spanner_tag is not None:
                                active_spanner_tag = None
                                ET.SubElement(notations, f"{ns}dashes", attrib={"type": "stop", "number": "1"})

                        if evt.tag == "ghost":
                            set_or_create(elem, "notehead", "x")

                        if evt.is_legato:
                            if technical.find(f"{ns}hammer-on") is None:
                                ET.SubElement(technical, f"{ns}hammer-on", attrib={"type": "start", "number": "1"}).text = "H"
                            if notations.find(f"{ns}slur") is None:
                                ET.SubElement(notations, f"{ns}slur", attrib={"type": "start", "number": "1"})

                        if evt.is_slide and notations.find(f"{ns}slide") is None:
                            ET.SubElement(notations, f"{ns}slide", attrib={"type": "start", "number": "1"}).text = "slide"

                        if evt.bends and any(abs(b) > 0.1 for b in evt.bends):
                            bend_elem = ET.SubElement(technical, f"{ns}bend")
                            bend_alter = ET.SubElement(bend_elem, f"{ns}bend-alter")
                            bend_alter.text = str(round(max(evt.bends), 1))
                        elif abs(evt.microtone_cents) > 10.0:
                            bend_elem = ET.SubElement(technical, f"{ns}bend")
                            bend_alter = ET.SubElement(bend_elem, f"{ns}bend-alter")
                            bend_alter.text = str(round(evt.microtone_cents / 100.0, 2))

                        tab_note = ET.fromstring(ET.tostring(elem))
                        tab_note.find(f"{ns}staff").text = "2"
                        tab_elements.append(tab_note)

                        note_global_idx += 1
                    elif elem.find(f"{ns}rest") is not None:
                        tab_rest = ET.fromstring(ET.tostring(elem))
                        tab_rest.find(f"{ns}staff").text = "2"
                        tab_elements.append(tab_rest)

            if measure_duration > 0 and tab_elements:
                backup_elem = ET.Element(f"{ns}backup")
                dur_elem = ET.SubElement(backup_elem, f"{ns}duration")
                dur_elem.text = str(measure_duration)

                measure.append(backup_elem)
                for tab_elem in tab_elements:
                    measure.append(tab_elem)

            if active_spanner_tag is not None:
                last_note = measure.findall(f"{ns}note")[-1] if measure.findall(f"{ns}note") else None
                if last_note is not None:
                    notations = last_note.find(f"{ns}notations") or ET.SubElement(last_note, f"{ns}notations")
                    ET.SubElement(notations, f"{ns}dashes", attrib={"type": "stop", "number": "1"})
                active_spanner_tag = None

    if ns:
        ET.register_namespace('', ns.strip('{}'))
    if hasattr(ET, 'indent'):
        ET.indent(tree, space="  ")

    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
