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
    """
    Decomposes duration values across measure and invisible half-measure boundaries.
    Ensures every returned chunk is a valid, non-complex MusicXML duration.
    """
    units = []
    rem = fractions.Fraction(dur_quarter).limit_denominator(16)
    curr_offset = fractions.Fraction(m_fill_offset).limit_denominator(16)

    while rem > 0:
        space = m_capacity - curr_offset
        if space <= 0:
            space = m_capacity
            curr_offset = fractions.Fraction(0, 1)

        if is_compound:
            half_capacity = fractions.Fraction(3, 1) if m_capacity == 6 else fractions.Fraction(6, 1)
        else:
            half_capacity = fractions.Fraction(2, 1)

        if curr_offset < half_capacity < (curr_offset + rem):
            take = half_capacity - curr_offset
        else:
            take = min(rem, space)

        sub_chunks = _split_into_valid_units(take)
        units.extend(sub_chunks)

        rem -= take
        curr_offset = (curr_offset + take) % m_capacity

    return units or [fractions.Fraction(1, 4)]


def consolidate_measure_notation(measure, m_capacity=fractions.Fraction(4, 1), is_compound=False):
    """
    Consolidates rests according to standard engraving rules:
    1. Fully empty measures convert to a single Full Measure Rest.
    2. Rests never consolidate across the measure midpoint (Beat 3 in 4/4).
    3. Merges rests only if the resulting length is a valid single duration unit.
    4. Merges tied notes of identical pitch cleanly.
    """
    elems = list(measure.notesAndRests)
    if not elems:
        return

    # 1. Full Measure Rest Check
    total_q = sum(fractions.Fraction(e.quarterLength).limit_denominator(16) for e in elems)
    all_rests = all(isinstance(e, note.Rest) for e in elems)
    if all_rests and total_q == m_capacity:
        full_rest = note.Rest()
        full_rest.quarterLength = float(m_capacity)
        full_rest.fullMeasure = True
        measure.elements = (full_rest,)
        return

    # 2. Beat-Aware Rest Consolidation
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

    # 3. Simplify tied notes of identical pitch within measure
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
    Injects metadata, tab tuning, and extended technique marks (slap, pop, palm mute,
    hammer-ons, pull-offs, slides, bends, and ghost noteheads) into MusicXML.
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

    # Metadata
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

    first_part = root.find(f"{ns}part")
    if first_part is not None:
        first_part.attrib["id"] = score_part_id
        first_measure = first_part.find(f"{ns}measure")
        if first_measure is not None:
            attrs = first_measure.find(f"{ns}attributes")
            if attrs is None:
                attrs = ET.Element(f"{ns}attributes")
                first_measure.insert(0, attrs)

            key_elem = attrs.find(f"{ns}key")
            if key_elem is None:
                key_elem = ET.SubElement(attrs, f"{ns}key")
                ET.SubElement(key_elem, f"{ns}fifths").text = "0"
                ET.SubElement(key_elem, f"{ns}mode").text = "major"

            time_elem = attrs.find(f"{ns}time")
            if time_elem is None:
                time_elem = ET.SubElement(attrs, f"{ns}time")
                ET.SubElement(time_elem, f"{ns}beats").text = "4"
                ET.SubElement(time_elem, f"{ns}beat-type").text = "4"

            clef_elem = attrs.find(f"{ns}clef")
            if clef_elem is None:
                clef_elem = ET.SubElement(attrs, f"{ns}clef")
            set_or_create(clef_elem, "sign", "F")
            set_or_create(clef_elem, "line", "4")

            staff_details = attrs.find(f"{ns}staff-details")
            if staff_details is None:
                staff_details = ET.SubElement(attrs, f"{ns}staff-details")
            set_or_create(staff_details, "staff-lines", "4")

            for st in list(staff_details.findall(f"{ns}staff-tuning")):
                staff_details.remove(st)

            tunings = [('G', 2), ('D', 2), ('A', 1), ('E', 1)]
            for idx, (step, oct_val) in enumerate(tunings, 1):
                s_tuning = ET.SubElement(staff_details, f"{ns}staff-tuning", attrib={"line": str(idx)})
                ET.SubElement(s_tuning, f"{ns}tuning-step").text = step
                ET.SubElement(s_tuning, f"{ns}tuning-octave").text = str(oct_val)

    # Injections: Slap/Pop, Palm Mute, Legato, Slides & Bends
    note_idx = 0
    for part in root.findall(f"{ns}part"):
        part.attrib["id"] = score_part_id
        for measure in part.findall(f"{ns}measure"):
            for note_elem in list(measure.findall(f"{ns}note")):
                if note_elem.find(f"{ns}rest") is not None:
                    continue

                if snapped_layer and note_idx < len(snapped_layer):
                    evt = snapped_layer[note_idx]

                    notations = note_elem.find(f"{ns}notations")
                    if notations is None:
                        notations = ET.SubElement(note_elem, f"{ns}notations")

                    technical = notations.find(f"{ns}technical")
                    if technical is None:
                        technical = ET.SubElement(notations, f"{ns}technical")

                    # Extended Techniques
                    if evt.tag == "slap":
                        ET.SubElement(technical, f"{ns}slap")
                    elif evt.tag == "pop":
                        ET.SubElement(technical, f"{ns}pop")
                    elif evt.tag == "palm_mute":
                        ET.SubElement(technical, f"{ns}other-technical").text = "P.M."

                    # Ghost Notes
                    if evt.tag == "ghost":
                        nh = note_elem.find(f"{ns}notehead")
                        if nh is None:
                            nh = ET.SubElement(note_elem, f"{ns}notehead")
                        nh.text = "x"

                    # Expressive Pitch Articulations
                    if evt.is_legato:
                        ET.SubElement(technical, f"{ns}hammer-on", attrib={"type": "start", "number": "1"}).text = "H"
                        slur = ET.SubElement(notations, f"{ns}slur", attrib={"type": "start", "number": "1"})
                    if evt.is_slide:
                        ET.SubElement(notations, f"{ns}slide", attrib={"type": "start", "number": "1"}).text = "slide"

                    # Pitch Bends
                    if evt.bends and any(abs(b) > 0.2 for b in evt.bends):
                        bend_elem = ET.SubElement(technical, f"{ns}bend")
                        bend_alter = ET.SubElement(bend_elem, f"{ns}bend-alter")
                        bend_alter.text = str(round(max(evt.bends), 1))

                    note_idx += 1

    if ns:
        ET.register_namespace('', ns.strip('{}'))
    if hasattr(ET, 'indent'):
        ET.indent(tree, space="  ")

    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
