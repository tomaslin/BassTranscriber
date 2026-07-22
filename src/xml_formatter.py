import re
import fractions
import xml.etree.ElementTree as ET
from music21 import note, tie


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


def decompose_duration_engraver_rules(dur_quarter, m_fill_offset, m_capacity, is_compound=False):
    """Decomposes duration values across measure and half-measure boundaries."""
    units = []
    rem = fractions.Fraction(dur_quarter).limit_denominator(16)
    curr_offset = fractions.Fraction(m_fill_offset).limit_denominator(16)

    while rem > 0:
        space = m_capacity - curr_offset
        if space <= 0:
            break

        half_capacity = fractions.Fraction(3, 1) if is_compound and m_capacity == 6 else fractions.Fraction(2, 1)
        if curr_offset < half_capacity < (curr_offset + rem):
            take = half_capacity - curr_offset
        else:
            take = min(rem, space)

        units.append(take)
        rem -= take
        curr_offset += take

    return units or [fractions.Fraction(1, 4)]


def consolidate_measure_notation(measure, max_rest_units=2520):
    """Consolidates rests and adjacent tied notes within a measure."""
    elems = list(measure.notesAndRests)
    if not elems:
        return

    max_rest_q = fractions.Fraction(max_rest_units, 5040)
    consolidated, curr = [], elems[0]

    for next_el in elems[1:]:
        if isinstance(curr, note.Note) and isinstance(next_el, note.Rest) and next_el.quarterLength <= max_rest_q:
            curr.quarterLength += next_el.quarterLength
        elif isinstance(curr, note.Rest) and isinstance(next_el, note.Rest):
            curr.quarterLength += next_el.quarterLength
        elif (isinstance(curr, note.Note) and isinstance(next_el, note.Note) and
              curr.pitch.midi == next_el.pitch.midi and curr.tie and
              curr.tie.type in ['start', 'continue']):

            curr.quarterLength += next_el.quarterLength
            c_type, n_type = curr.tie.type, next_el.tie.type if next_el.tie else None

            if c_type == 'start' and n_type == 'stop':
                curr.tie = None
            elif c_type == 'start' and n_type == 'continue':
                curr.tie = tie.Tie('start')
            elif c_type == 'continue' and n_type == 'stop':
                curr.tie = tie.Tie('stop')
            elif c_type == 'continue' and n_type == 'continue':
                curr.tie = tie.Tie('continue')
        else:
            consolidated.append(curr)
            curr = next_el

    consolidated.append(curr)

    # Rebuild measure elements efficiently
    measure.elements = tuple(consolidated)


def sanitize_and_inject_tablature(xml_path, artist_name, song_title, tuning_type, level=5):
    """Injects metadata and updates staff tuning and tablature annotations in MusicXML."""
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

    work_elem = root.find(f"{ns}work") or ET.Element(f"{ns}work")
    if work_elem not in root: root.insert(0, work_elem)
    set_or_create(work_elem, "work-title", song_title)

    # Clean up and deduplicate headers/creators
    ident = root.find(f"{ns}identification") or ET.SubElement(root, f"{ns}identification")
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

    first_part = root.find(f"{ns}part")
    if first_part is not None:
        first_measure = first_part.find(f"{ns}measure")
        if first_measure is not None:
            attrs = first_measure.find(f"{ns}attributes") or ET.Element(f"{ns}attributes")
            if attrs not in first_measure: first_measure.insert(0, attrs)

            # Ensure Bass Clef (F4) is explicitly present
            clef_elem = attrs.find(f"{ns}clef")
            if clef_elem is None:
                clef_elem = ET.SubElement(attrs, f"{ns}clef")
            set_or_create(clef_elem, "sign", "F")
            set_or_create(clef_elem, "line", "4")

            # Set staff tuning details
            staff_details = attrs.find(f"{ns}staff-details") or ET.SubElement(attrs, f"{ns}staff-details")
            set_or_create(staff_details, "staff-lines", "4")

            for st in list(staff_details.findall(f"{ns}staff-tuning")):
                staff_details.remove(st)

            tunings = [('G', 2), ('D', 2), ('A', 1), ('E', 1)]
            for idx, (step, oct_val) in enumerate(tunings, 1):
                s_tuning = ET.SubElement(staff_details, f"{ns}staff-tuning", attrib={"line": str(idx)})
                ET.SubElement(s_tuning, f"{ns}tuning-step").text = step
                ET.SubElement(s_tuning, f"{ns}tuning-octave").text = str(oct_val)

    prev_tech, prev_string, prev_fret = None, None, None

    for part in root.findall(f"{ns}part"):
        part.attrib["id"] = "P1"
        for measure in part.findall(f"{ns}measure"):
            for note_elem in list(measure.findall(f"{ns}note")):
                if note_elem.find(f"{ns}rest") is not None:
                    prev_tech, prev_string, prev_fret = None, None, None
                    continue

                string_num, fret_num = None, None
                notations = note_elem.find(f"{ns}notations")
                if notations is not None:
                    technical = notations.find(f"{ns}technical")
                    if technical is not None:
                        s_elem, f_elem = technical.find(f"{ns}string"), technical.find(f"{ns}fret")
                        if s_elem is not None and f_elem is not None:
                            string_num, fret_num = s_elem.text, f_elem.text

                # Fix self-referential legato bug: Only add pull-off/hammer-on if frets are DIFFERENT
                if string_num and fret_num and level >= 3 and prev_tech and prev_string == string_num and prev_fret:
                    p_fret, c_fret = int(prev_fret), int(fret_num)
                    if p_fret > 0 and c_fret > 0 and abs(c_fret - p_fret) <= 3 and c_fret != p_fret:
                        h_type = "hammer-on" if c_fret > p_fret else "pull-off"
                        label = "H" if c_fret > p_fret else "P"
                        ET.SubElement(prev_tech, f"{ns}{h_type}", attrib={"type": "start", "number": "1"}).text = label
                        ET.SubElement(technical, f"{ns}{h_type}", attrib={"type": "stop", "number": "1"}).text = label

                prev_tech, prev_string, prev_fret = technical if string_num else None, string_num, fret_num

    if ns:
        ET.register_namespace('', ns.strip('{}'))
    if hasattr(ET, 'indent'):
        ET.indent(tree, space="  ")

    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
