import re
import fractions
import xml.etree.ElementTree as ET
from music21 import note


def idiomatic_rhythm_snap(raw_dur_q, level=5, is_compound=False):
    """
    Quantizes raw note durations into musically idiomatic fractional quarter lengths
    based on the selected complexity/articulation level.
    """
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
        elif 0.85 <= raw_dur_q <= 1.20: return fractions.Fraction(1, 1)
        elif raw_dur_q < 0.35: return fractions.Fraction(1, 4)
    else:
        if 0.38 <= raw_dur_q <= 0.65: return fractions.Fraction(1, 2)
        if 0.85 <= raw_dur_q <= 1.15: return fractions.Fraction(1, 1)
        if 0.70 <= raw_dur_q < 0.85: return fractions.Fraction(3, 4)
        if raw_dur_q < 0.38: return fractions.Fraction(1, 4)
        
    eighth_steps = max(1, int(round(raw_dur_q * 2)))
    return fractions.Fraction(eighth_steps, 2)


def decompose_duration_engraver_rules(dur_quarter, m_fill_offset, m_capacity, is_compound=False):
    """
    Decomposes quarter lengths across measure boundaries and beat centers (Beat 3 in 4/4)
    following standard music engraving rules.
    """
    units = []
    rem = fractions.Fraction(dur_quarter).limit_denominator(16)
    curr_offset = fractions.Fraction(m_fill_offset).limit_denominator(16)

    while rem > 0:
        space = m_capacity - curr_offset
        if space <= 0: break

        if not is_compound and m_capacity == fractions.Fraction(4, 1):
            # Force boundary split at Beat 3 (offset 2.0)
            if curr_offset < fractions.Fraction(2, 1) and (curr_offset + rem) > fractions.Fraction(2, 1):
                take = fractions.Fraction(2, 1) - curr_offset
            else:
                take = min(rem, space)
        else:
            take = min(rem, space)

        units.append(take)
        rem -= take
        curr_offset += take
        
    return units if units else [fractions.Fraction(1, 4)]


def consolidate_measure_notation(measure):
    """
    Consolidates consecutive rests and tied notes of matching pitch within a measure,
    preserving cross-measure tie definitions and accurate MusicXML types.
    """
    elems = list(measure.notesAndRests)
    if not elems: return
    i = 0
    while i < len(elems) - 1:
        curr_el, next_el = elems[i], elems[i + 1]

        if isinstance(curr_el, note.Rest) and isinstance(next_el, note.Rest):
            curr_el.quarterLength += next_el.quarterLength
            measure.remove(next_el)
            elems.pop(i + 1)
            continue

        if (isinstance(curr_el, note.Note) and isinstance(next_el, note.Note) and
                curr_el.pitch.midi == next_el.pitch.midi and curr_el.tie and
                curr_el.tie.type in ['start', 'continue']):

            curr_el.quarterLength += next_el.quarterLength

            c_type = curr_el.tie.type
            n_type = next_el.tie.type if next_el.tie else None

            if c_type == 'start' and n_type == 'stop':
                curr_el.tie = None
            elif c_type == 'start' and n_type == 'continue':
                curr_el.tie.type = 'start'
            elif c_type == 'continue' and n_type == 'stop':
                curr_el.tie.type = 'stop'
            elif c_type == 'continue' and n_type == 'continue':
                curr_el.tie.type = 'continue'

            measure.remove(next_el)
            elems.pop(i + 1)
            continue

        i += 1


def sanitize_and_inject_tablature(xml_path, artist_name, song_title, tuning_type, level=5):
    """
    Performs full DOM post-processing:
    1. Removes binary wrapper headers and non-XML string noise.
    2. Injects proper score metadata, bass clef octave transposition (-1), and staff tuning details.
    3. Converts internal string/fret markers (S1:F3) into native <technical> tags.
    4. Enforces valid hammer-on/pull-off spanners (excluding open strings).
    5. Synchronizes sound <tie> elements with visual <tied> notation tags.
    6. Purges leftover dynamics attributes and orphaned lyrics.
    """
    # Phase 1: Raw String Pre-pass
    try:
        with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        xml_start_idx = content.find('<?xml')
        if xml_start_idx == -1:
            xml_start_idx = content.find('<score-partwise')

        xml_end_idx = content.rfind('</score-partwise>')

        if xml_start_idx != -1 and xml_end_idx != -1:
            clean_xml = content[xml_start_idx : xml_end_idx + len('</score-partwise>')]
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(clean_xml)
    except Exception as e:
        print(f"Warning during string pre-clean pass: {e}")

    # Phase 2: XML DOM Processing
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ""

    root.tag = f"{ns}score-partwise"
    part_list = root.find(f"{ns}part-list")
    if part_list is None:
        part_list = ET.SubElement(root, f"{ns}part-list")
    
    score_part = part_list.find(f"{ns}score-part")
    if score_part is None:
        score_part = ET.SubElement(part_list, f"{ns}score-part", attrib={"id": "P1"})
    else:
        score_part.attrib["id"] = "P1"

    part_name = score_part.find(f"{ns}part-name")
    if part_name is None:
        part_name = ET.SubElement(score_part, f"{ns}part-name")
    part_name.text = "Electric Bass"

    work_elem = root.find(f"{ns}work")
    if work_elem is None:
        work_elem = ET.Element(f"{ns}work")
        root.insert(0, work_elem)
    work_title = work_elem.find(f"{ns}work-title")
    if work_title is None:
        work_title = ET.SubElement(work_elem, f"{ns}work-title")
    work_title.text = song_title

    ident = root.find(f"{ns}identification")
    if ident is None:
        ident = ET.SubElement(root, f"{ns}identification")
    creator_elem = ident.find(f"{ns}creator")
    if creator_elem is None:
        creator_elem = ET.SubElement(ident, f"{ns}creator", attrib={"type": "composer"})
    creator_elem.text = artist_name

    # Set electric bass transposition (-1 octave) and staff details
    first_part = root.find(f"{ns}part")
    if first_part is not None:
        for measure in first_part.findall(f"{ns}measure"):
            attrs = measure.find(f"{ns}attributes")
            if attrs is not None:
                clef_elem = attrs.find(f"{ns}clef")
                if clef_elem is not None:
                    oct_change = clef_elem.find(f"{ns}clef-octave-change")
                    if oct_change is None:
                        oct_change = ET.SubElement(clef_elem, f"{ns}clef-octave-change")
                    oct_change.text = "-1"

        first_measure = first_part.find(f"{ns}measure")
        if first_measure is not None:
            attrs = first_measure.find(f"{ns}attributes")
            if attrs is None:
                attrs = ET.Element(f"{ns}attributes")
                first_measure.insert(0, attrs)

            staff_details = attrs.find(f"{ns}staff-details")
            if staff_details is None:
                staff_details = ET.SubElement(attrs, f"{ns}staff-details")
            
            staff_lines = staff_details.find(f"{ns}staff-lines")
            if staff_lines is None:
                staff_lines = ET.SubElement(staff_details, f"{ns}staff-lines")
            staff_lines.text = str(5 if tuning_type == '5_string_low_b' else 4)
            
            tunings = {'5_string_low_b': [('G', 2), ('D', 2), ('A', 1), ('E', 1), ('B', 0)],
                       '4_string_drop_d': [('G', 2), ('D', 2), ('A', 1), ('D', 1)],
                       '4_string_standard': [('G', 2), ('D', 2), ('A', 1), ('E', 1)]}.get(tuning_type, [('G', 2), ('D', 2), ('A', 1), ('E', 1)])
            
            for st in list(staff_details.findall(f"{ns}staff-tuning")):
                staff_details.remove(st)

            for s_idx, (step, oct_val) in enumerate(tunings, 1):
                s_tuning = ET.SubElement(staff_details, f"{ns}staff-tuning", attrib={"line": str(s_idx)})
                ET.SubElement(s_tuning, f"{ns}tuning-step").text = step
                ET.SubElement(s_tuning, f"{ns}tuning-octave").text = str(oct_val)

    prev_tech_elem = None
    prev_string_num = None
    prev_fret_num = None

    for part in root.findall(f"{ns}part"):
        part.attrib["id"] = "P1"
        for measure in part.findall(f"{ns}measure"):
            for dummy_rest in list(measure.findall(f"{ns}note")):
                if dummy_rest.attrib.get("print-object") == "no":
                    measure.remove(dummy_rest)

            for note_elem in list(measure.findall(f"{ns}note")):
                if note_elem.find(f"{ns}rest") is not None:
                    prev_tech_elem = None
                    prev_string_num = None
                    prev_fret_num = None
                    continue

                if 'dynamics' in note_elem.attrib:
                    del note_elem.attrib['dynamics']

                # Synchronize sound <tie> elements with visual <notations><tied> elements
                for tie_elem in note_elem.findall(f"{ns}tie"):
                    t_type = tie_elem.attrib.get('type')
                    if t_type:
                        notations_elem = note_elem.find(f"{ns}notations")
                        if notations_elem is None:
                            notations_elem = ET.SubElement(note_elem, f"{ns}notations")
                        if notations_elem.find(f"{ns}tied[@type='{t_type}']") is None:
                            ET.SubElement(notations_elem, f"{ns}tied", attrib={"type": t_type})

                string_num, fret_num = None, None
                
                for lyric_elem in list(note_elem.findall(f"{ns}lyric")):
                    text_elem = lyric_elem.find(f"{ns}text")
                    if text_elem is not None and text_elem.text and 'S' in text_elem.text and ':F' in text_elem.text:
                        try:
                            txt = text_elem.text.strip()
                            s_part, f_part = txt.split(':F')
                            string_num = s_part.replace('S', '').strip()
                            fret_num = f_part.strip()
                        except Exception:
                            pass
                    note_elem.remove(lyric_elem)

                if string_num is not None and fret_num is not None:
                    notations = note_elem.find(f"{ns}notations")
                    if notations is None:
                        notations = ET.SubElement(note_elem, f"{ns}notations")
                    technical = notations.find(f"{ns}technical")
                    if technical is None:
                        technical = ET.SubElement(notations, f"{ns}technical")

                    for old_s in list(technical.findall(f"{ns}string")):
                        technical.remove(old_s)
                    for old_f in list(technical.findall(f"{ns}fret")):
                        technical.remove(old_f)

                    ET.SubElement(technical, f"{ns}string").text = str(string_num)
                    ET.SubElement(technical, f"{ns}fret").text = str(fret_num)

                    # Guard: Require non-zero frets for hammer-ons and pull-offs
                    if (level >= 3 and prev_tech_elem is not None and
                        prev_string_num == string_num and
                        prev_fret_num is not None and fret_num != prev_fret_num):
                        
                        p_fret = int(prev_fret_num)
                        c_fret = int(fret_num)

                        if p_fret > 0 and c_fret > 0:
                            if c_fret > p_fret:
                                ET.SubElement(prev_tech_elem, f"{ns}hammer-on", attrib={"type": "start", "number": "1"}).text = "H"
                                ET.SubElement(technical, f"{ns}hammer-on", attrib={"type": "stop", "number": "1"}).text = "H"
                            elif c_fret < p_fret:
                                ET.SubElement(prev_tech_elem, f"{ns}pull-off", attrib={"type": "start", "number": "1"}).text = "P"
                                ET.SubElement(technical, f"{ns}pull-off", attrib={"type": "stop", "number": "1"}).text = "P"

                    prev_tech_elem = technical
                    prev_string_num = string_num
                    prev_fret_num = fret_num
                else:
                    prev_tech_elem = None
                    prev_string_num = None
                    prev_fret_num = None

                notehead = note_elem.find(f"{ns}notehead")
                if notehead is not None and notehead.text in ["cross", "x"]:
                    if level <= 1:
                        note_elem.remove(notehead)
                    else:
                        notehead.text = "x"

                notations = note_elem.find(f"{ns}notations")
                if notations is not None and len(notations) == 0:
                    note_elem.remove(notations)

    for elem in root.iter():
        if elem.text and 'stems_' in elem.text.lower():
            elem.text = re.sub(r'(?i)stems_?', '', elem.text).strip()

    if ns:
        ET.register_namespace('', ns.strip('{}'))
    if hasattr(ET, 'indent'):
        ET.indent(tree, space="  ")
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
