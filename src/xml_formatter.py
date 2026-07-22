import fractions
import xml.etree.ElementTree as ET

# Key Accidentals Map (-7 to +7 fifths)
KEY_ACCIDENTALS = {
    0: {},
    1: {"F": ("1", "sharp")},
    2: {"F": ("1", "sharp"), "C": ("1", "sharp")},
    3: {"F": ("1", "sharp"), "C": ("1", "sharp"), "G": ("1", "sharp")},
    4: {"F": ("1", "sharp"), "C": ("1", "sharp"), "G": ("1", "sharp"), "D": ("1", "sharp")},
    5: {"F": ("1", "sharp"), "C": ("1", "sharp"), "G": ("1", "sharp"), "D": ("1", "sharp"), "A": ("1", "sharp")},
    6: {"F": ("1", "sharp"), "C": ("1", "sharp"), "G": ("1", "sharp"), "D": ("1", "sharp"), "A": ("1", "sharp"), "E": ("1", "sharp")},
    7: {"F": ("1", "sharp"), "C": ("1", "sharp"), "G": ("1", "sharp"), "D": ("1", "sharp"), "A": ("1", "sharp"), "E": ("1", "sharp"), "B": ("1", "sharp")},
    -1: {"B": ("-1", "flat")},
    -2: {"B": ("-1", "flat"), "E": ("-1", "flat")},
    -3: {"B": ("-1", "flat"), "E": ("-1", "flat"), "A": ("-1", "flat")},
    -4: {"B": ("-1", "flat"), "E": ("-1", "flat"), "A": ("-1", "flat"), "D": ("-1", "flat")},
    -5: {"B": ("-1", "flat"), "E": ("-1", "flat"), "A": ("-1", "flat"), "D": ("-1", "flat"), "G": ("-1", "flat")},
    -6: {"B": ("-1", "flat"), "E": ("-1", "flat"), "A": ("-1", "flat"), "D": ("-1", "flat"), "G": ("-1", "flat"), "C": ("-1", "flat")},
    -7: {"B": ("-1", "flat"), "E": ("-1", "flat"), "A": ("-1", "flat"), "D": ("-1", "flat"), "G": ("-1", "flat"), "C": ("-1", "flat"), "F": ("-1", "flat")},
}

# Bass Staff Tunings (Ordered Line 1 [Bottom] -> Line N [Top])
BASS_TUNINGS = {
    "4_string_standard": [('E', 1), ('A', 1), ('D', 2), ('G', 2)],
    "5_string_standard": [('B', 0), ('E', 1), ('A', 1), ('D', 2), ('G', 2)],
    "6_string_standard": [('B', 0), ('E', 1), ('A', 1), ('D', 2), ('G', 2), ('C', 3)],
    "drop_d": [('D', 1), ('A', 1), ('D', 2), ('G', 2)],
    "4_string_drop_d": [('D', 1), ('A', 1), ('D', 2), ('G', 2)],
}

NOTE_SCHEMA_ORDER = [
    "grace", "cue", "chord", "pitch", "rest", "unpitched",
    "duration", "tie", "instrument", "voice", "type", "dot",
    "accidental", "time-modification", "stem", "notehead",
    "staff", "beam", "notations", "lyric"
]

ATTRS_SCHEMA_ORDER = [
    "footnote", "level", "divisions", "key", "time",
    "staves", "part-symbol", "instruments", "clef",
    "staff-details", "transpose", "directive", "measure-style"
]


def reorder_children(parent, order_list, ns=""):
    """Reorders child elements according to standard MusicXML schema order."""
    def get_rank(child):
        tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        return order_list.index(tag_name) if tag_name in order_list else len(order_list)

    children = list(parent)
    children.sort(key=get_rank)
    for child in children:
        parent.remove(child)
    for child in children:
        parent.append(child)


def set_or_create(parent, tag_name, text_val, attrib=None):
    ns = parent.tag.split('}')[0] + '}' if '}' in parent.tag else ""
    elem = parent.find(f"{ns}{tag_name}")
    if elem is None:
        elem = ET.SubElement(parent, f"{ns}{tag_name}", attrib=attrib or {})
    elem.text = str(text_val)
    return elem


def add_direction_words(measure, text, insert_idx=0):
    """Safely adds a <direction><direction-type><words> element to measure."""
    ns = measure.tag.split('}')[0] + '}' if '}' in measure.tag else ""
    direction = ET.Element(f"{ns}direction", attrib={"placement": "above"})
    dt = ET.SubElement(direction, f"{ns}direction-type")
    words = ET.SubElement(dt, f"{ns}words")
    words.text = text
    measure.insert(insert_idx, direction)


def idiomatic_rhythm_snap(dur_q, level=5, is_compound=False):
    """
    Snaps a quarterLength duration to idiomatic engraver grid, enforcing
    a minimum duration threshold (1/16th or 1/32nd) to prevent micro-durations.
    """
    if isinstance(dur_q, (float, int)):
        # Round floating point noise to avoid fraction exploding
        dur_q = round(float(dur_q), 4)
        dur_q = fractions.Fraction(dur_q).limit_denominator(16 if level <= 3 else 32)
    
    MIN_DURATION = fractions.Fraction(1, 16) if level <= 3 else fractions.Fraction(1, 32)
    if dur_q < MIN_DURATION:
        return MIN_DURATION

    if is_compound:
        return fractions.Fraction(dur_q).limit_denominator(12)
    return fractions.Fraction(round(float(dur_q) * 4), 4)


def decompose_duration_engraver_rules(dur_q, curr_m_fill, measure_capacity, is_compound=False):
    """Decomposes durations into tied chunks across measure/beat boundaries safely."""
    dur_q = fractions.Fraction(dur_q).limit_denominator(32)
    curr_m_fill = fractions.Fraction(curr_m_fill).limit_denominator(32)
    measure_capacity = fractions.Fraction(measure_capacity).limit_denominator(32)
    
    MIN_CHUNK = fractions.Fraction(1, 16)
    chunks = []

    while dur_q > 0:
        rem_in_m = measure_capacity - curr_m_fill
        if rem_in_m <= 0:
            curr_m_fill = fractions.Fraction(0, 1)
            rem_in_m = measure_capacity
            
        chunk = min(dur_q, rem_in_m)
        
        # Prevent trailing micro-chunks from being created by tie decompositions
        if 0 < chunk < MIN_CHUNK and chunks:
            chunks[-1] += chunk
            curr_m_fill += chunk
            dur_q -= chunk
            break

        chunks.append(chunk)
        dur_q -= chunk
        curr_m_fill += chunk
        if curr_m_fill >= measure_capacity:
            curr_m_fill = fractions.Fraction(0, 1)

    return [c for c in chunks if c > 0]


def consolidate_measure_notation(curr_measure, measure_capacity, is_compound=False):
    """Placeholder helper for music21 measure consolidation pass."""
    pass


def sanitize_and_inject_tablature(
    xml_path,
    artist_name,
    song_title,
    tuning_type,
    level=5,
    snapped_layer=None,
    expressive_data=None,
    time_sig_str="4/4",
):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ""
    root.tag = f"{ns}score-partwise"

    # Set metadata in root obeying MusicXML schema sequence
    set_or_create(root, "movement-title", song_title)
    work_elem = root.find(f"{ns}work") or ET.SubElement(root, f"{ns}work")
    set_or_create(work_elem, "work-title", song_title)

    ident = root.find(f"{ns}identification") or ET.SubElement(root, f"{ns}identification")
    for extra in ident.findall(f"{ns}creator"):
        ident.remove(extra)
    creator = ET.SubElement(ident, f"{ns}creator", attrib={"type": "composer"})
    creator.text = artist_name

    score_part = root.find(f".//{ns}score-part")
    score_part_id = score_part.attrib.get("id", "P1") if score_part is not None else "P1"

    if score_part is not None:
        midi_inst = score_part.find(f"{ns}midi-instrument") or ET.SubElement(
            score_part, f"{ns}midi-instrument", attrib={"id": f"{score_part_id}-I1"}
        )
        set_or_create(midi_inst, "midi-channel", "1")
        set_or_create(midi_inst, "midi-program", "33")  # Electric Bass (Finger)

    first_part = root.find(f"{ns}part")
    if first_part is None:
        return

    first_part.attrib["id"] = score_part_id
    measures = list(first_part.findall(f"{ns}measure"))

    # Measure Setup Pass
    for m_idx, measure in enumerate(measures, 1):
        if m_idx > 1 and (m_idx - 1) % 4 == 0:
            for existing_p in measure.findall(f"{ns}print"):
                measure.remove(existing_p)
            print_elem = ET.Element(f"{ns}print")
            print_elem.attrib["new-page" if (m_idx - 1) % 16 == 0 else "new-system"] = "yes"
            measure.insert(0, print_elem)

        if m_idx == 1:
            attrs = measure.find(f"{ns}attributes") or ET.Element(f"{ns}attributes")
            if measure.find(f"{ns}attributes") is None:
                measure.insert(0, attrs)

            if attrs.find(f"{ns}time") is None:
                time_elem = ET.SubElement(attrs, f"{ns}time")
                beats_val, beat_type_val = ("4", "4")
                if time_sig_str and '/' in time_sig_str:
                    beats_val, beat_type_val = [p.strip() for p in time_sig_str.split('/')]
                ET.SubElement(time_elem, f"{ns}beats").text = beats_val
                ET.SubElement(time_elem, f"{ns}beat-type").text = beat_type_val

            set_or_create(attrs, "staves", "2")
            for old_elem in list(attrs.findall(f"{ns}clef")) + list(attrs.findall(f"{ns}staff-details")):
                attrs.remove(old_elem)

            clef1 = ET.SubElement(attrs, f"{ns}clef", attrib={"number": "1"})
            ET.SubElement(clef1, f"{ns}sign").text = "F"
            ET.SubElement(clef1, f"{ns}line").text = "4"

            clef2 = ET.SubElement(attrs, f"{ns}clef", attrib={"number": "2"})
            ET.SubElement(clef2, f"{ns}sign").text = "TAB"
            ET.SubElement(clef2, f"{ns}line").text = "5"

            staff_details = ET.SubElement(attrs, f"{ns}staff-details", attrib={"number": "2"})
            
            # Line 1 = Bottom line (lowest pitch), Line N = Top line (highest pitch)
            tunings = BASS_TUNINGS.get(tuning_type, BASS_TUNINGS["4_string_standard"])

            ET.SubElement(staff_details, f"{ns}staff-lines").text = str(len(tunings))
            for idx_t, (step, oct_val) in enumerate(tunings, 1):
                s_tuning = ET.SubElement(staff_details, f"{ns}staff-tuning", attrib={"line": str(idx_t)})
                ET.SubElement(s_tuning, f"{ns}tuning-step").text = step
                ET.SubElement(s_tuning, f"{ns}tuning-octave").text = str(oct_val)

            reorder_children(attrs, ATTRS_SCHEMA_ORDER, ns)

    # Note & Tab Processing Pass
    note_evt_idx = 0
    active_spanner_tag = None
    in_legato = False
    in_slide = False
    current_fifths = 0

    last_note_elem = None

    for part in root.findall(f"{ns}part"):
        part.attrib["id"] = score_part_id
        for measure in part.findall(f"{ns}measure"):
            attrs = measure.find(f"{ns}attributes")
            if attrs is not None:
                key_elem = attrs.find(f"{ns}key")
                if key_elem is not None:
                    fifths_elem = key_elem.find(f"{ns}fifths")
                    if fifths_elem is not None and fifths_elem.text:
                        current_fifths = int(fifths_elem.text)

            m_children = list(measure)
            staff1_duration = 0
            tab_elements = []
            current_evt = None

            for elem in m_children:
                if elem.tag == f"{ns}note":
                    is_rest = elem.find(f"{ns}rest") is not None
                    is_chord = elem.find(f"{ns}chord") is not None
                    
                    # Detect tied continuation notes (tied from previous beat/measure)
                    is_tie_stop = (
                        elem.find(f"{ns}tie[@type='stop']") is not None or
                        elem.find(f"{ns}notations/tied[@type='stop']") is not None
                    ) and (
                        elem.find(f"{ns}tie[@type='start']") is None and
                        elem.find(f"{ns}notations/tied[@type='start']") is None
                    )

                    set_or_create(elem, "staff", "1")
                    set_or_create(elem, "voice", "1")

                    if not is_rest:
                        pitch_elem = elem.find(f"{ns}pitch")
                        if pitch_elem is not None:
                            step_val = pitch_elem.findtext(f"{ns}step") or "D"
                            try:
                                oct_val = int(pitch_elem.findtext(f"{ns}octave") or "3")
                            except ValueError:
                                oct_val = 3

                            # Bass Clef Stem Rules: Middle line is D3. Below D3 -> UP, D3 & above -> DOWN.
                            if oct_val < 3 or (oct_val == 3 and step_val == "C"):
                                set_or_create(elem, "stem", "up")
                            else:
                                set_or_create(elem, "stem", "down")

                            alter_elem = pitch_elem.find(f"{ns}alter")
                            acc_map = KEY_ACCIDENTALS.get(current_fifths, {})
                            if alter_elem is None and step_val in acc_map:
                                alt_val, acc_type = acc_map[step_val]
                                set_or_create(pitch_elem, "alter", alt_val)
                                set_or_create(elem, "accidental", acc_type)
                            elif alter_elem is not None and alter_elem.text:
                                alt_val = alter_elem.text
                                acc_type = "sharp" if alt_val == "1" else ("flat" if alt_val == "-1" else "natural")
                                set_or_create(elem, "accidental", acc_type)

                        # Event binding: Advance index ONLY on primary new note events
                        if not is_chord and not is_tie_stop:
                            if snapped_layer and note_evt_idx < len(snapped_layer):
                                current_evt = snapped_layer[note_evt_idx]
                                note_evt_idx += 1
                            else:
                                current_evt = None

                        evt = current_evt

                        # Expressive & Tab Processing
                        if evt is not None:
                            notations = elem.find(f"{ns}notations") or ET.SubElement(elem, f"{ns}notations")
                            technical = notations.find(f"{ns}technical") or ET.SubElement(notations, f"{ns}technical")

                            string_elem = technical.find(f"{ns}string")
                            fret_elem = technical.find(f"{ns}fret")
                            string_val = string_elem.text if string_elem is not None else getattr(evt, 'string_idx', None)
                            try:
                                fret_val = int(fret_elem.text) if fret_elem is not None and fret_elem.text else getattr(evt, 'fret_val', None)
                            except ValueError:
                                fret_val = getattr(evt, 'fret_val', None)

                            # Techniques
                            if evt.tag == "slap" and technical.find(f"{ns}slap") is None:
                                ET.SubElement(technical, f"{ns}slap")
                            elif evt.tag == "pop" and technical.find(f"{ns}pop") is None:
                                ET.SubElement(technical, f"{ns}pop")

                            # Dashes Spanner State Machine (P.M. / Let Ring)
                            if evt.tag in ["palm_mute", "let_ring"]:
                                if active_spanner_tag != evt.tag:
                                    if active_spanner_tag is not None:
                                        ET.SubElement(notations, f"{ns}dashes", attrib={"type": "stop", "number": "1"})
                                    active_spanner_tag = evt.tag
                                    ET.SubElement(notations, f"{ns}dashes", attrib={"type": "start", "number": "1"})
                                    add_direction_words(measure, "P.M." if evt.tag == "palm_mute" else "let ring")
                            else:
                                if active_spanner_tag is not None:
                                    ET.SubElement(notations, f"{ns}dashes", attrib={"type": "stop", "number": "1"})
                                    active_spanner_tag = None

                            if evt.tag == "ghost":
                                set_or_create(elem, "notehead", "x")

                            # Legato Spanner Tracking
                            is_evt_legato = getattr(evt, 'is_legato', False)
                            if in_legato and not is_evt_legato:
                                ET.SubElement(technical, f"{ns}hammer-on", attrib={"type": "stop", "number": "1"})
                                ET.SubElement(notations, f"{ns}slur", attrib={"type": "stop", "number": "1"})
                                in_legato = False
                            elif is_evt_legato and not in_legato:
                                ET.SubElement(technical, f"{ns}hammer-on", attrib={"type": "start", "number": "1"}).text = "H"
                                ET.SubElement(notations, f"{ns}slur", attrib={"type": "start", "number": "1"})
                                in_legato = True

                            # Slide Spanner Tracking
                            is_evt_slide = getattr(evt, 'is_slide', False)
                            if in_slide and not is_evt_slide:
                                ET.SubElement(notations, f"{ns}slide", attrib={"type": "stop", "number": "1"})
                                in_slide = False
                            elif is_evt_slide and not in_slide:
                                ET.SubElement(notations, f"{ns}slide", attrib={"type": "start", "number": "1"}).text = "sl."
                                in_slide = True

                            # Bends
                            positive_bends = [b for b in (getattr(evt, 'bends', []) or []) if b > 0.05]
                            if fret_val is not None and fret_val > 0 and positive_bends:
                                bend_elem = ET.SubElement(technical, f"{ns}bend")
                                ET.SubElement(bend_elem, f"{ns}bend-alter").text = str(round(max(positive_bends), 1))
                            elif fret_val is not None and fret_val > 0 and getattr(evt, 'microtone_cents', 0) > 10.0:
                                bend_elem = ET.SubElement(technical, f"{ns}bend")
                                ET.SubElement(bend_elem, f"{ns}bend-alter").text = str(round(evt.microtone_cents / 100.0, 2))

                            # Clean string/fret from Staff 1
                            if string_elem is not None: technical.remove(string_elem)
                            if fret_elem is not None: technical.remove(fret_elem)
                            if len(technical) == 0: notations.remove(technical)
                            if len(notations) == 0: elem.remove(notations)

                            last_note_elem = elem

                        # Clone note for TAB Staff (Staff 2)
                        tab_note = ET.fromstring(ET.tostring(elem))
                        set_or_create(tab_note, "staff", "2")
                        set_or_create(tab_note, "voice", "1")

                        tab_notations = tab_note.find(f"{ns}notations") or ET.SubElement(tab_note, f"{ns}notations")
                        tab_tech = tab_notations.find(f"{ns}technical") or ET.SubElement(tab_notations, f"{ns}technical")

                        if tab_tech.find(f"{ns}string") is None and string_val is not None:
                            ET.SubElement(tab_tech, f"{ns}string").text = str(string_val)
                        if tab_tech.find(f"{ns}fret") is None and fret_val is not None:
                            ET.SubElement(tab_tech, f"{ns}fret").text = str(fret_val)

                        # Clean redundant non-TAB markings on Staff 2 note
                        tab_accidental = tab_note.find(f"{ns}accidental")
                        if tab_accidental is not None:
                            tab_note.remove(tab_accidental)

                        reorder_children(elem, NOTE_SCHEMA_ORDER, ns)
                        reorder_children(tab_note, NOTE_SCHEMA_ORDER, ns)

                        tab_elements.append(tab_note)

                    else:
                        # Processing Rest
                        set_or_create(elem, "staff", "1")
                        set_or_create(elem, "voice", "1")
                        reorder_children(elem, NOTE_SCHEMA_ORDER, ns)

                        tab_rest = ET.fromstring(ET.tostring(elem))
                        set_or_create(tab_rest, "staff", "2")
                        set_or_create(tab_rest, "voice", "1")
                        reorder_children(tab_rest, NOTE_SCHEMA_ORDER, ns)

                        tab_elements.append(tab_rest)

                    dur_text = elem.findtext(f"{ns}duration")
                    if not is_chord and dur_text:
                        staff1_duration += int(dur_text)

            # Append TAB Voice via Backup
            if staff1_duration > 0 and tab_elements:
                backup_elem = ET.Element(f"{ns}backup")
                ET.SubElement(backup_elem, f"{ns}duration").text = str(staff1_duration)
                measure.append(backup_elem)
                for tab_elem in tab_elements:
                    measure.append(tab_elem)

    # DANGLING SPANNER CLEANUP: Close any remaining open spanners on the last note
    if last_note_elem is not None:
        notations = last_note_elem.find(f"{ns}notations") or ET.SubElement(last_note_elem, f"{ns}notations")
        technical = notations.find(f"{ns}technical") or ET.SubElement(notations, f"{ns}technical")

        if active_spanner_tag is not None:
            ET.SubElement(notations, f"{ns}dashes", attrib={"type": "stop", "number": "1"})
            active_spanner_tag = None

        if in_legato:
            ET.SubElement(technical, f"{ns}hammer-on", attrib={"type": "stop", "number": "1"})
            ET.SubElement(notations, f"{ns}slur", attrib={"type": "stop", "number": "1"})
            in_legato = False

        if in_slide:
            ET.SubElement(notations, f"{ns}slide", attrib={"type": "stop", "number": "1"})
            in_slide = False

        reorder_children(last_note_elem, NOTE_SCHEMA_ORDER, ns)

    if ns:
        ET.register_namespace('', ns.strip('{}'))
    if hasattr(ET, 'indent'):
        ET.indent(tree, space="  ")

    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
