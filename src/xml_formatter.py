import fractions
import xml.etree.ElementTree as ET
from music21 import duration

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


from utils import (
    idiomatic_rhythm_snap,
    build_m21_duration,
    decompose_duration_engraver_rules,
    consolidate_measure_notation,
)


def ensure_note_type(note_elem, current_divisions, ns=""):
    """Ensures that a MusicXML note or rest element has an explicit <type> child."""
    type_elem = note_elem.find(f"{ns}type")
    if type_elem is None or not type_elem.text:
        dur_elem = note_elem.find(f"{ns}duration")
        if dur_elem is not None and dur_elem.text and current_divisions > 0:
            try:
                dur_val = int(dur_elem.text)
                ql = dur_val / float(current_divisions)
                type_str = None
                if ql >= 3.5: type_str = "whole"
                elif ql >= 1.75: type_str = "half"
                elif ql >= 0.875: type_str = "quarter"
                elif ql >= 0.4375: type_str = "eighth"
                elif ql >= 0.21875: type_str = "16th"
                elif ql >= 0.109375: type_str = "32nd"
                elif ql >= 0.0546875: type_str = "64th"
                else: type_str = "128th"

                if type_str:
                    set_or_create(note_elem, "type", type_str)
            except ValueError:
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
        set_or_create(midi_inst, "midi-program", "33")

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
            tunings = BASS_TUNINGS.get(tuning_type, BASS_TUNINGS["4_string_standard"])

            ET.SubElement(staff_details, f"{ns}staff-lines").text = str(len(tunings))
            for idx_t, (step, oct_val) in enumerate(tunings, 1):
                s_tuning = ET.SubElement(staff_details, f"{ns}staff-tuning", attrib={"line": str(idx_t)})
                ET.SubElement(s_tuning, f"{ns}tuning-step").text = step
                ET.SubElement(s_tuning, f"{ns}tuning-octave").text = str(oct_val)

            reorder_children(attrs, ATTRS_SCHEMA_ORDER, ns)

    note_evt_idx = 0
    current_divisions = 1

    for part in root.findall(f"{ns}part"):
        part.attrib["id"] = score_part_id
        for measure in part.findall(f"{ns}measure"):
            attrs = measure.find(f"{ns}attributes")
            if attrs is not None:
                divs_elem = attrs.find(f"{ns}divisions")
                if divs_elem is not None and divs_elem.text:
                    try:
                        current_divisions = int(divs_elem.text)
                    except ValueError:
                        pass

            notes_in_m = list(measure.findall(f"{ns}note"))
            for note in notes_in_m:
                ensure_note_type(note, current_divisions, ns)
                if note.find(f"{ns}pitch") is not None:
                    set_or_create(note, "staff", "1")
                    if snapped_layer and note_evt_idx < len(snapped_layer):
                        note_evt = snapped_layer[note_evt_idx]
                        if note.find(f"{ns}chord") is None:
                            if note_evt.category and note_evt.category != "melodic":
                                add_direction_words(measure, f"[{note_evt.category.upper()}]")
                            if note_evt.anchor_pattern and note_evt.is_anchor:
                                add_direction_words(measure, f"Anchor: {note_evt.anchor_pattern}")
                            note_evt_idx += 1

                reorder_children(note, NOTE_SCHEMA_ORDER, ns)

    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
