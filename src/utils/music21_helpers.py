import fractions
import os
import sys
from music21 import duration

# Inject parent directory of utils to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def idiomatic_rhythm_snap(dur_q, level=5, is_compound=False):
    """
    Snaps a quarterLength duration to idiomatic engraver grid, enforcing
    a minimum duration threshold (1/16th or 1/32nd) to prevent micro-durations.
    """
    if isinstance(dur_q, (float, int)):
        dur_q = round(float(dur_q), 4)

    dur_q = fractions.Fraction(dur_q).limit_denominator(12 if is_compound else 32)

    MIN_DURATION = fractions.Fraction(1, 16) if level <= 3 else fractions.Fraction(1, 32)
    if dur_q < MIN_DURATION:
        return MIN_DURATION

    return dur_q


def build_m21_duration(dur_q):
    """
    Constructs a music21 duration.Duration object explicitly using type names,
    preventing MusicXMLExportException during MusicXML serialization.
    """
    dur_frac = fractions.Fraction(dur_q).limit_denominator(32)

    MAP = {
        fractions.Fraction(4, 1): ("whole", 0, None),
        fractions.Fraction(3, 1): ("half", 1, None),
        fractions.Fraction(2, 1): ("half", 0, None),
        fractions.Fraction(3, 2): ("quarter", 1, None),
        fractions.Fraction(1, 1): ("quarter", 0, None),
        fractions.Fraction(3, 4): ("eighth", 1, None),
        fractions.Fraction(2, 3): ("quarter", 0, duration.Tuplet(3, 2)),
        fractions.Fraction(1, 2): ("eighth", 0, None),
        fractions.Fraction(3, 8): ("16th", 1, None),
        fractions.Fraction(1, 3): ("eighth", 0, duration.Tuplet(3, 2)),
        fractions.Fraction(1, 4): ("16th", 0, None),
        fractions.Fraction(3, 16): ("32nd", 1, None),
        fractions.Fraction(1, 6): ("16th", 0, duration.Tuplet(3, 2)),
        fractions.Fraction(1, 8): ("32nd", 0, None),
        fractions.Fraction(1, 12): ("32nd", 0, duration.Tuplet(3, 2)),
        fractions.Fraction(1, 16): ("64th", 0, None),
        fractions.Fraction(1, 32): ("128th", 0, None),
    }

    if dur_frac in MAP:
        type_str, dots, tup = MAP[dur_frac]
        d = duration.Duration(type=type_str)
        d.dots = dots
        if tup is not None:
            d.tuplets = (tup,)
        return d

    closest_frac = min(MAP.keys(), key=lambda f: abs(f - dur_frac))
    type_str, dots, tup = MAP[closest_frac]
    d = duration.Duration(type=type_str)
    d.dots = dots
    if tup is not None:
        d.tuplets = (tup,)
    return d


def decompose_duration_engraver_rules(dur_q, curr_m_fill, measure_capacity, is_compound=False):
    """
    Decomposes durations strictly into expressible MusicXML standard & triplet values 
    to prevent inexpressible duration exceptions in music21.
    """
    dur_q = fractions.Fraction(dur_q).limit_denominator(32)
    curr_m_fill = fractions.Fraction(curr_m_fill).limit_denominator(32)
    measure_capacity = fractions.Fraction(measure_capacity).limit_denominator(32)

    allowed_values = [
        fractions.Fraction(4, 1),
        fractions.Fraction(3, 1),
        fractions.Fraction(2, 1),
        fractions.Fraction(3, 2),
        fractions.Fraction(1, 1),
        fractions.Fraction(3, 4),
        fractions.Fraction(2, 3),
        fractions.Fraction(1, 2),
        fractions.Fraction(3, 8),
        fractions.Fraction(1, 3),
        fractions.Fraction(1, 4),
        fractions.Fraction(3, 16),
        fractions.Fraction(1, 6),
        fractions.Fraction(1, 8),
        fractions.Fraction(1, 12),
        fractions.Fraction(1, 16),
        fractions.Fraction(1, 32),
    ]

    chunks = []
    while dur_q > 0:
        rem_in_m = measure_capacity - curr_m_fill
        if rem_in_m <= 0:
            curr_m_fill = fractions.Fraction(0, 1)
            rem_in_m = measure_capacity

        best_val = None
        for val in allowed_values:
            if val <= dur_q and val <= rem_in_m:
                best_val = val
                break

        if best_val is None:
            if dur_q >= fractions.Fraction(1, 64):
                best_val = min(fractions.Fraction(1, 32), rem_in_m)
                dur_q = fractions.Fraction(0, 1)
            else:
                break

        chunks.append(best_val)
        dur_q -= best_val
        curr_m_fill += best_val
        if curr_m_fill >= measure_capacity:
            curr_m_fill = fractions.Fraction(0, 1)

    return [c for c in chunks if c > 0]


def consolidate_measure_notation(curr_measure, measure_capacity, is_compound=False):
    """
    Consolidates measure notation, verifying all contained notes and rests have valid
    MusicXML duration types for downstream exporters and rendering engines.
    """
    if curr_measure is None:
        return

    for elem in list(curr_measure.notesAndRests):
        if not hasattr(elem, 'duration') or elem.duration is None:
            elem.duration = build_m21_duration(1.0)
        elif not elem.duration.type or elem.duration.type == 'inexpressible':
            elem.duration = build_m21_duration(elem.duration.quarterLength)
