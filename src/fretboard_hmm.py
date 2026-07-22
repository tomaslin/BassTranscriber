import math
from note_event import NoteEvent
from pitch_theory import fold_pitch_to_bass_range

TUNING_PROFILES = {
    '4_string_standard': {1: 43, 2: 38, 3: 33, 4: 28},        # G2, D2, A1, E1
    '5_string_standard': {1: 43, 2: 38, 3: 33, 4: 28, 5: 23},  # G2, D2, A1, E1, B0
    '6_string_standard': {1: 48, 2: 43, 3: 38, 4: 33, 5: 28, 6: 23}, # C3, G2, D2, A1, E1, B0
}


class ErgonomicFretboardHMMSolver:
    def __init__(self, tuning_type='4_string_standard', beam_width=8):
        self.tuning_type = tuning_type if tuning_type in TUNING_PROFILES else '4_string_standard'
        self.beam_width = beam_width
        self.strings = TUNING_PROFILES[self.tuning_type]
        self.num_frets = 24

    def get_valid_positions(self, midi_pitch: int):
        min_p = min(self.strings.values())
        max_p = max(self.strings.values()) + self.num_frets
        midi_pitch = fold_pitch_to_bass_range(midi_pitch, min_pitch=min_p, max_pitch=max_p)

        positions = []
        for s, open_p in self.strings.items():
            fret = midi_pitch - open_p
            if 0 <= fret <= self.num_frets:
                if fret == 0:
                    positions.append((s, 0, 0))
                else:
                    fingers = [1, 2, 4] if fret <= 5 else [1, 2, 3, 4]
                    for f in fingers:
                        positions.append((s, fret, f))
        return positions

    def _get_local_anchor_fret(self, note_events: list[NoteEvent], t: int, window=8) -> float:
        start = max(0, t - window)
        end = min(len(note_events), t + window + 1)
        local_pitches = [n.pitch for n in note_events[start:end]]

        median_pitch = sorted(local_pitches)[len(local_pitches) // 2]
        open_pitches = sorted(self.strings.values())
        median_open = open_pitches[len(open_pitches) // 2]

        return float(max(1, min(self.num_frets, median_pitch - median_open)))

    def solve(self, raw_note_events: list[NoteEvent], bpm=120.0):
        if not raw_note_events:
            return [], [], [], []

        note_events = sorted(raw_note_events, key=lambda x: x.start)
        T = len(note_events)
        sec_per_beat = 60.0 / bpm if bpm > 0 else 0.5

        sequence_states = [
            self.get_valid_positions(n.pitch) or
            self.get_valid_positions(n.pitch - 12) or
            [(list(self.strings.keys())[0], 0, 0)]
            for n in note_events
        ]

        V = [{} for _ in range(T)]
        backpointer = [{} for _ in range(T)]

        initial_anchor = self._get_local_anchor_fret(note_events, 0)
        for state in sequence_states[0]:
            string_num, fret, finger = state
            tag = note_events[0].tag
            note_dur = note_events[0].duration

            open_cost = (-2.0 if note_dur > 0.3 else 1.5) if fret == 0 else 0.0
            box_cost = (fret * 0.08 if fret <= 7 else fret * 0.20) + open_cost

            tech_cost = 0.0
            if tag == "pop":
                tech_cost = 0.0 if string_num in [1, 2] else 25.0
            elif tag == "slap":
                tech_cost = 0.0 if string_num >= 3 else 18.0

            anchor_dist = abs(fret - initial_anchor) if fret > 0 else 0.0
            anchor_cost = anchor_dist * 0.15

            V[0][state] = box_cost + tech_cost + anchor_cost
            backpointer[0][state] = None

        if len(V[0]) > self.beam_width:
            V[0] = dict(sorted(V[0].items(), key=lambda x: x[1])[:self.beam_width])

        for t in range(1, T):
            prev_onset, prev_offset = note_events[t-1].start, note_events[t-1].end
            curr_onset, curr_offset = note_events[t].start, note_events[t].end

            onset_dt_sec = max(0.01, curr_onset - prev_onset)
            onset_dt_beats = max(0.125, onset_dt_sec / sec_per_beat)

            curr_dur = note_events[t].duration
            overlap_dur = max(0.0, prev_offset - curr_onset)
            tag = note_events[t].tag

            local_anchor = self._get_local_anchor_fret(note_events, t)

            for c_state in sequence_states[t]:
                c_string, c_fret, c_finger = c_state
                best_cost, best_prev = float('inf'), None

                for p_state in V[t-1]:
                    p_string, p_fret, p_finger = p_state

                    if overlap_dur > 0.08 and c_string == p_string and c_fret != p_fret:
                        overlap_penalty = 150.0
                    else:
                        overlap_penalty = overlap_dur * 20.0

                    fret_span = abs(c_fret - p_fret) if (p_fret > 0 and c_fret > 0) else 0
                    if fret_span > 4:
                        inertia_penalty = 80.0 + (25.0 * (fret_span - 4))
                    else:
                        inertia_penalty = fret_span * 1.2

                    if onset_dt_beats < 0.5 and fret_span >= 4:
                        inertia_penalty += 120.0

                    if p_fret == 0 or c_fret == 0:
                        fret_dist = 0.2
                        stretch_penalty = 0.0
                    else:
                        d_prev = 1.0 - math.pow(2, -p_fret / 12.0)
                        d_curr = 1.0 - math.pow(2, -c_fret / 12.0)
                        fret_dist = abs(d_curr - d_prev) * 25.0

                        if min(p_fret, c_fret) <= 5 and fret_span > 3:
                            stretch_penalty = 35.0
                        elif fret_span > 4:
                            stretch_penalty = 20.0
                        else:
                            stretch_penalty = 0.0

                    p_anchor = p_fret - (p_finger - 1) if p_finger > 0 else p_fret
                    c_anchor = c_fret - (c_finger - 1) if c_finger > 0 else c_fret
                    anchor_shift = abs(c_anchor - p_anchor)

                    if anchor_shift == 0:
                        finger_diff = c_finger - p_finger
                        fret_diff = c_fret - p_fret
                        strain = 8.0 if (fret_diff > 0 and finger_diff < 0) or (fret_diff < 0 and finger_diff > 0) else 0.0
                        transition_step_cost = (fret_dist * 0.3) + strain
                    else:
                        transition_step_cost = (anchor_shift * 3.0) / (onset_dt_beats + 0.1)

                    string_diff = c_string - p_string
                    if string_diff == 0:
                        string_shift = 0.0
                    elif string_diff > 0:
                        string_shift = math.pow(string_diff, 1.3) * 1.8 + (80.0 if fret_span >= 4 else 0.0)
                    else:
                        string_shift = math.pow(abs(string_diff), 1.4) * 2.5

                    open_cost = (-3.0 if (onset_dt_beats > 0.5 or curr_dur > 0.4) else 2.0) if c_fret == 0 else 0.0

                    tech_cost = 0.0
                    if tag == "pop":
                        tech_cost = 0.0 if c_string in [1, 2] else 25.0
                    elif tag == "slap":
                        tech_cost = 0.0 if c_string >= 3 else 18.0

                    anchor_dist = abs(c_fret - local_anchor) if c_fret > 0 else 0.0
                    anchor_cost = anchor_dist * 0.15

                    local_cost = (
                        transition_step_cost + stretch_penalty + inertia_penalty +
                        string_shift + open_cost + tech_cost + anchor_cost + overlap_penalty
                    )
                    total_score = V[t-1][p_state] + local_cost + (0.1 * math.pow(local_cost, 2))

                    if total_score < best_cost:
                        best_cost, best_prev = total_score, p_state

                if best_prev is not None:
                    V[t][c_state] = best_cost
                    backpointer[t][c_state] = best_prev

            if not V[t]:
                fallback_prev = min(V[t-1], key=V[t-1].get) if V[t-1] else sequence_states[t-1][0]
                for c_state in sequence_states[t]:
                    V[t][c_state] = V[t-1].get(fallback_prev, 0.0) + 100.0
                    backpointer[t][c_state] = fallback_prev

            if len(V[t]) > self.beam_width:
                V[t] = dict(sorted(V[t].items(), key=lambda x: x[1])[:self.beam_width])

        optimal_states_full = [None] * T
        best_last_state = min(V[-1], key=V[-1].get) if V[-1] else sequence_states[-1][0]
        optimal_states_full[-1] = best_last_state

        for t in range(T - 1, 0, -1):
            curr_state = optimal_states_full[t]
            optimal_states_full[t-1] = backpointer[t].get(curr_state, sequence_states[t-1][0])

        optimal_positions = optimal_states_full

        rakes = [False] * T
        legatos = [False] * T
        slides = [False] * T
        for i in range(1, T):
            onset_dt = note_events[i].start - note_events[i-1].start
            p_string, p_fret = optimal_states_full[i-1][0], optimal_states_full[i-1][1]
            c_string, c_fret = optimal_states_full[i][0], optimal_states_full[i][1]

            if (c_string - p_string) == 1 and onset_dt < 0.12:
                rakes[i] = True

            if c_string == p_string and p_fret > 0 and c_fret > 0 and p_fret != c_fret:
                fret_diff = abs(c_fret - p_fret)
                if fret_diff in [1, 2, 3] and onset_dt < 0.08:
                    legatos[i] = True
                elif fret_diff >= 3 and onset_dt < 0.18:
                    slides[i] = True

        return optimal_positions, rakes, legatos, slides
