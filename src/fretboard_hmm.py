class ErgonomicFretboardHMMSolver:
    def __init__(self, tuning_type='4_string_standard'):
        self.tuning_type = tuning_type
        if tuning_type == '5_string_low_b': self.strings = {1: 43, 2: 38, 3: 33, 4: 28, 5: 23}
        elif tuning_type == '4_string_drop_d': self.strings = {1: 43, 2: 38, 3: 33, 4: 26}
        else: self.strings = {1: 43, 2: 38, 3: 33, 4: 28}
        self.num_frets = 20

    def get_valid_positions(self, midi_pitch):
        # Force midi_pitch into bass range first (up to G4 / MIDI 67)
        while midi_pitch > 67: midi_pitch -= 12
        while midi_pitch < 23: midi_pitch += 12
        return [(s, midi_pitch - open_p) for s, open_p in self.strings.items() if 0 <= midi_pitch - open_p <= self.num_frets]

    def solve(self, note_events):
        if not note_events: return [], [], []
        sequence_states = [
            self.get_valid_positions(n[2]) or
            self.get_valid_positions(n[2]-12) or
            [(list(self.strings.keys())[0], 0)]
            for n in note_events
        ]
        T, V, path = len(sequence_states), [{}], {}

        for state in sequence_states[0]:
            string_num, fret = state
            tag = note_events[0][5]
            # Open string cost bonus (-2.0) to prefer open string positions over position jumps
            open_string_bonus = -2.0 if fret == 0 else 0.0
            box_cost = (0.0 if (0 <= fret <= 5) else (fret - 5) * 4.0) + open_string_bonus
            tech_cost = 10.0 if (tag == "pop" and string_num > 2) else 5.0 if (tag == "slap" and string_num < 3) else 0.0
            V[0][state], path[state] = -(box_cost + tech_cost), [state]

        for t in range(1, T):
            V.append({})
            new_path = {}
            tag = note_events[t][5]

            for c_state in sequence_states[t]:
                c_string, c_fret = c_state
                best_cost, best_prev = -float('inf'), None

                for p_state in sequence_states[t-1]:
                    if p_state not in V[t-1]: continue
                    p_string, p_fret = p_state
                    
                    # Increased fret jump penalties (3.5x multiplier) to eliminate >4 fret jumps
                    fret_shift = 0.2 if (c_fret == 0 or p_fret == 0) else abs(c_fret - p_fret) * 3.5
                    string_shift = abs(c_string - p_string) * 2.0
                    high_fret_penalty = (c_fret - 5) * 5.0 if c_fret > 5 else 0.0
                    
                    # Open string preference bonus
                    open_string_cost = -2.0 if c_fret == 0 else 0.0
                    
                    tech_cost = 15.0 if tag == "pop" and c_string > 2 else 8.0 if tag == "slap" and c_string < 3 else 0.0
                    total_score = V[t-1][p_state] - (fret_shift + string_shift + high_fret_penalty + tech_cost + open_string_cost)
                    
                    if total_score > best_cost:
                        best_cost, best_prev = total_score, p_state

                if best_prev is None and sequence_states[t-1]:
                    best_prev, best_cost = sequence_states[t-1][0], V[t-1].get(sequence_states[t-1][0], 0.0) - 10.0

                V[t][c_state], new_path[c_state] = best_cost, path.get(best_prev, [c_state]) + [c_state]
            path = new_path

        optimal_states = path.get(max(V[-1], key=V[-1].get), [sequence_states[-1][0]]) if V[-1] else [s[0] for s in sequence_states]
        rakes, legatos = [False] * T, [False] * T
        for i in range(1, T):
            dt = note_events[i][0] - note_events[i-1][1]
            if optimal_states[i][0] > optimal_states[i-1][0] and dt < 0.12: rakes[i] = True
            # Restrict legato tags to valid, non-zero frets on the same string
            p_fret = optimal_states[i-1][1]
            c_fret = optimal_states[i][1]
            if optimal_states[i][0] == optimal_states[i-1][0] and p_fret > 0 and c_fret > 0 and abs(c_fret - p_fret) in [1, 2, 3] and dt < 0.04:
                legatos[i] = True
        return optimal_states, rakes, legatos
