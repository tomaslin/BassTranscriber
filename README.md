# Bass Transcription Engine

This is an experiment to see how far AI and audio processing can be pushed to transcribe bass lines from music files.

While polished tools exist for piano—like [Google’s Magenta](https://magenta.tensorflow.org/)—there aren’t many free options that handle bass translations well. 

This pipeline is heavily aided by AI, so if you don’t like AI tools, this won’t be for you.

My local machine is a M1 mac so I’m optimizing for this. I tried in Google AI studio but it simply didn’t really work. 

Slowly trying to improve this script but not in a rush. 

---

## The Process

The execution pipeline moves through the following steps:

* **Audio Splitting:** [Meta’s Demucs](https://github.com/facebookresearch/demucs) (via [demucs-mlx](https://github.com/awni/demucs-mlx)) splits the original track into isolated stems, providing a dedicated `bass.wav` and `drums.wav`. This is all encapsulated in the split.sh script because this step is the most straightforward but tedious one. 


* **Genre Profiling:** [Librosa](https://librosa.org/) calculates the track’s tempo, spectral centroid, and note density. It compares these metrics against a built-in dictionary to determine the closest musical genre.


* **Targeted Filtering:** A [SciPy](https://scipy.org/) butterworth bandpass filter is applied to the bass track. The low-cut and high-cut frequencies change dynamically based on the detected genre to isolate the bass frequencies and reduce bleed.

* **Pitch Prediction:** Spotify’s [Basic Pitch](https://github.com/spotify/basic-pitch) processes the filtered audio to extract raw MIDI notes and pitch bend data.

* **Fretboard Mapping:** A Viterbi algorithm maps the raw MIDI notes onto a simulated bass fretboard layout. It calculates finger movement costs to pick the most efficient, physically playable string and fret positions. This is so that your fingers don’t hurt. 


* **Notation Generation:** [Music21](https://web.mit.edu/music21/) handles the final transcription structural assembly. It quantizes the timing, adds chord symbols, and maps performance techniques like slaps, pops, ghost notes, slides, and vibrato based on the audio’s volume and frequency changes.


---

## Why This Approach is Used

* **Context-Based Settings:** Pitch tracking accuracy drops when treating a sub-bass electronic line the same as a bright funk line. Shifting the filtering and detection thresholds based on the genre profile helps clean up the raw data input.


* **Tab Generation Physics:** Raw MIDI transcription often outputs unplayable layouts for stringed instruments. Running the notes through a fretboard solver forces the script to prioritize fingerings a human can execute.


* **Multi-Tier Output:** The script automatically generates three separate [MusicXML](https://www.musicxml.com/) files (`simple`, `normal`, and `complex`) so you can choose between basic root notes or the full performance detail. This is really for learning. I am a dumb newbie player and I want fairly unsophisticated root notes. But I also want to hear how the complex stuff sounds. 
