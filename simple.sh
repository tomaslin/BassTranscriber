#!/bin/bash
# transcribe_bass_simple.sh
# Usage: ./transcribe_bass_simple.sh path/to/stems_folder

set -euo pipefail

STEMS_DIR="${1:-}"
if [ -z "$STEMS_DIR" ] || [ ! -d "$STEMS_DIR" ]; then
    echo "Error: Missing or invalid stems directory."
    echo "Usage: $0 <path_to_stems_folder>"
    exit 1
fi

if [ -d "/opt/homebrew/bin" ]; then export PATH="/opt/homebrew/bin:$PATH"; fi
if command -v python3.11 &> /dev/null; then PY_CMD="python3.11"; elif command -v python3 &> /dev/null; then PY_CMD="python3"; else echo "Error: Python 3.11+ required."; exit 1; fi

OUT_DIR="./output_bass"
ENV_DIR=".venv_bass"
mkdir -p "$OUT_DIR"

if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PY_CMD -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

echo "Installing compatible foundations..."
"$ENV_DIR/bin/python" -m pip install --upgrade pip wheel
"$ENV_DIR/bin/python" -m pip install "setuptools<82"

echo "Verifying basic dependencies..."
"$ENV_DIR/bin/python" -m pip install --no-cache-dir torch torchaudio librosa music21 basic-pitch soundfile pretty_midi resampy==0.4.2 onnxruntime

cleanup() { rm -f run_engine_simple.py; }
trap cleanup EXIT

# ==============================================================================
# SIMPLE BASS TRANSCRIPTION ENGINE
# ==============================================================================
cat << 'EOF' > run_engine_simple.py
import sys
from pathlib import Path
import torch
import warnings
import librosa
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Simple Engine Status: Online | Processing platform: {device.type.upper()}")

from basic_pitch.inference import predict
from music21 import midi, instrument, clef, metadata, tempo, stream

def main():
    stems_dir = Path(sys.argv[1]).resolve()
    out_dir = Path("./output_bass").resolve()
    
    bass_wav = stems_dir / "bass.wav"
    project_name = stems_dir.name.replace('stems_', '')
    
    if not bass_wav.exists():
        print(f"Error: Could not find bass.wav in {stems_dir}")
        sys.exit(1)
        
    print("\n[1/3] Loading and conditioning audio track...")
    audio_data, sr = librosa.load(str(bass_wav), sr=22050, mono=True)
    audio_data = np.squeeze(audio_data)
    
    conditioned_wav = out_dir / "conditioned_bass_temp.wav"
    sf.write(str(conditioned_wav), audio_data, sr)
        
    print("Running raw basic_pitch inference...")
    _, bass_midi, _ = predict(str(conditioned_wav))
    
    print("[2/3] Processing MIDI tracks to sheet music via framework defaults...")
    tmp_mid = out_dir / "simple_temp.mid"
    bass_midi.write(str(tmp_mid))
    
    # CRITICAL FIX: Load using the framework's native translation engine
    # midiFileToObject handles the raw stream segmentation safely
    mf = midi.MidiFile()
    mf.open(str(tmp_mid))
    mf.read()
    mf.close()
    
    # Translate directly to a structured Score object (automatically handles Measures)
    sc = midi.translate.midiFileToStream(mf)
    
    # Extract notes to verify existence
    all_notes = sc.flatten().notes
    if len(all_notes) == 0:
        print("Error: The model could not map any musical notes from this audio file.")
        if tmp_mid.exists(): tmp_mid.unlink()
        if conditioned_wav.exists(): conditioned_wav.unlink()
        sys.exit(1)
        
    # Standard framework cleanup: Enforce Bass Clef and Instrument properties on the main parts
    for part in sc.getElementsByClass(stream.Part):
        # Insert structural elements at the very start of the first measure
        first_measure = part.getElementsByClass('Measure')
        if first_measure:
            first_measure[0].insert(0, instrument.ElectricBass())
            first_measure[0].insert(0, clef.BassClef())
        else:
            part.insert(0, instrument.ElectricBass())
            part.insert(0, clef.BassClef())
            
        # Quantize and layout using framework rules inside the measure contexts
        part.quantize(quarterLengthDivisors=(2, 4, 8), inPlace=True)
        part.makeBeams(inPlace=True)
        part.makeRests(fillGaps=True, inPlace=True)
        part.makeTies(inPlace=True)

    # Attach proper score metadata wrapper
    sc.metadata = metadata.Metadata(title=f"{project_name.title()} - Simple Transcription")
    sc.insert(0, tempo.MetronomeMark(number=120))
    
    print("[3/3] Exporting simple arrangement...")
    out_file = out_dir / f"{project_name}_simple.musicxml"
    sc.write('musicxml', fp=str(out_file))
    
    if tmp_mid.exists(): tmp_mid.unlink()
    if conditioned_wav.exists(): conditioned_wav.unlink()
    print(f"\nSuccess! Simple MusicXML saved to: {out_file.name}\n")

if __name__ == "__main__":
    main()
EOF

"$ENV_DIR/bin/python" run_engine_simple.py "$STEMS_DIR"
