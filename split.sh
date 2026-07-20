#!/bin/bash
set -x
# split.sh
# Usage: ./split.sh path/to/file1.mp3 path/to/file2.wav path/to/file3.mp3

set -euo pipefail

# Check if at least one argument was passed
if [ $# -eq 0 ]; then
    echo "❌ Error: Missing input audio file target(s)."
    echo "Usage: $0 <path_to_audio_file1.mp3> [path_to_audio_file2.mp3 ...]"
    exit 1
fi

echo "=== 🎼 Deploying Stem Separation Engine ==="

if [ -d "/opt/homebrew/bin" ]; then export PATH="/opt/homebrew/bin:$PATH"; fi
if command -v python3.11 &> /dev/null; then PY_CMD="python3.11"; elif command -v python3 &> /dev/null; then PY_CMD="python3"; else echo "❌ Error: Python 3.11 or later required."; exit 1; fi

ENV_DIR=".venv_profound"
if [ ! -d "$ENV_DIR" ]; then $PY_CMD -m venv "$ENV_DIR"; fi
source "$ENV_DIR/bin/activate"

pip install --upgrade pip "setuptools<70.0.0"
pip install "demucs-mlx[convert]" librosa soundfile

cat << 'EOF' > run_split.py
import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import tempfile
from demucs_mlx.api import Separator, save_audio

def process_file(file_path_str, separator):
    audio_path = Path(file_path_str).resolve()
    
    if not audio_path.exists():
        print(f"❌ Error: File not found: {audio_path}")
        return

    out_dir = Path(f"./stems_{audio_path.stem}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n==================================================")
    print(f"🎵 Processing: {audio_path.name}")
    print(f"==================================================")
    print(f"[1/2] Loading audio file into memory...")
    
    # mono=False ensures we preserve standard stereo layout
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    
    if y.ndim == 1:
        y = y[np.newaxis, :]
        
    total_samples = y.shape[1]
    chunk_length_sec = 15
    chunk_samples = int(chunk_length_sec * sr)
    
    print(f"Loaded audio track: {y.shape[0]} channels at {sr}Hz.")
    print(f"[2/2] Processing source separation in sequential {chunk_length_sec}-second chunks...")
    
    accumulated_stems = {}
    
    # Slice along the time domain
    for start_idx in range(0, total_samples, chunk_samples):
        end_idx = min(start_idx + chunk_samples, total_samples)
        chunk_data = y[:, start_idx:end_idx]
        
        # Create a fast temporary WAV file for the MLX context to read
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            # Transpose data because soundfile expects (frames, channels)
            sf.write(tmp_path, chunk_data.T, sr)
            _, stems = separator.separate_audio_file(tmp_path)
            
            for name, stem_audio in stems.items():
                if name not in accumulated_stems:
                    accumulated_stems[name] = []
                # Explicitly cast to standard numpy array for seamless stitching
                accumulated_stems[name].append(np.asarray(stem_audio))
                
            print(f"  ✓ Processed chunk: {start_idx/sr:.1f}s to {end_idx/sr:.1f}s")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
                
    print("\n🎼 Stitching stems back together and exporting files...")
    for name, chunks in accumulated_stems.items():
        # Concatenate along the time axis (last dimension)
        full_stem = np.concatenate(chunks, axis=-1)
        stem_path = out_dir / f"{name}.wav"
        save_audio(full_stem, stem_path, samplerate=separator.samplerate)
        print(f"Saved complete stem: {stem_path}")

def main():
    # Instantiate the separator once so it doesn't reload for every file
    print("Initializing Demucs separator model...")
    separator = Separator(model="htdemucs_6s")
    
    # Process all file arguments passed from Bash
    for file_path in sys.argv[1:]:
        try:
            process_file(file_path, separator)
        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")

if __name__ == "__main__":
    main()
EOF

# Execute the chunking pipeline passing all provided arguments ($@)
python run_split.py "$@"
