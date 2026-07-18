import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import tempfile
from demucs_mlx.api import Separator, save_audio

def main():
    audio_path = Path(sys.argv[1]).resolve()
    out_dir = Path(f"./stems_{audio_path.stem}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[1/2] Loading audio file into memory...")
    # mono=False ensures we preserve standard stereo layout
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    
    if y.ndim == 1:
        y = y[np.newaxis, :]
        
    total_samples = y.shape[1]
    chunk_length_sec = 15
    chunk_samples = int(chunk_length_sec * sr)
    
    print(f"Loaded audio track: {y.shape[0]} channels at {sr}Hz.")
    print(f"[2/2] Processing source separation in sequential {chunk_length_sec}-second chunks...")
    
    separator = Separator(model="htdemucs_6s")
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

if __name__ == "__main__":
    main()
