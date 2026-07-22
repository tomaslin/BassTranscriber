import os
import argparse
from pipeline import AudioTranscriptionPipeline


def main():
    parser = argparse.ArgumentParser(description="Bass Transcription Engine")
    parser.add_argument('folders', nargs='+', help="Path to stem folder(s)")
    parser.add_argument('-a', '--all-levels', action='store_true', help="Generate outputs for all levels")
    parser.add_argument('-o', '--output-dir', help="Custom output directory")
    parser.add_argument('--level', type=int, default=5, help="Complexity level (0-5) - Defaults to 5")
    parser.add_argument('-g', '--gpu', action='store_true', help="Use GPU stack")

    args = parser.parse_args()

    pipeline = AudioTranscriptionPipeline(output_dir=args.output_dir)

    for folder in args.folders:
        if os.path.isdir(folder):
            pipeline.run(folder, generate_all_levels=args.all_levels, level=args.level, use_gpu=args.gpu)
        else:
            print(f"Directory not found: {folder}")


if __name__ == "__main__":
    main()
