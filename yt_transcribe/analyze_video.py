import os
import argparse
import subprocess
from datetime import datetime

def process_youtube_url(url: str, base_dir: str = "analysis_pipeline") -> None:
    """Process a single YouTube URL through the entire pipeline"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_dir = os.path.join(base_dir, f"analysis_{timestamp}")
    os.makedirs(pipeline_dir, exist_ok=True)

    print(f"\n=== Starting analysis pipeline for: {url} ===\n")

    # Step 1: Transcribe the video
    print("Step 1: Transcribing video...")
    try:
        subprocess.run([
            "python",
            "whisper_transcribe_videos.py",
            "-u", url
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during transcription: {e}")
        return

    # Find the most recent transcription file
    transcriptions_dir = "transcriptions"
    if not os.path.exists(transcriptions_dir):
        print(f"Error: {transcriptions_dir} directory not found")
        return

    transcription_files = [
        f for f in os.listdir(transcriptions_dir)
        if f.endswith(".txt")
    ]

    if not transcription_files:
        print("Error: No transcription files found")
        return

    latest_transcription = max(
        transcription_files,
        key=lambda x: os.path.getctime(os.path.join(transcriptions_dir, x))
    )
    transcription_path = os.path.join(transcriptions_dir, latest_transcription)

    # Step 2: Run the analysis pipeline
    print("\nStep 2: Analyzing transcription...")
    try:
        subprocess.run([
            "python",
            "advanced_summarizer.py",
            "-f", transcription_path,
            "-o", pipeline_dir
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during analysis: {e}")
        return

    print(f"\n=== Analysis pipeline completed ===")
    print(f"Results saved in: {pipeline_dir}")

def main():
    parser = argparse.ArgumentParser(description='YouTube Video Transcription and Analysis Pipeline')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-u', '--url', help='YouTube video URL')
    input_group.add_argument('-f', '--file', help='File containing YouTube URLs (one per line)')
    parser.add_argument('-o', '--output-dir', default='analysis_pipeline',
                       help='Base output directory (default: analysis_pipeline)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.url:
        # Process single URL
        process_youtube_url(args.url, args.output_dir)
    else:
        # Process multiple URLs from file
        with open(args.file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]

        for i, url in enumerate(urls, 1):
            print(f"\nProcessing URL {i} of {len(urls)}")
            process_youtube_url(url, args.output_dir)

if __name__ == "__main__":
    main()
