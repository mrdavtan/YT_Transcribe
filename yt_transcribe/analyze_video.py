import os
import subprocess
from datetime import datetime
import argparse
import torch
import logging
from pathlib import Path
import json
from typing import Optional, Dict
import gc
import yt_dlp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """Clear GPU memory between major processing steps"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

class AnalysisState:
    """Track the state of analysis for each video"""
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.state_file = self.base_dir / "analysis_state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load existing state or create new one"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}

    def save_state(self):
        """Save current state"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)

    def update_video_state(self, video_id: str, status: Dict):
        """Update state for a specific video"""
        current_state = self.state.get(video_id, {})
        current_state.update(status)
        self.state[video_id] = current_state
        self.save_state()

    def get_video_state(self, video_id: str) -> Optional[Dict]:
        """Get state for a specific video"""
        return self.state.get(video_id)

class VideoDownloadProgressHook:
    """Progress hook for video downloads"""
    def __init__(self):
        self._downloaded_bytes = 0
        self._total_bytes = 0
        self._start_time = datetime.now()

    def __call__(self, d):
        if d['status'] == 'downloading':
            self._downloaded_bytes = d.get('downloaded_bytes', 0)
            self._total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate', 0)

            # Calculate progress
            if self._total_bytes:
                progress = (self._downloaded_bytes / self._total_bytes) * 100
                speed = d.get('speed', 0)
                if speed:
                    eta = d.get('eta', 0)
                    logger.info(
                        f"Download progress: {progress:.1f}% "
                        f"({self._downloaded_bytes}/{self._total_bytes} bytes) "
                        f"Speed: {speed/1024/1024:.1f}MB/s ETA: {eta}s"
                    )

        elif d['status'] == 'finished':
            duration = datetime.now() - self._start_time
            logger.info(f"Download completed in {duration.total_seconds():.1f}s")

def check_video_exists(video_id: str) -> Optional[str]:
    """Check if video is already downloaded"""
    videos_dir = Path("videos")
    if videos_dir.exists():
        for video_file in videos_dir.glob(f"*{video_id}*.mp4"):
            return str(video_file)
    return None

def download_video(url: str, video_id: str) -> str:
    """Download video if not already present"""
    existing_video = check_video_exists(video_id)
    if existing_video:
        logger.info(f"Video already downloaded: {existing_video}")
        return existing_video

    logger.info("Starting video download...")
    videos_dir = Path("videos")
    videos_dir.mkdir(exist_ok=True)

    ydl_opts = {
        'format': 'best',
        'outtmpl': str(videos_dir / f'%(title)s_{video_id}_%(id)s.%(ext)s'),
        'progress_hooks': [VideoDownloadProgressHook()],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info)

    return video_path

def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL"""
    if 'watch?v=' in url:
        return url.split('watch?v=')[1].split('&')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[1].split('?')[0]
    return None

def process_youtube_url(url: str, base_dir: str = "analysis_pipeline", reprocess: bool = False) -> None:
    """Process a single YouTube URL through the entire pipeline"""
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from URL: {url}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_dir = Path(base_dir) / f"analysis_{video_id}_{timestamp}"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    # Initialize state tracking
    state_manager = AnalysisState(base_dir)
    video_state = state_manager.get_video_state(video_id)

    logger.info(f"\n=== Starting analysis pipeline for: {url} ===\n")

    try:
        # Step 1: Download video if needed
        video_path = check_video_exists(video_id)
        if video_path:
            logger.info(f"Video already downloaded: {video_path}")
        else:
            video_path = download_video(url, video_id)
            logger.info(f"Video downloaded to: {video_path}")

        # Step 2: Check for existing transcription
        if not reprocess and video_state and video_state.get("transcription_complete"):
            logger.info("Using existing transcription...")
            transcription_path = video_state["transcription_path"]
            if not Path(transcription_path).exists():
                logger.warning("Existing transcription not found, will retranscribe...")
                transcription_path = None
        else:
            transcription_path = None

        # Step 3: Transcribe if needed
        if not transcription_path:
            logger.info("Starting transcription...")
            clear_gpu_memory()

            result = subprocess.run([
                "python",
                "whisper_transcribe_videos.py",
                "-u", url
            ], check=True, capture_output=True, text=True)

            # Find the transcription file
            transcriptions_dir = Path("transcriptions")
            if not transcriptions_dir.exists():
                raise FileNotFoundError(f"Error: {transcriptions_dir} directory not found")

            transcription_files = list(transcriptions_dir.glob("*.txt"))
            if not transcription_files:
                raise FileNotFoundError("Error: No transcription files found")

            transcription_path = str(max(transcription_files, key=lambda x: x.stat().st_mtime))

            # Update state
            state_manager.update_video_state(video_id, {
                "transcription_complete": True,
                "transcription_path": transcription_path,
                "transcription_timestamp": timestamp
            })

        # Step 4: Run Phase 1 - Detailed Summary
        logger.info("\nStep 4: Running Phase 1 - Detailed Summary...")
        clear_gpu_memory()

        phase1_result = subprocess.run([
            "python",
            "advanced_summarizer.py",
            "-f", transcription_path,
            "-o", str(pipeline_dir),
            "--phase", "1"
        ], check=True, capture_output=True, text=True)

        # Update state after Phase 1
        state_manager.update_video_state(video_id, {
            "phase1_complete": True,
            "phase1_timestamp": timestamp,
            "analysis_dir": str(pipeline_dir)
        })

        # Step 5: Run Phase 2 - Analysis
        logger.info("\nStep 5: Running Phase 2 - Analysis...")
        clear_gpu_memory()

        phase2_result = subprocess.run([
            "python",
            "advanced_summarizer.py",
            "--phase", "2",
            "--analysis-dir", str(pipeline_dir)
        ], check=True, capture_output=True, text=True)

        # Update state with analysis completion
        state_manager.update_video_state(video_id, {
            "phase2_complete": True,
            "phase2_timestamp": timestamp,
            "analysis_complete": True
        })

        logger.info(f"\n=== Analysis pipeline completed ===")
        logger.info(f"Results saved in: {pipeline_dir}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error during processing: {e}")
        logger.error(f"Process output: {e.output if hasattr(e, 'output') else 'No output available'}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        clear_gpu_memory()

def main():
    parser = argparse.ArgumentParser(description='YouTube Video Transcription and Analysis Pipeline')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-u', '--url', help='YouTube video URL')
    input_group.add_argument('-f', '--file', help='File containing YouTube URLs (one per line)')

    parser.add_argument('-o', '--output-dir', default='analysis_pipeline',
                       help='Base output directory (default: analysis_pipeline)')
    parser.add_argument('--reprocess', action='store_true',
                       help='Reprocess video even if previous analysis exists')
    parser.add_argument('--retry-failed', action='store_true',
                       help='Retry previously failed analyses')
    parser.add_argument('--start-phase', type=int, choices=[1, 2],
                       help='Start processing from specific phase')

    args = parser.parse_args()

    try:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        if args.url:
            # Process single URL
            process_youtube_url(args.url, args.output_dir, args.reprocess)
        else:
            # Process multiple URLs from file
            with open(args.file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]

            state_manager = AnalysisState(args.output_dir)

            for i, url in enumerate(urls, 1):
                logger.info(f"\nProcessing URL {i} of {len(urls)}")
                video_id = extract_video_id(url)

                if video_id:
                    video_state = state_manager.get_video_state(video_id)

                    # Skip if already processed and not reprocessing
                    if not args.reprocess and video_state and video_state.get("analysis_complete"):
                        logger.info(f"Skipping {url} - already processed")
                        continue

                    # Skip if failed and not retrying
                    if not args.retry_failed and video_state and video_state.get("failed"):
                        logger.info(f"Skipping {url} - previously failed")
                        continue

                try:
                    process_youtube_url(url, args.output_dir, args.reprocess)
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {e}")
                    if video_id:
                        state_manager.update_video_state(video_id, {
                            "failed": True,
                            "error": str(e),
                            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                        })
                    continue
                finally:
                    clear_gpu_memory()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
