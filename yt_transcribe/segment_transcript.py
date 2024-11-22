import os
import argparse
import transformers
import torch
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import gc
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('segmentation.log')
    ]
)
logger = logging.getLogger(__name__)

class TopicSegmenter:
    def __init__(self, model_id="unsloth/llama-3-8b-Instruct-bnb-4bit"):
        self.pipeline = self._initialize_pipeline(model_id)

    def _initialize_pipeline(self, model_id):
        return transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "use_cache": True,
            },
            device_map="auto",
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            max_new_tokens=768,
        )

    def _clear_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def _generate_response(self, prompt: str, desc: str = None) -> str:
        """Generate response with proper error handling"""
        try:
            if desc:
                logger.info(f"Generating: {desc}")

            messages = [
                {"role": "system", "content": "You are a research assistant, analyzing conversation transcripts and identifying topic changes and themes."},
                {"role": "user", "content": f"<|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|>"},
                {"role": "assistant", "content": "<|start_header_id|>assistant<|end_header_id|>"}
            ]

            formatted_prompt = self.pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=768,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )

            response = outputs[0]["generated_text"][len(formatted_prompt):].strip()

            del outputs
            self._clear_memory()

            return response
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            self._clear_memory()
            raise

    def identify_topic_segments(self, text: str, window_size: int = 5000) -> List[Dict]:
        """Identify major topic segments in the text using a sliding window approach"""
        segments = []
        current_pos = 0
        overlap = 1000  # Overlap between windows to maintain context

        logger.info("Identifying topic segments...")

        while current_pos < len(text):
            # Get current window of text with overlap
            window_end = min(current_pos + window_size, len(text))
            window_text = text[current_pos:window_end]

            # Create prompt for topic identification
            prompt = f"""Analyze this segment of a conversation transcript and identify the major topic or topics being discussed.
Focus on clear topic transitions and thematic changes.

For each identified topic segment, provide:
1. The timestamp range where it appears (if timestamps are present)
2. A clear, specific topic title
3. Key speakers or main points discussed
4. Importance level (High/Medium/Low) based on content significance

Format as JSON with these fields:
- start_time: timestamp or "continuing"
- end_time: timestamp or "continues"
- topic: clear title
- subtopics: list of specific points
- key_speakers: list of speakers
- importance: "High"/"Medium"/"Low"
- key_points: list of main points
- continues_previous: true/false
- continues_next: true/false

Transcript segment:
{window_text}"""

            # Generate and parse response
            response = self._generate_response(prompt, f"Processing window {current_pos}-{window_end}")

            try:
                # Extract JSON from response (might need cleaning/parsing)
                segments_data = json.loads(response)
                segments.extend(segments_data if isinstance(segments_data, list) else [segments_data])
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from response at position {current_pos}. Skipping.")

            # Move window forward with overlap
            current_pos = window_end - overlap
            if current_pos < 0:
                break

        return self._consolidate_segments(segments)

    def _consolidate_segments(self, segments: List[Dict]) -> List[Dict]:
        """Consolidate overlapping segments and clean up transitions"""
        consolidated = []
        current_segment = None

        for segment in segments:
            if not current_segment:
                current_segment = segment
                continue

            # Check if segments should be merged
            if (segment.get('continues_previous') and
                current_segment.get('continues_next')):
                # Merge segments
                current_segment['end_time'] = segment['end_time']
                current_segment['subtopics'].extend(segment['subtopics'])
                current_segment['key_points'].extend(segment['key_points'])
            else:
                consolidated.append(current_segment)
                current_segment = segment

        if current_segment:
            consolidated.append(current_segment)

        return consolidated

def process_transcript(file_path: str, output_dir: str) -> None:
    """Process a transcript file and generate topic segments"""
    logger.info(f"Processing transcript: {file_path}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_subdir = os.path.join(output_dir, f"{base_name}_segments_{timestamp}")
    os.makedirs(output_subdir, exist_ok=True)

    try:
        # Read transcript
        with open(file_path, 'r') as f:
            transcript = f.read()

        # Initialize segmenter
        segmenter = TopicSegmenter()

        # Identify topics
        segments = segmenter.identify_topic_segments(transcript)

        # Save topic metadata
        metadata_path = os.path.join(output_subdir, "topic_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "original_file": file_path,
                "timestamp": timestamp,
                "segments": segments
            }, f, indent=2)

        # Create individual segment files
        for i, segment in enumerate(segments, 1):
            segment_file = os.path.join(output_subdir, f"segment_{i:03d}.txt")
            with open(segment_file, 'w') as f:
                f.write(f"Topic: {segment['topic']}\n")
                f.write(f"Timestamps: {segment['start_time']} - {segment['end_time']}\n")
                f.write(f"Importance: {segment['importance']}\n")
                f.write("\nKey Points:\n")
                for point in segment['key_points']:
                    f.write(f"- {point}\n")
                f.write("\nSubtopics:\n")
                for subtopic in segment['subtopics']:
                    f.write(f"- {subtopic}\n")

        # Create topic index
        index_path = os.path.join(output_subdir, "topic_index.txt")
        with open(index_path, 'w') as f:
            f.write("Topic Index\n")
            f.write("=" * 50 + "\n\n")
            for i, segment in enumerate(segments, 1):
                f.write(f"{i:03d}. {segment['topic']}\n")
                f.write(f"    Time: {segment['start_time']} - {segment['end_time']}\n")
                f.write(f"    Importance: {segment['importance']}\n\n")

        logger.info(f"Segmentation complete. Output saved to: {output_subdir}")

    except Exception as e:
        logger.error(f"Error processing transcript: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Segment transcript into topic-based chunks')
    parser.add_argument('-f', '--file', help='Path to transcript file', required=True)
    parser.add_argument('-o', '--output-dir', default='segmented_transcripts',
                       help='Output directory (default: segmented_transcripts)')

    args = parser.parse_args()

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        process_transcript(args.file, args.output_dir)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
