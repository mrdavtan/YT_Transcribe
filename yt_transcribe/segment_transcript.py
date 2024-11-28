import os
import argparse
import transformers
import torch
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
import gc

@dataclass
class TopicSegment:
    start_pos: int
    end_pos: int
    start_time: Optional[str]
    end_time: Optional[str]
    topic: str
    subtopics: List[str]
    key_points: List[str]
    importance: str
    references: List[Dict[str, str]]  # Track cross-references to other segments

class TopicSegmenter:
    def __init__(self, model_id: str = "unsloth/llama-3-8b-Instruct-bnb-4bit",
                 window_size: int = 4000, overlap: int = 800):
        self.window_size = window_size
        self.overlap = overlap
        self.pipeline = self._initialize_pipeline(model_id)
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _initialize_pipeline(self, model_id: str):
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
            max_new_tokens=512  # Reduced to help with memory
        )

    def _clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def _extract_json_safely(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response with fallback parsing"""
        try:
            # Try to find JSON block
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx + 1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse JSON response")
            return None

    def _generate_segment_analysis(self, text: str, context: Optional[Dict] = None) -> Optional[Dict]:
        prompt = f"""Analyze this transcript segment and identify the main topic and key points.
Return only a JSON object with this structure:
{{
    "topic": "Clear topic title",
    "start_time": "HH:MM:SS if present, else null",
    "end_time": "HH:MM:SS if present, else null",
    "key_points": ["Main point 1", "Main point 2"],
    "subtopics": ["Subtopic 1", "Subtopic 2"],
    "importance": "High/Medium/Low",
    "references_previous": "Brief note if references earlier content",
    "continues_next": boolean
}}

Transcript text:
{text}

Previous topic (if any): {context.get('prev_topic') if context else 'None'}
"""
        try:
            response = self.pipeline(prompt)[0]['generated_text']
            return self._extract_json_safely(response)
        except Exception as e:
            self.logger.error(f"Generation error: {str(e)}")
            return None

    def segment_transcript(self, transcript: str) -> List[TopicSegment]:
        segments = []
        current_pos = 0
        prev_context = None

        while current_pos < len(transcript):
            window_end = min(current_pos + self.window_size, len(transcript))
            window_text = transcript[current_pos:window_end]

            # Analyze current window
            result = self._generate_segment_analysis(
                window_text,
                context=prev_context
            )

            if result:
                segment = TopicSegment(
                    start_pos=current_pos,
                    end_pos=window_end,
                    start_time=result.get('start_time'),
                    end_time=result.get('end_time'),
                    topic=result['topic'],
                    subtopics=result.get('subtopics', []),
                    key_points=result.get('key_points', []),
                    importance=result.get('importance', 'Medium'),
                    references=[]
                )
                segments.append(segment)

                prev_context = {
                    'prev_topic': result['topic'],
                    'references_previous': result.get('references_previous'),
                    'continues_next': result.get('continues_next', False)
                }

            # Advance window with overlap
            current_pos = window_end - self.overlap
            self._clear_memory()

        return self._post_process_segments(segments)

    def _post_process_segments(self, segments: List[TopicSegment]) -> List[TopicSegment]:
        """Connect related segments and identify cross-references"""
        for i, segment in enumerate(segments):
            # Look for topic continuations
            if i > 0 and segments[i-1].topic == segment.topic:
                segment.start_pos = segments[i-1].start_pos
                segments[i-1] = segment

            # Track cross-references
            for point in segment.key_points:
                for other_segment in segments[:i]:
                    if any(p.lower() in point.lower() for p in other_segment.key_points):
                        segment.references.append({
                            'topic': other_segment.topic,
                            'point': point
                        })

        return [s for i, s in enumerate(segments)
                if i == 0 or s.topic != segments[i-1].topic]

    def save_analysis(self, segments: List[TopicSegment], output_dir: Path):
        """Save segmentation results with cross-reference tracking"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed analysis
        analysis_path = output_dir / "topic_analysis.json"
        with analysis_path.open('w') as f:
            json.dump([{
                'topic': s.topic,
                'time_range': f"{s.start_time or 'start'} - {s.end_time or 'end'}",
                'key_points': s.key_points,
                'subtopics': s.subtopics,
                'importance': s.importance,
                'references': s.references
            } for s in segments], f, indent=2)

        # Save topic overview
        overview_path = output_dir / "topic_overview.txt"
        with overview_path.open('w') as f:
            for segment in segments:
                f.write(f"\n{segment.topic}\n{'='*len(segment.topic)}\n")
                f.write(f"Time: {segment.start_time or 'start'} - {segment.end_time or 'end'}\n")
                f.write(f"Importance: {segment.importance}\n\n")

                if segment.references:
                    f.write("References to earlier topics:\n")
                    for ref in segment.references:
                        f.write(f"- {ref['topic']}: {ref['point']}\n")
                f.write("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, help='Transcript file')
    parser.add_argument('-o', '--output-dir', default='segment_analysis')
    args = parser.parse_args()

    segmenter = TopicSegmenter()
    with open(args.file) as f:
        transcript = f.read()

    segments = segmenter.segment_transcript(transcript)
    segmenter.save_analysis(segments, Path(args.output_dir))

if __name__ == "__main__":
    main()
