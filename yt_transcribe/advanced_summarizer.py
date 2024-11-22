import os
import argparse
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import gc
from typing import List, Dict, Any
from tqdm import tqdm
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('summarizer.log')
    ]
)
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class TextProcessor:
    def __init__(self, model_id="unsloth/llama-3-8b-Instruct-bnb-4bit", chunk_size=2048):
        self.pipeline = self._initialize_pipeline(model_id)
        self.chunk_size = chunk_size

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
                {"role": "system", "content": "You are a research assistant, creating insightful and concise analysis"},
                {"role": "user", "content": f"<|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|>"},
                {"role": "assistant", "content": "<|start_header_id|>assistant<|end_header_id|>"}
            ]

            formatted_prompt = self.pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            terminators = [
                self.pipeline.tokenizer.eos_token_id,
                self.pipeline.tokenizer.convert_tokens_to_ids("<|end_of_text|>")
            ]

            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=768,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
            )

            response = outputs[0]["generated_text"][len(formatted_prompt):].strip()

            del outputs
            self._clear_memory()

            return response

        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            self._clear_memory()
            raise

    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into smaller chunks for processing"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def generate_detailed_summary(self, text: str) -> str:
        """First pass: Generate focused, concise summary with timestamps"""
        prompt = f"""Create a focused summary of the following text. Include only the most important points and essential quotes with their timestamps. Prioritize:
1. Core arguments and main ideas
2. Key statistics and facts
3. Critical quotes (only the most impactful ones)
4. Major theme transitions
5. Significant conclusions

Be concise and avoid redundancy. Skip tangential discussions and repeated points.

{text}"""
        return self._generate_response(prompt)

def process_phase1(file_path: str, processor: TextProcessor, output_dir: str) -> str:
    """Phase 1: Generate and save detailed summary"""
    logger.info(f"\nPhase 1 - Detailed Summary: {file_path}")

    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_subdir = os.path.join(output_dir, f"{base_name}_{timestamp}")
    os.makedirs(output_subdir, exist_ok=True)

    try:
        # Read the input file
        with open(file_path, 'r') as f:
            text = f.read()

        # Process chunks and generate detailed summary
        logger.info("Generating focused summary...")
        chunks = processor.split_text_into_chunks(text)
        detailed_summaries = []

        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            summary = processor.generate_detailed_summary(chunk)
            detailed_summaries.append(summary)
            processor._clear_memory()

        # Combine summaries and save
        detailed_summary = "\n\n".join(detailed_summaries)
        summary_path = os.path.join(output_subdir, "detailed_summary.txt")

        with open(summary_path, 'w') as f:
            f.write(detailed_summary)

        # Save phase completion marker
        with open(os.path.join(output_subdir, "phase1_complete"), 'w') as f:
            f.write(datetime.now().isoformat())

        logger.info(f"Phase 1 complete. Summary saved to: {summary_path}")
        return output_subdir

    except Exception as e:
        logger.error(f"Error in Phase 1: {str(e)}")
        raise

def process_phase2(output_dir: str, processor: TextProcessor) -> None:
    """Phase 2: Generate bullet points and analysis from saved summary"""
    logger.info("\nPhase 2 - Analysis Generation")

    # Verify Phase 1 completion
    if not os.path.exists(os.path.join(output_dir, "phase1_complete")):
        raise ValidationError("Phase 1 must be completed before running Phase 2")

    try:
        # Load the detailed summary
        summary_path = os.path.join(output_dir, "detailed_summary.txt")
        with open(summary_path, 'r') as f:
            detailed_summary = f.read()

        # Clear memory before starting Phase 2
        processor._clear_memory()

        # Generate bullet points
        logger.info("Generating bullet points...")
        prompt = f"""Create a comprehensive, hierarchical bullet-point summary focusing on:
- Main themes and key insights
- Supporting points and evidence
- Unique or particularly valuable ideas
- Practical applications or implications

{detailed_summary}"""

        bullet_points = processor._generate_response(prompt, "Generating bullet points")
        processor._clear_memory()

        # Save bullet points
        with open(os.path.join(output_dir, "bullet_points.txt"), 'w') as f:
            f.write(bullet_points)

        # Generate key insights
        logger.info("Generating key insights...")
        insights_prompt = f"""Analyze these bullet points to identify the 3-5 most valuable insights:

{bullet_points}"""

        key_insights = processor._generate_response(insights_prompt, "Generating key insights")
        processor._clear_memory()

        # Save key insights
        with open(os.path.join(output_dir, "key_insights.txt"), 'w') as f:
            f.write(key_insights)

        # Generate executive summary
        logger.info("Generating executive summary...")
        exec_prompt = f"""Create a compelling 500-word executive summary based on these key insights:

{key_insights}"""

        exec_summary = processor._generate_response(exec_prompt, "Generating executive summary")
        processor._clear_memory()

        # Save executive summary
        with open(os.path.join(output_dir, "executive_summary.txt"), 'w') as f:
            f.write(exec_summary)

        # Generate final analysis
        logger.info("Generating final analysis...")
        analysis_prompt = f"""Create a comprehensive analysis combining these elements:

Key Insights:
{key_insights}

Bullet Points:
{bullet_points}"""

        final_analysis = processor._generate_response(analysis_prompt, "Generating final analysis")

        # Save final analysis
        with open(os.path.join(output_dir, "final_analysis.txt"), 'w') as f:
            f.write(final_analysis)

        # Create index file
        index_content = f"""Analysis Index
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

1. executive_summary.txt - 500-word overview
2. key_insights.txt - Core insights
3. bullet_points.txt - Structured summary
4. detailed_summary.txt - Full summary with timestamps
5. final_analysis.txt - Complete analysis"""

        with open(os.path.join(output_dir, "index.txt"), 'w') as f:
            f.write(index_content)

        # Save phase completion marker
        with open(os.path.join(output_dir, "phase2_complete"), 'w') as f:
            f.write(datetime.now().isoformat())

        logger.info(f"Phase 2 complete. All analyses saved in: {output_dir}")

    except Exception as e:
        logger.error(f"Error in Phase 2: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Phase-based text analysis pipeline')
    parser.add_argument('-f', '--file', help='Path to input text file', required=True)
    parser.add_argument('-o', '--output-dir', default='analysis_output',
                       help='Output directory for analysis (default: analysis_output)')
    parser.add_argument('--phase', type=int, choices=[1, 2],
                       help='Specific phase to run (1=detailed summary, 2=analysis)')
    parser.add_argument('--analysis-dir',
                       help='Directory containing phase 1 output (required for phase 2)')
    parser.add_argument('--chunk-size', type=int, default=2048,
                       help='Size of text chunks for processing (default: 2048)')

    args = parser.parse_args()

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        processor = TextProcessor(chunk_size=args.chunk_size)

        if not args.phase or args.phase == 1:
            output_dir = process_phase1(args.file, processor, args.output_dir)
            if not args.phase:  # If no specific phase specified, continue to phase 2
                process_phase2(output_dir, processor)
        elif args.phase == 2:
            if not args.analysis_dir:
                raise ValueError("--analysis-dir is required for phase 2")
            process_phase2(args.analysis_dir, processor)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        processor._clear_memory()

if __name__ == "__main__":
    main()
