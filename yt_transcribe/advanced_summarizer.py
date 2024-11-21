import os
import argparse
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from typing import List, Dict, Any
import json

class TextProcessor:
    def __init__(self, model_id="unsloth/llama-3-8b-Instruct-bnb-4bit"):
        self.pipeline = self._initialize_pipeline(model_id)

    def _initialize_pipeline(self, model_id):
        return transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "use_cache": False,
            },
            device_map="auto",
            do_sample=True,
            top_p=0.9,
            temperature=0.6,
            max_new_tokens=512,  # Increased for more detailed analysis
        )

    def _generate_response(self, prompt: str) -> str:
        """Generate response using the pipeline with proper templating"""
        messages = [
            {"role": "system", "content": "You are a research assistant, creating insightful analysis and summaries"},
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
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,  # Slightly higher for more creative analysis
            top_p=0.9,
        )

        return outputs[0]["generated_text"][len(formatted_prompt):].strip()

    def generate_detailed_summary(self, text: str) -> str:
        """First pass: Generate detailed summary with timestamps and key points"""
        prompt = f"""Create a detailed summary of the following text. Include key points, important quotes, and maintain any timestamp references. Focus on preserving the most valuable insights and concrete information:

{text}"""
        return self._generate_response(prompt)

    def generate_bullet_points(self, text: str) -> str:
        """Second pass: Generate structured bullet points from the detailed summary"""
        prompt = f"""Create a hierarchical bullet-point summary of the following text. Focus on:
- Main themes and key insights
- Supporting points and evidence
- Unique or particularly valuable ideas
- Practical applications or implications

Format with clear headers and subpoints:

{text}"""
        return self._generate_response(prompt)

    def generate_analytical_report(self, original_text: str, detailed_summary: str, bullet_points: str) -> Dict[str, str]:
        """Third pass: Generate comprehensive analytical report"""
        # Extract key insights
        prompt_insights = f"""Analyze these summaries to identify the 3-5 most valuable and unique insights. Consider:
1. What makes these insights particularly valuable?
2. How do they connect to broader themes?
3. What practical implications do they have?

Text:
{detailed_summary}

Bullet Points:
{bullet_points}"""
        key_insights = self._generate_response(prompt_insights)

        # Generate executive summary
        prompt_exec_summary = f"""Create a compelling 500-word executive summary that:
1. Introduces the main topic and its importance
2. Highlights the key insights and their significance
3. Provides a clear conclusion and implications

Base this on:
{key_insights}"""
        executive_summary = self._generate_response(prompt_exec_summary)

        # Generate full analysis
        prompt_analysis = f"""Create a comprehensive analysis that weaves together the most important points and insights. Include:
1. Introduction
2. Key themes and patterns
3. Detailed discussion of main insights
4. Supporting evidence and examples
5. Implications and applications
6. Conclusion

Use the following sources:
Original Text: {original_text}
Detailed Summary: {detailed_summary}
Bullet Points: {bullet_points}
Key Insights: {key_insights}"""
        full_analysis = self._generate_response(prompt_analysis)

        return {
            "key_insights": key_insights,
            "executive_summary": executive_summary,
            "full_analysis": full_analysis
        }

def split_text_into_chunks(text: str, chunk_size: int = 1024) -> List[str]:
    """Split text into manageable chunks while preserving paragraph structure"""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current_length + paragraph_length > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(paragraph)
        current_length += paragraph_length

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks

def process_file(file_path: str, processor: TextProcessor, output_dir: str) -> None:
    """Process a single file through the multi-pass analysis pipeline"""
    print(f"\nProcessing: {file_path}")

    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_subdir = os.path.join(output_dir, f"{base_name}_{timestamp}")
    os.makedirs(output_subdir, exist_ok=True)

    # Read and process the input file
    with open(file_path, 'r') as f:
        text = f.read()

    # Step 1: Generate detailed summary
    print("Generating detailed summary...")
    chunks = split_text_into_chunks(text)
    detailed_summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{len(chunks)}")
        summary = processor.generate_detailed_summary(chunk)
        detailed_summaries.append(summary)
    detailed_summary = "\n\n".join(detailed_summaries)

    # Step 2: Generate bullet points from the detailed summary
    print("Generating bullet points...")
    bullet_points = processor.generate_bullet_points(detailed_summary)

    # Step 3: Generate final analytical report
    print("Generating analytical report...")
    analysis = processor.generate_analytical_report(text, detailed_summary, bullet_points)

    # Save all outputs
    outputs = {
        "detailed_summary.txt": detailed_summary,
        "bullet_points.txt": bullet_points,
        "key_insights.txt": analysis["key_insights"],
        "executive_summary.txt": analysis["executive_summary"],
        "full_analysis.txt": analysis["full_analysis"]
    }

    # Create index file
    index_content = f"""Analysis Index
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

1. executive_summary.txt - 500-word overview of key findings
2. key_insights.txt - Core insights and their significance
3. bullet_points.txt - Hierarchical summary of main points
4. detailed_summary.txt - Comprehensive summary with timestamps
5. full_analysis.txt - Complete analytical report

Original file: {file_path}"""

    outputs["index.txt"] = index_content

    # Save all files
    for filename, content in outputs.items():
        output_path = os.path.join(output_subdir, filename)
        with open(output_path, 'w') as f:
            f.write(content)

    print(f"Analysis complete. Output saved to: {output_subdir}")

def main():
    parser = argparse.ArgumentParser(description='Advanced multi-pass text analysis pipeline')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-f', '--file', help='Path to a single input text file')
    input_group.add_argument('-d', '--directory', help='Path to directory containing multiple text files')
    parser.add_argument('-o', '--output-dir', default='analysis_output',
                       help='Output directory for analysis (default: analysis_output)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    processor = TextProcessor()

    if args.file:
        process_file(args.file, processor, args.output_dir)
    else:
        for filename in os.listdir(args.directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(args.directory, filename)
                process_file(file_path, processor, args.output_dir)

if __name__ == "__main__":
    main()
