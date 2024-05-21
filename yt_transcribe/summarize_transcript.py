import os
import argparse
import transformers
import torch

def split_text_into_chunks(text, chunk_size=1024):
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []

    for paragraph in paragraphs:
        if len(" ".join(current_chunk)) + len(paragraph) <= chunk_size:
            current_chunk.append(paragraph)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [paragraph]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def generate_summaries(pipeline, chunks):
    summaries = []
    for idx, chunk in enumerate(chunks, start=1):
        messages = [
            {"role": "system", "content": "You are a research assistant, finding the most useful insights for a book"},
            {"role": "user", "content": f"<|start_header_id|>user<|end_header_id|>Create a summary capturing the main points and key details with headings and subheadings based on what you can understand and why it is relevant:\n\n{chunk}<|eot_id|>"},
            {"role": "assistant", "content": "<|start_header_id|>assistant<|end_header_id|>"}
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        summary = outputs[0]["generated_text"][len(prompt):].strip()
        numbered_summary = f"Summary {idx}:\n{summary}"
        summaries.append(numbered_summary)

    return summaries

def main():
    parser = argparse.ArgumentParser(description='Generate a summary from a text file using Meta Llama 3.')
    parser.add_argument('input_file', help='Path to the input text file')
    args = parser.parse_args()

    with open(args.input_file, 'r') as file:
        input_text = file.read()

    model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=model_id,
        model_kwargs={
            "torch_dtype": torch.float16,
            "quantization_config": {"load_in_4bit": True},
            "low_cpu_mem_usage": True,
        },
        device_map="auto",
        max_length=3000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
    )

    chunks = split_text_into_chunks(input_text, chunk_size=1024)
    summaries = generate_summaries(pipeline, chunks)

    os.makedirs('summaries', exist_ok=True)

    input_file_name = os.path.splitext(os.path.basename(args.input_file))[0]
    output_file_name = f"{input_file_name}_summary.txt"
    output_file_path = os.path.join('summaries', output_file_name)

    with open(output_file_path, 'w') as file:
        file.write("\n\n".join(summaries))

    print(f"Summaries saved to: {output_file_path}")

if __name__ == "__main__":
    main()
