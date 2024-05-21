import os
import argparse
import transformers
import torch

def split_text_into_chunks(text, chunk_size=1024):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk)) + len(word) <= chunk_size:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def generate_summary(pipeline, chunk):
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

    return summary

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
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "use_cache": False,
        },
        device_map="auto",
        do_sample=True,
        top_p=0.9,
        temperature=0.6,
        max_new_tokens=256,
    )

    chunks = split_text_into_chunks(input_text, chunk_size=1024)

    os.makedirs('summaries', exist_ok=True)

    input_file_name = os.path.splitext(os.path.basename(args.input_file))[0]
    output_file_name = f"{input_file_name}_summary.txt"
    output_file_path = os.path.join('summaries', output_file_name)

    with open(output_file_path, 'w') as output_file:
        for idx, chunk in enumerate(chunks, start=1):
            summary = generate_summary(pipeline, chunk)
            numbered_summary = f"Summary {idx}:\n{summary}\n\n"
            output_file.write(numbered_summary)
            print(f"Generated summary for chunk {idx}")

    print(f"Summaries saved to: {output_file_path}")

if __name__ == "__main__":
    main()
