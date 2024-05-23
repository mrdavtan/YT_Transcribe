import os
import argparse
import transformers
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]

def split_text_into_chunks(text, chunk_size=2048):
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

def generate_summaries(pipeline, dataloader):
    summaries = []
    for idx, batch in enumerate(dataloader, start=1):
        messages = [
            {"role": "system", "content": "You are a research assistant, finding the most useful insights for a book"},
            {"role": "user", "content": f"<|start_header_id|>user<|end_header_id|>Create a concise summary of the following text in the form of a list of bullet points with headings, subheadings, and sub-subheadings. Aim for less than 20 main bullet points, each with no more than 5 sub-points. Focus on capturing the key ideas and main points for better readability and understanding:\n\n{batch[0]}<|eot_id|>"},
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
        model_kwargs={
            "torch_dtype": torch.float16,
            "quantization_config": {"load_in_4bit": True},
            "low_cpu_mem_usage": True,
        },
    )

    chunks = split_text_into_chunks(input_text)
    dataset = TextDataset(chunks)
    dataloader = DataLoader(dataset, batch_size=1)
    summaries = generate_summaries(pipeline, dataloader)

    os.makedirs('shortbulletpoints', exist_ok=True)

    input_file_name = os.path.splitext(os.path.basename(args.input_file))[0]
    output_file_name = input_file_name.replace('_bulletpoints', '_shortbulletpoints') + '.txt'
    output_file_path = os.path.join('shortbulletpoints', output_file_name)

    with open(output_file_path, 'w') as file:
        file.write("\n\n".join(summaries))

    print(f"Short bullet points saved to: {output_file_path}")

if __name__ == "__main__":
    main()
