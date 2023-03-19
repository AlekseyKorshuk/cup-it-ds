from reward_model import GPTRewardModel
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json


def get_dataset():
    ds = load_dataset("AlekseyKorshuk/cup-it-ds")["test"].to_dict()
    return [dict(zip(ds, t)) for t in zip(*ds.values())]


def get_positions(scores):
    n = len(scores)
    sorted_positions = sorted(range(n), key=lambda k: scores[k], reverse=True)
    positions = [sorted_positions.index(i) for i in range(n)]
    return positions


ds = get_dataset()

tokenizer = AutoTokenizer.from_pretrained("gpt2", truncation_side="left")
tokenizer.pad_token = tokenizer.eos_token

PAD_ID = tokenizer("<|endoftext|>")["input_ids"][0]

MODEL_PATH = "gpt2"

model = GPTRewardModel(MODEL_PATH, tokenizer(tokenizer.pad_token)["input_ids"][0])

model.load_state_dict(torch.load("./rm_checkpoint/no-context/checkpoint-4956/pytorch_model.bin"),
                      strict=True)
model.half()  # Converts to fp16 for faster inference
model.eval()
model.cuda()
max_length = 512

for row in tqdm(ds):
    prompt = row["text"]
    scores = []
    for comment in row["comments"]:
        encodings_dict = tokenizer(
            "<|startoftext|>" + prompt + "\n\n" + comment["text"] + "<|endoftext|>",
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            output = model(**encodings_dict)
        reward = output[0][0]
        scores.append(reward)
    positions = get_positions(scores)
    for comment, position in zip(row["comments"], positions):
        comment["score"] = position

with open('no-context-output.jsonl', 'w') as outfile:
    for entry in ds:
        json.dump(entry, outfile)
        outfile.write('\n')
