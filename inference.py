from reward_model import GPTRewardModel
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import json

from train_reward_model import PairwiseDataset


def pairwise_data_collator(data):
    if len(data[0]) == 4:
        return {'input_ids': torch.cat([f[0] for f in data] + [f[2] for f in data]),
                'attention_mask': torch.cat([f[1] for f in data] + [f[3] for f in data])}
    elif len(data[0]) == 2:
        return {'input_ids': torch.cat([f[0] for f in data]),
                'attention_mask': torch.cat([f[1] for f in data])}
    else:
        raise ValueError("Invalid data format")


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
        # print(output)
        reward = output[0][0]
        scores.append(reward)
    positions = get_positions(scores)
    for comment, position in zip(row["comments"], positions):
        comment["score"] = position

with open('no-context-output.jsonl', 'w') as outfile:
    for entry in ds:
        json.dump(entry, outfile)
        outfile.write('\n')


# data = load_dataset("AlekseyKorshuk/synthetic-instruct-gptj-pairwise")
# eval_dataset = PairwiseDataset(data["test"], tokenizer, max_length=max_length)
# batch_size = 1
# dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, collate_fn=pairwise_data_collator)


def gen(inputs):
    input_ids = inputs["input_ids"].to("cuda")
    # print(torch.cuda.memory_summary())
    with torch.no_grad():
        output = model(input_ids)
    # Correct reward score is last element - unless there are issues with padding?
    rewards = output[:, -1]
    return rewards


# cnt = 0
# chosen_rewards = []
# rejected_rewards = []
# for i, batch in tqdm(enumerate(dataloader)):
#     if i > 1000:
#         break
#     rewards = gen(batch)
#     chosen_rewards += rewards[:rewards.shape[0] // 2].tolist()
#     rejected_rewards += rewards[rewards.shape[0] // 2:].tolist()
#     # accuracy.append(int(chosen_rewards[0] > rejected_rewards[0]))
#
# accuracy = [int(chosen > rejected) for chosen, rejected in zip(chosen_rewards, rejected_rewards)]
# print(pd.DataFrame(accuracy).describe())

import pdb;

pdb.set_trace()

# data = {'input_ids': batch["input_ids"][0].unsqueeze(0), 'attention_mask': batch["attention_mask"][0].unsqueeze(0)}
