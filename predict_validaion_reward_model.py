import tqdm
from transformers import pipeline
import torch
from datasets import load_dataset, Dataset
from reward_model import GPTRewardModel
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import json

tokenizer = AutoTokenizer.from_pretrained("gpt2", truncation_side="left")
tokenizer.pad_token = tokenizer.eos_token

PAD_ID = tokenizer("<|endoftext|>")["input_ids"][0]

MODEL_PATH = "gpt2"

model = GPTRewardModel(MODEL_PATH, tokenizer(tokenizer.pad_token)["input_ids"][0])

model.load_state_dict(torch.load("./pytorch_model.bin"),
                      strict=True)
model.half()  # Converts to fp16 for faster inference
model.eval()
model.cuda()
max_length = 512

dataset = load_dataset("ummagumm-a/cup_it_ds_split_with_lang_with_topic", split="validation")

updated_comments = []
for row in tqdm.tqdm(dataset):

    texts = [
        "<|startoftext|>" + row["text"].strip() + "\n\n" + comment["text"].strip() + "<|endoftext|>"
        for comment in row["comments"]
    ]
    predictions = []
    for text in texts:
        encodings_dict = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            output = model(**encodings_dict)
        reward = output[0][0]
        predictions.append(float(reward))
    comments = row["comments"]
    for comment, result in zip(comments, predictions):
        comment["prediction"] = result
    updated_comments.append(comments)

dataset = dataset.remove_columns("comments").add_column("comments", updated_comments)
dataset.push_to_hub("AlekseyKorshuk/reward-model-no-topic-predictions")
