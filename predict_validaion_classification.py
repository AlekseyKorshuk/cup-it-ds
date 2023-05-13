import tqdm
from transformers import pipeline
import torch
from datasets import load_dataset, Dataset

model_name = "AlekseyKorshuk/roberta"

device = 0 if torch.cuda.is_available() else 'cpu'
pipe = pipeline("text-classification", model_name, device=device)

top_k = len(pipe.model.config.id2label)

dataset = load_dataset("ummagumm-a/cup_it_ds_split_with_lang_with_topic", split="validation")

updated_comments = []
for row in tqdm.tqdm(dataset):
    texts = [row["text"].strip() + pipe.tokenizer.sep_token + comment["text"].strip() for comment in row["comments"]]
    results = pipe(texts, top_k=top_k)
    comments = row["comments"]
    for comment, result in zip(comments, results):
        comment["prediction"] = result
    updated_comments.append(comments)

dataset = dataset.remove_columns("comments").add_column("comments", updated_comments)
dataset.push_to_hub("AlekseyKorshuk/roberta-no-topic-predictions")
