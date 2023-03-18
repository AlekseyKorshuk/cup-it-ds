import scipy
from datasets import load_dataset

imdb = load_dataset("AlekseyKorshuk/cup-it-ds-classification")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_imdb = imdb.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import evaluate

accuracy = evaluate.load("accuracy")
nDCG_metric = evaluate.load('JP-SystemsX/nDCG')

import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    results = nDCG_metric.compute(references=labels, predictions=scipy.special.softmax(predictions, axis=1))
    acc = accuracy.compute(predictions=np.argmax(predictions, axis=1), references=labels)
    return results.update(acc)


id2label = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}
label2id = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=5, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="/tmp/bert",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub()
