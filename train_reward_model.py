import os

import pandas as pd
import torch
from datasets import load_dataset
from reward_model import GPTRewardModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
import wandb
import re


def create_comparison_dataset(dataset):
    pairs = []
    for sample in tqdm(dataset, desc="Creating comparison dataset"):
        pair = {}
        prompt = sample["prompt"]
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        if chosen_summary == rejected_summary:
            continue
        # if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
        #     continue
        pair["chosen"] = prompt + "<|endoftext|>" + chosen_summary
        pair["rejected"] = prompt + "<|endoftext|>" + rejected_summary
        pairs.append(pair)
    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in tqdm(pairs, desc="Tokenizing train dataset"):
            chosen, rejected = pair["chosen"], pair["rejected"]
            if chosen == rejected:
                continue
            chosen_encodings_dict = tokenizer(
                "<|startoftext|>" + chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                "<|startoftext|>" + rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
            self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
            self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
            self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )


class PairwiseEvalDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []

        for pair in tqdm(pairs, desc="Tokenizing eval dataset"):
            chosen, rejected = pair["chosen"], pair["rejected"]
            if chosen == rejected:
                continue
            chosen_encodings_dict = tokenizer(
                "<|startoftext|>" + chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                "<|startoftext|>" + rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.input_ids.append(chosen_encodings_dict['input_ids'])
            self.attn_masks.append(chosen_encodings_dict['attention_mask'])
            self.input_ids.append(rejected_encodings_dict['input_ids'])
            self.attn_masks.append(rejected_encodings_dict['attention_mask'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        if len(data[0]) == 4:
            batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
            batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        elif len(data[0]) == 2:
            batch["input_ids"] = torch.cat([f[0] for f in data])
            batch["attention_mask"] = torch.cat([f[1] for f in data])
        else:
            raise ValueError("Invalid data format")
        return batch


def compute_metrics(eval_preds):
    print("EVAL!!!")
    preds = eval_preds.predictions[0].view(-1, 2)
    acc = sum(preds[:, 0] >= preds[:, 1]) / preds.shape[0]
    if torch.distributed.get_rank() == 0:
        wandb.log({"acc": acc})
    return {"accuracy": acc}


class SparsePairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        assert len(inputs["input_ids"].shape) == 2
        bs = inputs["input_ids"].shape[0] // 2
        rewards = model(**inputs)
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        return (loss, rewards) if return_outputs else loss


class MyCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # print(state, metrics)
        preds = torch.tensor(trainer.predict(val_dataset)[0])
        preds = preds.view(-1, 2)
        samples = {"prompt": [], "chosen": [], "rejected": [], "scores": []}
        for i in range(16):
            ele = dataset[validation_split_name][i]
            samples["prompt"].append(ele["prompt"])
            samples["chosen"].append(ele["chosen"])
            samples["rejected"].append(ele["rejected"])
            samples["scores"].append(preds[i].tolist())
        # Subtracting rejected scores from chosen scores
        diff = preds[:, 0] - preds[:, 1]
        acc = (diff >= 0).type(torch.float32).mean().item()
        if torch.distributed.get_rank() == 0:
            print("Testing accuracy: ", acc)
            if torch.distributed.get_rank() == 0:
                wandb.log({"samples": wandb.Table(data=pd.DataFrame(samples))})
                wandb.log({"acc": acc})


if __name__ == "__main__":
    # MODEL_PATH = "Dahoas/gptneo-sft-static"
    MODEL_PATH = "gpt2"
    TOKENIZER_PATH = "gpt2"
    # MODEL_PATH = "AlekseyKorshuk/gpt-neo-125M-sft"

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, truncation_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists("rm_checkpoint"):
        os.mkdir("rm_checkpoint")

    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir="rm_checkpoint/",
        num_train_epochs=1,
        logging_steps=10,
        gradient_accumulation_steps=2,
        save_strategy="epoch",
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_steps=1000,
        save_steps=1,
        warmup_steps=100,
        logging_dir="./logs",
        fp16=True,
        bf16=False,
        learning_rate=1e-5,
        deepspeed="ds_config_gpt_j.json",
        weight_decay=0.01,
        # save_total_limit=5,
        report_to="wandb"
    )

    tokenizer.pad_token = tokenizer.eos_token

    # Initialize the reward model from the (supervised) fine-tuned GPT-J
    model = GPTRewardModel(MODEL_PATH, tokenizer(tokenizer.pad_token)["input_ids"][0])

    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.7 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    # Create the comparisons datasets
    data_path = "AlekseyKorshuk/cup-it-ds-pairwise-small"
    # data_path = "Dahoas/rm-static"
    # data_path = "CarperAI/openai_summarize_comparisons"

    # dataset = load_dataset(data_path)
    # dataset["test"] = load_dataset(
    #     data_path,
    #     split=f"train[:{5}%]",
    # )
    # dataset["train"] = load_dataset(
    #     data_path,
    #     split=f"train[{5}%:{50}%]",
    # )

    validation_split_name = "validation"
    validation_split_percentage = 5

    dataset = load_dataset(data_path)
    if validation_split_name not in dataset.keys():
        dataset[validation_split_name] = load_dataset(
            data_path,
            split=f"train[:{validation_split_percentage}%]",
        )
        dataset["train"] = load_dataset(
            data_path,
            split=f"train[{validation_split_percentage}%:]",
        )

    dataset = dataset.shuffle(seed=42)
    train_pairs = create_comparison_dataset(dataset["train"])
    val_pairs = create_comparison_dataset(dataset[validation_split_name])

    # Make pairwise datasets for training
    max_length = 512
    train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
    val_dataset = PairwiseEvalDataset(val_pairs, tokenizer, max_length=max_length)

    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward()

    callbacks = [
        MyCallback()
    ]
    trainer = SparsePairwiseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        preprocess_logits_for_metrics=None,
        callbacks=callbacks
    )
    trainer.train()

    # NOTE: In order to run this install transformers from source
    # per https://github.com/huggingface/transformers/issues/20942
    preds = torch.tensor(trainer.predict(val_dataset)[0])
    preds = preds.view(-1, 2)
    samples = {"prompt": [], "chosen": [], "rejected": [], "scores": []}
    for i in range(16):
        ele = dataset[validation_split_name][i]
        samples["prompt"].append(ele["prompt"])
        samples["chosen"].append(ele["chosen"])
        samples["rejected"].append(ele["rejected"])
        samples["scores"].append(preds[i].tolist())
    # Subtracting rejected scores from chosen scores
    diff = preds[:, 0] - preds[:, 1]
    acc = (diff >= 0).type(torch.float32).mean().item()
    print("Testing accuracy: ", acc)
    if torch.distributed.get_rank() == 0:
        wandb.log({"samples": wandb.Table(data=pd.DataFrame(samples))})
        wandb.log({"acc": acc})
