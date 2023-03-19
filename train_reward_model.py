import functools
import os

import pandas as pd
import scipy
import torch
from datasets import load_dataset
from torch import nn

from reward_model import GPTRewardModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
import wandb


def create_comparison_dataset(dataset):
    pairs = []
    for sample in tqdm(dataset, desc="Creating comparison dataset"):
        pair = {}
        prompt = sample["prompt"]
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        if chosen_summary == rejected_summary:
            continue
        pair["chosen"] = prompt + "\n\n" + chosen_summary
        pair["rejected"] = prompt + "\n\n" + rejected_summary
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


import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer


def dcg_score(y_true, y_score, k=5):
    """Discounted cumulative gain (DCG) at rank K.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score(ground_truth, predictions, k=5):
    """Normalized discounted cumulative gain (NDCG) at rank K.

    Normalized Discounted Cumulative Gain (NDCG) measures the performance of a
    recommendation system based on the graded relevance of the recommended
    entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal
    ranking of the entities.

    Parameters
    ----------
    ground_truth : array, shape = [n_samples]
        Ground truth (true labels represended as integers).
    predictions : array, shape = [n_samples, n_classes]
        Predicted probabilities.
    k : int
        Rank.

    Returns
    -------
    score : float

    Example
    -------
    >>> ground_truth = [1, 0, 2]
    >>> predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    1.0
    >>> predictions = [[0.9, 0.5, 0.8], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    0.6666666666
    """
    lb = LabelBinarizer()
    lb.fit(range(len(predictions) + 1))
    T = lb.transform(ground_truth)

    scores = []

    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


from sklearn.metrics import ndcg_score as sk_ndcg_score


class MyCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # print(state, metrics)
        preds = torch.tensor(trainer.predict(val_dataset)[0])
        print("before", preds.shape)
        preds = preds.view(-1, 2)
        print("after", preds.shape)

        samples = {"prompt": [], "chosen": [], "rejected": [], "scores": []}
        for i in range(16):
            ele = dataset[validation_split_name][i]
            samples["prompt"].append(ele["prompt"])
            samples["chosen"].append(ele["chosen"])
            samples["rejected"].append(ele["rejected"])
            samples["scores"].append(preds[i].tolist())

        grouped_preds = []
        for x in batch(preds, 4):
            if len(x) != 4:
                continue
            grouped_preds.append(
                [
                    x[0][0],
                    x[1][0],
                    x[2][0],
                    x[3][0],
                    x[3][1],
                ]
            )

        ground_truth = [0] * len(grouped_preds)
        custom_ndcg = {}
        for k in range(1, 5 + 1):
            ndcg_ = ndcg_score(ground_truth, scipy.special.softmax(grouped_preds, axis=1), k=k)
            custom_ndcg[f"grouped/custom_ndcg/k={k}"] = ndcg_

        sk_ndcg = {}
        for k in range(1, 5 + 1):
            sk_ndcg_ = sk_ndcg_score(
                y_true=[[4, 3, 2, 1, 0]] * len(grouped_preds),
                y_score=scipy.special.softmax(grouped_preds, axis=1),
                k=k
            )
            sk_ndcg[f"grouped/sk_ndcg/k={k}"] = sk_ndcg_

        pair_sk_ndcg = {}
        for k in range(1, 5 + 1):
            sk_ndcg_ = sk_ndcg_score(
                y_true=[[1, 0]] * len(preds),
                y_score=scipy.special.softmax(preds, axis=1),
                k=k
            )
            pair_sk_ndcg[f"paired/sk_ndcg/k={k}"] = sk_ndcg_

        pair_custom_ndcg = {}
        for k in range(1, 5 + 1):
            sk_ndcg_ = ndcg_score(
                [0] * len(preds),
                scipy.special.softmax(preds, axis=1),
                k=k
            )
            pair_custom_ndcg[f"paired/custom_ndcg/k={k}"] = sk_ndcg_

        # Subtracting rejected scores from chosen scores
        diff = preds[:, 0] - preds[:, 1]
        acc = (diff >= 0).type(torch.float32).mean().item()
        if torch.distributed.get_rank() == 0:
            print("Testing accuracy: ", acc)
            if torch.distributed.get_rank() == 0:
                wandb.log({"samples": wandb.Table(data=pd.DataFrame(samples))})
                results = {"acc": acc}
                results.update(sk_ndcg)
                results.update(custom_ndcg)
                results.update(pair_sk_ndcg)
                results.update(pair_custom_ndcg)
                wandb.log(results)


def rgetattr(obj, attr: str, *args):
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def findattr(obj, attrs):
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)


def hf_get_causal_hidden_layers(model: nn.Module):
    """Returns the hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
    """
    hidden_layers_attrs = (
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
        "transformer.layers",
    )
    return findattr(model, hidden_layers_attrs)


if __name__ == "__main__":
    MODEL_PATH = "AlekseyKorshuk/cup-it-ds-sft-pretrained"
    TOKENIZER_PATH = "AlekseyKorshuk/cup-it-ds-sft-pretrained"
    data_path = "AlekseyKorshuk/cup-it-ds-pairwise"

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
        eval_steps=100,
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
    layers = hf_get_causal_hidden_layers(model)
    num_layers = len(layers)
    num_unfrozen = int(0.7 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    # Create the comparisons datasets
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

    # dataset["train"] = dataset["train"].shuffle(seed=42)
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
