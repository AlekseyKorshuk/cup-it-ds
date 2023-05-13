import scipy
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import os

model_path = "roberta-large"
dataset = load_dataset("ummagumm-a/cup_it_ds_split_with_lang_with_topic")

tokenizer = AutoTokenizer.from_pretrained(model_path)


def preprocess_function(examples):
    labels = []
    prepared_examples = []
    examples = [dict(zip(examples, t)) for t in zip(*examples.values())]
    for example in examples:
        for comment in example["comments"]:
            prepared_examples.append(
                example['text'].strip() + tokenizer.sep_token + comment["text"].strip()
            )
            labels.append(int(comment["score"]))
    tokenized_dict = tokenizer(prepared_examples, truncation=True)
    tokenized_dict["label"] = labels
    return tokenized_dict


tokenized_dataset = dataset.map(preprocess_function,
                                batched=True,
                                remove_columns=dataset["train"].features,
                                num_proc=os.cpu_count()
                                )

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")


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


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    results = {
        "ndcg": ndcg_score(labels, scipy.special.softmax(predictions, axis=1), k=2)
    }
    acc = accuracy.compute(predictions=np.argmax(predictions, axis=1), references=labels)
    results.update(acc)
    print(results)
    return results


id2label = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}
label2id = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
# id2label = {0: "0", 1: "1"}
# label2id = {"0": 0, "1": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=len(id2label.keys()), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="/tmp/roberta",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=400,
    save_strategy="steps",
    save_steps=400,
    load_best_model_at_end=True,
    push_to_hub=True,
    deepspeed="ds_config_gpt_j.json",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub()
