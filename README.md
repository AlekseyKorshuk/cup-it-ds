# CUP IT | Hackathon

## About

In this task we are required to sort/range comment in social network. We desided to use reward part from RLHF approach.

## Approach

The pairwise training of reward model, as used in the InstructGPT paper by OpenAI, is an effective method for rating comments on a given post. This approach allows the model to learn from the relative differences between comments, rather than relying on an absolute rating scale. This is especially useful when the rating scale is inconsistent or not well-defined. By training the model on relative differences between pairs of comments, it becomes less reliant on a predefined rating scale and is better able to generalize to new data. Overall, pairwise training of reward model is a sound choice for training a model to rate comments on a given post, as it enables more nuanced and accurate ratings and is more robust to inconsistencies in the rating scale.

For each row in the dataset, a list of comemnts is given with the score from 0 to 4 (0 - best, 4 - worst). We use this data to train a reward model that maps a **(post, comment)** pair to a reward **r**. The reward model is trained to predict which comment a human will prefer, using the rewards as logits.

There are 2 main steps:
- Supervised fine-tuning on the given dataset
- Reward model training based on comparisons

### Reward model training deteils

We used the following loss function to train our reward model:

```python
loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
```


## Results

We used NDCG metrics to compare runs.

We compared 2 main models:
- with no additional context
- with post context from Google Big Query dataset: [link](https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=hacker_news&t=full&page=table&project=trim-field-381012&ws=!1m10!1m4!4m3!1sbigquery-public-data!2shacker_news!3sstories!1m4!4m3!1sbigquery-public-data!2shacker_news!3scomments!1m10!1m4!1m3!1strim-field-381012!2sbquxjob_70493aea_186f5af4856!3sUS!1m4!4m3!1sbigquery-public-data!2shacker_news!3sfull)

### Grouped NDCG with k=5

<img width="1241" alt="Screenshot 2023-03-19 at 18 51 28" src="https://user-images.githubusercontent.com/48794610/226228640-cabdba0d-281e-4749-b46b-606878ef1eaa.png">


### Paired NDCG with k=2

<img width="1246" alt="Screenshot 2023-03-19 at 18 50 54" src="https://user-images.githubusercontent.com/48794610/226228591-3d0d3f37-caa4-40db-b967-04003f15e5bb.png">


### Weights & Biases Report

https://wandb.ai/aleksey-korshuk/huggingface/reports/CUP-IT-Report--VmlldzozODMzODI0

## Train SFT model

```bash
deepspeed train_sft.py \
  --model_name_or_path gpt2 \
  --dataset_name AlekseyKorshuk/up-it-ds-sft \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --do_train \
  --do_eval \
  --output_dir /tmp/test-clm \
  --push_to_hub
```

## Train reward model

```bash
deepspeed train_reward_model.py \
  --model_path AlekseyKorshuk/cup-it-ds-sft-pretrained \
  --dataset_path AlekseyKorshuk/cup-it-ds-pairwise \
  --output_dir no-context
```

Resulting model: https://huggingface.co/AlekseyKorshuk/cup-it-ds-reward-model-no-context

```bash
deepspeed train_reward_model.py \
  --model_path AlekseyKorshuk/cup-it-ds-sft-pretrained \
  --dataset_path ummagumm-a/cup-it-ds-classification-pairwise-train-val \
  --output_dir with-context
```

Resulting model: https://huggingface.co/AlekseyKorshuk/cup-it-ds-reward-model-with-context

## Inference

To generate scores for test dataset:

```bash
wget  https://huggingface.co/AlekseyKorshuk/cup-it-ds-reward-model-no-context/resolve/main/pytorch_model.bin -O ./rm_checkpoint/no-context/checkpoint-4956/pytorch_model.bin
python3 inference.py
```
