# CUP IT | Hackathon

## Weights & Biases Report

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
