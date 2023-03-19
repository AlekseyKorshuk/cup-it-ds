# CUP IT | Hackathon

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

```bash
deepspeed train_reward_model.py \
  --model_path AlekseyKorshuk/cup-it-ds-sft-pretrained \
  --dataset_path ummagumm-a/cup-it-ds-classification-pairwise \
  --output_dir with-context
```