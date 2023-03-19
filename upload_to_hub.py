from huggingface_hub import HfApi

repo_id = "AlekseyKorshuk/cup-it-ds-reward-model-with-context"

api = HfApi()
api.create_repo(repo_id=repo_id, exist_ok=True)
api.upload_file(
    path_or_fileobj="./rm_checkpoint/no-context/checkpoint-4956/pytorch_model.bin",
    path_in_repo="pytorch_model.bin",
    repo_id=repo_id,
)
