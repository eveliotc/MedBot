from dotenv import load_dotenv
from huggingface_hub import HfApi
import os

load_dotenv()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def _hf_fetch(repo_id, matching, ignore=[]):
    api = HfApi()

    all_remote_files = api.list_repo_files(repo_id)
    remote_files = list(
        filter(
            lambda remote_file: matching in remote_file
            and not any(ignore_keyword in remote_file for ignore_keyword in ignore),
            all_remote_files,
        )
    )

    for remote_file in remote_files:
        api.hf_hub_download(
            repo_id=repo_id, force_download=False, filename=remote_file, local_dir="."
        )


_datasets = ["textbooks", "pubmed", "statpearls", "wikipedia"]


def _dataset_fetch(dataset):
    assert dataset in _datasets
    _hf_fetch(repo_id="eveliotc/MedBot", matching=f"corpus/{dataset}", ignore=[".npy"])


if __name__ == "__main__":
    _dataset_fetch("textbooks")
