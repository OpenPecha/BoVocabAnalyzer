"""Hugging Face Hub utilities for downloading ARPA language model files."""

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from BoVocabAnalyzer.core.models import ModelInfo

_ARPA_EXTENSION = ".arpa"


def _find_arpa_file(repo_id: str) -> str:
    """List files in a Hugging Face repo and return the first .arpa filename.

    Args:
        repo_id: Hugging Face repository identifier (e.g. ``"org/model"``).

    Returns:
        The relative path of the ``.arpa`` file inside the repo.

    Raises:
        FileNotFoundError: If no ``.arpa`` file exists in the repository.
    """
    api = HfApi()
    files = api.list_repo_files(repo_id)
    arpa_files = [f for f in files if f.endswith(_ARPA_EXTENSION)]

    if not arpa_files:
        raise FileNotFoundError(
            f"No {_ARPA_EXTENSION} file found in Hugging Face repo '{repo_id}'. "
            f"Available files: {files}"
        )

    return arpa_files[0]


def download_arpa_from_hf(repo_id: str) -> ModelInfo:
    """Download an ARPA file from a Hugging Face repository.

    Automatically detects the ``.arpa`` file in the repo, downloads it to
    the local Hugging Face cache, and returns a :class:`ModelInfo` with
    the resolved local path.

    Args:
        repo_id: Hugging Face repository identifier (e.g. ``"org/model"``).

    Returns:
        A :class:`ModelInfo` populated with the model name, local ARPA path,
        and the original repo id.

    Raises:
        FileNotFoundError: If no ``.arpa`` file is found in the repo.
    """
    arpa_filename = _find_arpa_file(repo_id)

    local_path = hf_hub_download(repo_id=repo_id, filename=arpa_filename)

    model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

    return ModelInfo(
        name=model_name,
        arpa_path=Path(local_path),
        source=repo_id,
    )
