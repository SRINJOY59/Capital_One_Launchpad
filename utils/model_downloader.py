from huggingface_hub import snapshot_download
import os, shutil

def download_model_flat(repo_id: str, save_dir: str):
    """
    Download a model repo from Hugging Face Hub into a flat directory
    (no symlinks, full files).
    """
    temp_cache = snapshot_download(
        repo_id=repo_id,
        local_dir=save_dir,
        local_dir_use_symlinks=False
    )

    if os.path.isdir(temp_cache):
        for f in os.listdir(temp_cache):
            src = os.path.join(temp_cache, f)
            dst = os.path.join(save_dir, f)
            if not os.path.exists(dst):
                shutil.move(src, dst)

    if os.path.isdir(temp_cache) and not os.listdir(temp_cache):
        os.rmdir(temp_cache)

    abs_path = os.path.abspath(save_dir)
    print(f"✅ Model ready for offline use at: {abs_path}")
    return abs_path


if __name__ == "__main__":
    base_repo = "Qwen/Qwen1.5-1.8B"
    base_dir = "../models/Qwen_1.5_1.8B"
    download_model_flat(base_repo, base_dir)

    adapter_repo = "anushree0107/agri-fine-tunes-qwen"
    adapter_dir = "../models/Qwen_1.5_Finetuned"
    download_model_flat(adapter_repo, adapter_dir)
