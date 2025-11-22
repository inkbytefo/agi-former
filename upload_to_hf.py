import os
from huggingface_hub import HfApi, create_repo
import argparse

def upload_to_hf(repo_id, token=None, model_dir="."):
    print(f"Uploading {model_dir} to Hugging Face Hub: {repo_id}...")
    
    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", token=token, exist_ok=True)
        print(f"Repository {repo_id} ready.")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # Upload files
    # We specifically want the model weights and the source code for reproducibility
    files_to_upload = [
        "best_model.pth",
        "last_model.pth",
        "train.py",
        "generate.py",
        "requirements.txt",
        "src/models/agiformer.py",
        "src/models/encoder.py",
        "src/models/layers.py",
        "src/data/real_data.py"
    ]
    
    for file_path in files_to_upload:
        full_path = os.path.join(model_dir, file_path)
        if os.path.exists(full_path):
            print(f"Uploading {file_path}...")
            try:
                api.upload_file(
                    path_or_fileobj=full_path,
                    path_in_repo=file_path,
                    repo_id=repo_id,
                    repo_type="model",
                    token=token
                )
            except Exception as e:
                print(f"Failed to upload {file_path}: {e}")
        else:
            print(f"Skipping {file_path} (not found)")
            
    print("Upload complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload AGIFORMER to Hugging Face")
    parser.add_argument("--repo", type=str, required=True, help="Hugging Face Repo ID (e.g., username/agiformer)")
    parser.add_argument("--token", type=str, help="Hugging Face Write Token (optional if logged in)")
    
    args = parser.parse_args()
    
    # Try to get token from env if not provided
    token = args.token or os.environ.get("HF_TOKEN")
    
    upload_to_hf(args.repo, token)
