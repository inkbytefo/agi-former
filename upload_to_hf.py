## Developer: inkbytefo
## Modified: 2025-11-22

import os
from huggingface_hub import HfApi, create_repo
import argparse

def upload_to_hf(repo_id, token, commit_message="Update AGIFORMER"):
    """
    Upload AGIFORMER model and code to Hugging Face Hub.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "username/repo-name")
        token: HuggingFace API token
        commit_message: Commit message for the upload
    """
    api = HfApi()
    
    # Create repo if doesn't exist
    try:
        create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
        print(f"✅ Repository {repo_id} ready")
    except Exception as e:
        print(f"⚠️ Repo creation: {e}")
    
    # Files to upload
    files_to_upload = [
        # Models (English)
        ("best_model.pth", "models/english/"),
        ("last_model.pth", "models/english/"),
        
        # Models (Turkish)
        ("best_model_turkish.pth", "models/turkish/"),
        ("last_model_turkish.pth", "models/turkish/"),
        
        # Source code
        ("src/models/agiformer.py", "src/models/"),
        ("src/models/encoder.py", "src/models/"),
        ("src/models/layers.py", "src/models/"),
        ("src/models/reasoning.py", "src/models/"),
        ("src/data/real_data.py", "src/data/"),
        ("src/data/turkish_wiki.py", "src/data/"),
        
        # Training scripts
        ("train.py", ""),
        ("train_turkish.py", ""),
        ("generate.py", ""),
        ("inspect_reasoning.py", ""),
        
        # Benchmark results
        ("metrics_english.json", "benchmark/"),
        ("metrics_turkish.json", "benchmark/"),
        ("comparison_turkish_vs_english.png", "benchmark/"),
        ("benchmark_report.md", "benchmark/"),
        
        # Comparison
        ("compare_benchmarks.py", ""),
        
        # Documentation
        ("README.md", ""),
        ("docs/architecture.md", "docs/"),
        ("docs/training.md", "docs/"),
        ("docs/inference.md", "docs/"),
        ("docs/api.md", "docs/"),
        
        # Config
        ("requirements.txt", ""),
    ]
    
    print(f"\n📤 Uploading to {repo_id}...")
    
    uploaded = 0
    skipped = 0
    
    for local_path, hf_path in files_to_upload:
        if not os.path.exists(local_path):
            print(f"⏭️  Skip: {local_path} (not found)")
            skipped += 1
            continue
        
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=hf_path + os.path.basename(local_path),
                repo_id=repo_id,
                token=token,
                commit_message=commit_message
            )
            print(f"✅ {local_path} → {hf_path}")
            uploaded += 1
        except Exception as e:
            print(f"❌ Failed: {local_path} - {e}")
    
    print(f"\n📊 Summary: {uploaded} uploaded, {skipped} skipped")
    print(f"🌐 View at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True, help="HF repo ID")
    parser.add_argument("--token", type=str, required=True, help="HF token")
    parser.add_argument("--message", type=str, default="Update AGIFORMER with Turkish benchmark", help="Commit message")
    
    args = parser.parse_args()
    
    upload_to_hf(args.repo, args.token, args.message)
