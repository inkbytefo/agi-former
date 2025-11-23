## Developer: inkbytefo
## Modified: 2025-11-23

import os
from huggingface_hub import HfApi, create_repo
import argparse

def upload_to_hf(repo_id, token, commit_message="Update AGIFORMER Phase 7"):
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
        # Phase 7: Curriculum Models (PRIMARY)
        ("best_model_curriculum.pth", "models/curriculum/"),
        ("last_model_curriculum.pth", "models/curriculum/"),
        ("metrics_curriculum.json", "models/curriculum/"),
        
        # Models (Turkish - Baseline)
        ("best_model_turkish.pth", "models/turkish_baseline/"),
        ("last_model_turkish.pth", "models/turkish_baseline/"),
        ("metrics_turkish.json", "models/turkish_baseline/"),
        
        # Source code
        ("src/models/agiformer.py", "src/models/"),
        ("src/models/encoder.py", "src/models/"),
        ("src/models/layers.py", "src/models/"),
        ("src/models/reasoning.py", "src/models/"),
        ("src/models/memory.py", "src/models/"),  # Hebbian Memory
        ("src/data/clean_turkish_data.py", "src/data/"),
        ("src/data/curriculum.py", "src/data/"),  # Phase 7
        
        # Training scripts
        ("train_curriculum.py", ""),  # Phase 7 main training
        ("train_turkish.py", ""),
        ("generate.py", ""),
        ("test_recall.py", ""),
        ("inspect_reasoning.py", ""),
        
        # Documentation
        ("README.md", ""),
        ("PROGRESS_REPORT_Phase7.md", ""),  # Phase 7 report
        ("AGIFORMER_Technical_Report.md", ""),
        ("docs/architecture.md", "docs/"),
        ("docs/training.md", "docs/"),
        ("docs/inference.md", "docs/"),
        ("docs/api.md", "docs/"),
        ("docs/RFC_007_Curriculum_Learning.md", "docs/"),  # Phase 7 RFC
        
        # Config
        ("requirements.txt", ""),
    ]
    
    print(f"\n📤 Uploading to {repo_id}...")
    print(f"🎓 Phase 7: Curriculum Learning with Neuroplasticity")
    
    uploaded = 0
    skipped = 0
    
    for local_path, hf_path in files_to_upload:
        if not os.path.exists(local_path):
            print(f"⏭️  Skip: {local_path} (not found)")
            skipped += 1
            continue
        
        try:
            file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=hf_path + os.path.basename(local_path),
                repo_id=repo_id,
                token=token,
                commit_message=commit_message
            )
            print(f"✅ {local_path} ({file_size:.1f}MB) → {hf_path}")
            uploaded += 1
        except Exception as e:
            print(f"❌ Failed: {local_path} - {e}")
    
    print(f"\n📊 Summary: {uploaded} uploaded, {skipped} skipped")
    print(f"🌐 View at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True, help="HF repo ID (e.g., username/agiformer)")
    parser.add_argument("--token", type=str, required=True, help="HF token")
    parser.add_argument("--message", type=str, default="Phase 7: Curriculum Learning (20K steps, BPC 1.78)", help="Commit message")
    
    args = parser.parse_args()
    
    upload_to_hf(args.repo, args.token, args.message)
