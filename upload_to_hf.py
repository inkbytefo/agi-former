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
