import os

def inspect_data():
    path = "data/trwiki_clean_train.bin"
    if not os.path.exists(path):
        print(f"{path} not found.")
        return

    with open(path, "rb") as f:
        data = f.read(1000)
        
    print(f"--- First 1000 bytes of {path} ---")
    print(data)
    print("\n--- Decoded ---")
    try:
        print(data.decode('utf-8'))
    except Exception as e:
        print(f"Decode error: {e}")

if __name__ == "__main__":
    inspect_data()
