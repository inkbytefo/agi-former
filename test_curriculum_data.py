import os
import sys
from src.data.curriculum import prepare_dictionary_data, prepare_stories_data

def test_data_prep():
    print("Testing Dictionary Prep...")
    path1 = prepare_dictionary_data("./data")
    if path1 and os.path.exists(path1):
        print(f"✅ Dictionary data created at {path1}")
        size = os.path.getsize(path1)
        print(f"Size: {size/1024/1024:.2f} MB")
    else:
        print("❌ Dictionary data failed")

    print("\nTesting Stories Prep...")
    path2 = prepare_stories_data("./data")
    if path2 and os.path.exists(path2):
        print(f"✅ Stories data created at {path2}")
        size = os.path.getsize(path2)
        print(f"Size: {size/1024/1024:.2f} MB")
    else:
        print("❌ Stories data failed")

if __name__ == "__main__":
    test_data_prep()
