"""
Convenience entrypoint for preprocessing all crowd counting datasets.
Implementation lives in tools/preprocess_all_datasets.py.
"""
from tools.preprocess_all_datasets import preprocess_shanghaitech, preprocess_mall, preprocess_jhu

if __name__ == "__main__":
    preprocess_shanghaitech()
    preprocess_mall()
    preprocess_jhu()
    print("=" * 60)
    print("All datasets preprocessed and standardized successfully.")
