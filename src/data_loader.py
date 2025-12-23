from pathlib import Path 

DATA_DIR = Path("data/samples")

def load_dataset(name):
    dataset_path = DATA_DIR / name # take data folder and go to the dataset folder
    code_file = dataset_path / "code.txt" #
    label_file = dataset_path / "labels.txt"

    if not code_file.exists() or not label_file.exists():
        raise FileNotFoundError("Dataset must contain code.txt and labels.txt")

    with open(code_file, "r", encoding='utf-8')as f:
        code = f.read().splitlines()

    with open(label_file, "r", encoding="utf-8")as f:
        labels = list(map(int, f.read().splitlines())) 

    if len(code) != len(labels):
        raise ValueError("Code and labels length mismatch")
    
    return code, labels

def list_datasets():
    if not DATA_DIR.exists():
        return []
    return [p.name for p in DATA_DIR.iterdir() if p.is_dir()]

if __name__ == "__main__":
    print("Available datasets:", list_datasets())

if __name__ == "__main__":
    code, labels = load_dataset("basic")
    print(code)
    print(labels)


