from pathlib import Path 

DATA_ROOTS = [ 
    Path("data/samples"),
    Path("data/raw"), 
]
def load_dataset(name):
    
    dataset_path = None

    for root in DATA_ROOTS:
        canidate = root / name 
        if canidate.exists():
            dataset_path = canidate
            break

    if dataset_path is None:
        raise FileNotFoundError(
            f"Dataset {name} not found in any directory."
        )

    code_file = dataset_path / "code.txt"
    label_file = dataset_path / "labels.txt"

    if not code_file.exists() or not label_file.exists():
        raise FileNotFoundError(
            "Dataset must contain code.txt and labels.txt"
        )

    with open(code_file, "r") as f:
        code = [line.strip() for line in f if line.strip()]

    with open(label_file, "r") as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    return code, labels




def list_datasets():
    datasets = set()

    for root in DATA_ROOTS:
        if root.exists():
            for p in root.iterdir():
                if p.is_dir():
                    datasets.add(p.name)

    return sorted(datasets)


if __name__ == "__main__":
    print("Available datasets:", list_datasets())

if __name__ == "__main__":
    code, labels = load_dataset("github_ai")
    
    print("total code lines:", len(code))
    print("total labels:", len(labels))

    


