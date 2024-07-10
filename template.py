import os
from pathlib import Path

list_of_files = [
    "src/component/ingest.py",
    "src/component/train.py",
    "src/component/predict.py",
    "src/component/evaluate.py",
    "src/component/visualize.py",
    "setup.py",
    "main.py"
]

for file in list_of_files:
    dir_name, file_name = os.path.split(file)
    
    # Create the directory if it doesn't exist
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    # Create the file if it doesn't exist
    if not os.path.exists(file):
        with open(file, "w") as f:
            pass
    elif os.path.getsize(file) == 0:
        print("File exists but its size is zero")
    else:
        print("File exists and its size is greater than zero")