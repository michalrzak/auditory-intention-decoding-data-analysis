from pathlib import Path


def file_to_label_file(file: Path) -> Path:
    return file.parent / (str(file.stem) + "-labels.csv")
