from pathlib import Path
import openneuro


def download_subject_data(dataset_id="ds003061", subject_id="sub-001",
                          target_dir="./ds003061"):
    """Resume downloads: an existing directory does not mean complete data."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    openneuro.download(dataset=dataset_id, target_dir=str(target_dir),
                       include=[subject_id + "/*"])

