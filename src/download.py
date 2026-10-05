import os
import openneuro

def download_subject_data(dataset_id: str = "ds003061", subject_id: str = "sub-001", target_dir: str = "./ds003061"):
    """Downloads a specific subject's data from an OpenNeuro dataset."""
    if not os.path.exists(target_dir):
        print(f"Downloading {subject_id} from {dataset_id}...")
        openneuro.download(
            dataset=dataset_id,
            target_dir=target_dir,
            include=[f"{subject_id}/*"]
        )
        print("Download complete.")
    else:
        print(f"Dataset directory '{target_dir}' already exists. Skipping download.")