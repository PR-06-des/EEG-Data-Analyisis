from pathlib import Path

import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath, read_raw_bids

from src.analysis import epoch_and_average
from src.download import download_subject_data
from src.preprocessing import preprocess_eeg


DATASET_DIR = Path(__file__).parent / "ds003061"


def main():
    # 1. Download dataset
    download_subject_data(dataset_id="ds003061", subject_id="sub-001", target_dir=DATASET_DIR)

    # 2. Read BIDS data
    bids_path = BIDSPath(
        subject="001",
        task="P300",
        run="1",
        datatype="eeg",
        root=DATASET_DIR
    )
    raw = read_raw_bids(bids_path=bids_path, verbose=False)

    # 3. Preprocess
    raw_clean = preprocess_eeg(raw)

    # 4. Epoch & Compute ERPs
    evoked_target, evoked_standard = epoch_and_average(raw_clean)

    # 5. Visualize Results
    mne.viz.plot_compare_evokeds(
        {"Oddball (Target)": evoked_target, "Standard Tone": evoked_standard},
        picks="Pz",
        title="ds003061 sub-001: P300 Response at Pz"
    )
    
    evoked_target.plot_topomap(times=[0.1, 0.2, 0.35, 0.5], ch_type="eeg")
    plt.show()

if __name__ == "__main__":
    main()