import mne
from mne.preprocessing import ICA

def preprocess_eeg(raw: mne.io.Raw, l_freq: float = 0.5, h_freq: float = 30.0) -> mne.io.Raw:
    """
    Cleans raw EEG signal:
    1. Sets standard 10-20 montage.
    2. Filters (bandpass + notch).
    3. Re-references to average.
    4. Removes ocular artifacts via ICA.
    """

    # Filtering and ICA require the signal to be loaded into memory.
    if not raw.preload:
        raw.load_data()

    # Set standard 10-20 montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)

    # Filter the data (bandpass + notch)
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', skip_by_annotation='edge')
    raw.notch_filter(freqs=60.0)

    # Re-reference to average
    raw.set_eeg_reference("average", projection=False)

    # Remove ocular artifacts via ICA
    n_eeg_channels = len(mne.pick_types(raw.info, eeg=True, exclude="bads"))
    if n_eeg_channels < 2:
        raise ValueError("At least two non-bad EEG channels are required for ICA.")

    ica = ICA(n_components=min(15, n_eeg_channels), random_state=97)
    ica.fit(raw)
    ica.exclude = [0]  # Exclude the first component (ocular artifact)
    raw = ica.apply(raw)

    return raw