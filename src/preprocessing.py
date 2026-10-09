import mne
from mne.preprocessing import ICA


def preprocess_eeg(raw, l_freq=1.0, h_freq=40.0, *, bad_channels=(),
                   ica_exclude=(), return_ica=False):
    """Clean a copy; exclusions require visual review, never assume IC 0."""
    clean = raw.copy().load_data()
    unknown = set(bad_channels) - set(clean.ch_names)
    if unknown:
        raise ValueError(f"Unknown bad channels: {sorted(unknown)}")
    clean.info["bads"] = sorted(set(clean.info["bads"]) | set(bad_channels))
    if clean.get_montage() is None:
        clean.set_montage(mne.channels.make_standard_montage("standard_1020"),
                          match_case=False, on_missing="raise")
    clean.filter(l_freq, h_freq, picks="eeg",
                 skip_by_annotation=("edge", "bad_acq_skip"))
    line = clean.info.get("line_freq")
    if line and (h_freq is None or line < h_freq):
        clean.notch_filter([line], picks="eeg")
    clean.set_eeg_reference("average", projection=False)
    rank = mne.compute_rank(clean).get("eeg", 0)
    if rank < 2:
        raise ValueError("At least two independent EEG signals are required.")
    training = clean.copy()
    if l_freq is None or l_freq < 1:
        training.filter(1.0, None, picks="eeg")
    ica = ICA(n_components=rank, method="infomax",
              fit_params=dict(extended=True), random_state=97, max_iter="auto")
    ica.fit(training, picks="eeg", reject_by_annotation=True)
    exclusions = sorted(set(ica_exclude))
    if any(i < 0 or i >= ica.n_components_ for i in exclusions):
        raise ValueError("ICA exclusion index is outside the fitted range.")
    ica.exclude = exclusions
    if exclusions:
        ica.apply(clean)
    if clean.info["bads"]:
        clean.interpolate_bads(reset_bads=True)
        clean.set_eeg_reference("average", projection=False)
    return (clean, ica) if return_ica else clean

