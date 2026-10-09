import warnings

import mne
from mne.preprocessing import ICA


def preprocess_eeg(raw: mne.io.Raw, l_freq: float = 0.5, h_freq: float = 30.0) -> mne.io.Raw:
    """Filter and average-reference EEG; remove EOG artifacts when identifiable."""

    if not raw.preload:
        raw.load_data()

    # Preserve channel positions loaded from the recording/BIDS metadata.
    if raw.get_montage() is None:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False)

    raw.filter(
        l_freq=l_freq, h_freq=h_freq, picks="eeg",
        fir_design="firwin", skip_by_annotation=("edge", "bad_acq_skip"),
    )
    # Use acquisition metadata; ds003061 uses 50 Hz, not 60 Hz.
    line_freq = raw.info.get("line_freq")
    if line_freq and (h_freq is None or line_freq < h_freq):
        raw.notch_filter(freqs=line_freq, picks="eeg")

    raw.set_eeg_reference("average", projection=False)

    # Average referencing reduces the number of independent EEG signals.
    rank = mne.compute_rank(raw).get("eeg", 0)
    if rank < 2:
        raise ValueError("At least two independent non-bad EEG signals are required for ICA.")

    # Fit ICA on >=1 Hz data while retaining the requested ERP filter.
    ica_raw = raw.copy()
    if l_freq is None or l_freq < 1.0:
        ica_raw.filter(l_freq=1.0, h_freq=None, picks="eeg")
    ica = ICA(n_components=min(15, rank), random_state=97, max_iter="auto")
    ica.fit(ica_raw, picks="eeg", reject_by_annotation=True)

    # Component 0 is not necessarily an ocular artifact.
    eog_picks = mne.pick_types(raw.info, eog=True, exclude="bads")
    if len(eog_picks):
        ica.exclude, _ = ica.find_bads_eog(ica_raw)
        if ica.exclude:
            raw = ica.apply(raw)
    else:
        warnings.warn(
            "No EOG channels are marked: ICA components were not removed. "
            "Review the components or identify a validated EOG channel first.",
            RuntimeWarning,
        )

    return raw
