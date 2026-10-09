import csv
from collections import Counter
import numpy as np
import mne

EVENT_ID = {"standard": 1, "oddball": 2, "noise": 3}
LABELS = {
    "standard": "standard", "standard_with_reponse": "standard",
    "standard_with_response": "standard",
    "oddball": "oddball", "oddball_with_reponse": "oddball",
    "oddball_with_response": "oddball",
    "noise": "noise", "noise_with_reponse": "noise",
    "noise_with_response": "noise",
}


def events_from_tsv(raw, path, *, correct_only=False):
    """Read ds003061 value column, keeping stimuli and excluding responses/glitches."""
    events, counts = [], Counter()
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            label = row["value"].strip()
            counts[label] += 1
            condition = LABELS.get(label)
            if condition is None or row["trial_type"] != "stimulus":
                continue
            if correct_only and label not in (
                "standard", "oddball_with_reponse", "oddball_with_response", "noise"
            ):
                continue
            sample = int(round(float(row["onset"]) * raw.info["sfreq"]))
            events.append([sample + raw.first_samp, 0, EVENT_ID[condition]])
    if not events:
        raise ValueError("No recognized stimulus events in events.tsv.")
    events = np.array(sorted(events), dtype=int)
    if len(np.unique(events[:, 0])) != len(events):
        raise ValueError("Duplicate stimulus onsets require review.")
    return events, dict(counts)


def make_epochs(raw, events, tmin=-0.2, tmax=0.8, reject=None):
    if not tmin < 0 < tmax:
        raise ValueError("Epoch must contain baseline and post-stimulus data.")
    ids = {key: code for key, code in EVENT_ID.items() if code in events[:, 2]}
    epochs = mne.Epochs(raw, events, ids, tmin, tmax, baseline=(tmin, 0),
                        picks="eeg", reject=reject, reject_by_annotation=True,
                        preload=True, event_repeated="error")
    for condition in ("oddball", "standard"):
        if condition not in ids or len(epochs[condition]) == 0:
            raise ValueError(f"No retained {condition} trials; inspect events and drop log.")
    return epochs


def epoch_and_average(raw, tmin=-0.2, tmax=0.8, reject=None, *, events=None):
    if events is None:
        ann_events, ann_ids = mne.events_from_annotations(raw)
        mapping = {code: EVENT_ID[LABELS[label]] for label, code in ann_ids.items()
                   if label in LABELS}
        events = np.array([[sample, 0, mapping[code]]
                           for sample, _, code in ann_events if code in mapping], dtype=int)
        if not len(events):
            raise ValueError("Annotations lack condition labels. Pass events_from_tsv output.")
    epochs = make_epochs(raw, events, tmin, tmax, reject)
    return epochs["oddball"].average(), epochs["standard"].average()

