import mne

def epoch_and_average(
    raw: mne.io.Raw,
    tmin: float = -0.2,
    tmax: float = 0.8,
    reject: dict[str, float] | None = None,
):
    """Extract target and standard epochs and compute their average responses."""
    events, event_id = mne.events_from_annotations(raw)

    target_matches = [
        key for key in event_id if "oddball" in key.lower() or "target" in key.lower()
    ]
    standard_matches = [key for key in event_id if "standard" in key.lower()]
    if not target_matches or not standard_matches:
        raise ValueError(
            "Could not identify target and standard events. "
            f"Available event labels: {sorted(event_id)}"
        )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=(tmin, 0.0),
        reject=reject,
        preload=True
    )

    target_key = target_matches[0]
    standard_key = standard_matches[0]
    target_epochs = epochs[target_key]
    standard_epochs = epochs[standard_key]
    if len(target_epochs) == 0 or len(standard_epochs) == 0:
        raise RuntimeError(
            "No usable epochs remain for target or standard events. "
            "Try a longer recording window or pass a less strict reject threshold. "
            f"Target dropped: {len(epochs[target_key].drop_log)}, "
            f"standard dropped: {len(epochs[standard_key].drop_log)}."
        )

    evoked_target = target_epochs.average()
    evoked_standard = standard_epochs.average()

    return evoked_target, evoked_standard