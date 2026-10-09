import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath, read_raw_bids
from src.analysis import events_from_tsv, make_epochs
from src.preprocessing import preprocess_eeg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="Local ds003061 BIDS root")
    p.add_argument("--subject", default="001")
    p.add_argument("--run", default="1")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--highpass", type=float, default=1.0)
    p.add_argument("--lowpass", type=float, default=40.0)
    p.add_argument("--bad", nargs="*", default=[])
    p.add_argument("--exclude-ica", nargs="*", type=int, default=[])
    p.add_argument("--reject-uv", type=float, default=None)
    p.add_argument("--correct-only", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    bids = BIDSPath(subject=a.subject, task="P300", run=a.run, datatype="eeg", root=a.data)
    raw = read_raw_bids(bids, verbose=False)
    event_path = a.data / f"sub-{a.subject}" / "eeg" / (
        f"sub-{a.subject}_task-P300_run-{a.run}_events.tsv")
    events, counts = events_from_tsv(raw, event_path, correct_only=a.correct_only)
    clean, ica = preprocess_eeg(raw, a.highpass, a.lowpass,
                               bad_channels=a.bad, ica_exclude=a.exclude_ica,
                               return_ica=True)
    ica.save(a.out / "decomposition-ica.fif", overwrite=True)
    clean.save(a.out / "clean-raw.fif", overwrite=True)
    reject = None if a.reject_uv is None else {"eeg": a.reject_uv * 1e-6}
    epochs = make_epochs(clean, events, reject=reject)
    epochs.save(a.out / "stimulus-epo.fif", overwrite=True)
    evokeds = {key: epochs[key].average() for key in ("standard", "oddball")}
    mne.write_evokeds(a.out / "conditions-ave.fif", list(evokeds.values()), overwrite=True)
    channels = [ch for ch in ("Fz", "Cz", "Pz", "P1") if ch in clean.ch_names]
    fig, axes = plt.subplots(len(channels), 1, figsize=(9, 3 * len(channels)), squeeze=False)
    for ax, ch in zip(axes[:, 0], channels):
        for condition, evoked in evokeds.items():
            ax.plot(evoked.times * 1000, evoked.get_data(picks=[ch])[0] * 1e6,
                    label=f"{condition} (n={evoked.nave})")
        ax.axvline(0, color="black", linewidth=.6)
        ax.axhline(0, color="black", linewidth=.6)
        ax.set(title=ch, xlabel="Time from stimulus (ms)", ylabel="Amplitude (µV)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(a.out / "erp.png", dpi=180)
    plt.close(fig)
    limit = max(np.abs(e.get_data()).max() for e in evokeds.values()) * 1e6
    for condition, evoked in evokeds.items():
        fig = evoked.plot_topomap(times=[.02, .05, .1, .2, .35, .5],
                                   vlim=(-limit, limit), show=False)
        fig.savefig(a.out / f"{condition}-topography.png", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for condition in evokeds:
        spectrum = epochs[condition].compute_psd(method="welch", fmin=1,
                    fmax=a.lowpass, n_fft=min(256, len(epochs.times)), picks="eeg")
        power, freqs = spectrum.get_data(return_freqs=True)
        ax.plot(freqs, 10 * np.log10(np.maximum(power.mean(axis=(0, 1)) * 1e12,
                                              np.finfo(float).tiny)), label=condition)
    ax.set(xlabel="Frequency (Hz)", ylabel="PSD (dB µV²/Hz)", title="Trial and channel mean PSD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.out / "psd.png", dpi=180)
    plt.close(fig)
    report = dict(subject=a.subject, run=a.run, highpass=a.highpass,
                  lowpass=a.lowpass, sfreq=clean.info["sfreq"],
                  line_freq=clean.info.get("line_freq"), bad_channels=a.bad,
                  excluded_components=ica.exclude, original_event_counts=counts,
                  correct_only=a.correct_only, reject_uv=a.reject_uv,
                  retained={key: len(epochs[key]) for key in epochs.event_id},
                  drop_log=[list(entry) for entry in epochs.drop_log],
                  mne_version=mne.__version__,
                  note="ICA components require visual review; no automatic ICLabel removal.")
    (a.out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved figures and audit to {a.out.resolve()}")


if __name__ == "__main__":
    main()

