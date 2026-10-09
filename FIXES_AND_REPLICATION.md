# Preprocessing fixes and replication status

Audited repository revision: 42f551ac8140bb8bf7025f391bad06163f7b798b.
This revision updates preprocessing and plotting; exact paper replication remains incomplete.

## What was fixed

- Removed unconditional ICA component 0 deletion. Fit a reproducible, rank-aware extended Infomax decomposition; remove only indices explicitly selected after review.
- Preserve BIDS electrode coordinates instead of overwriting them with a template. Keep non-EEG channels out of filtering, reference, ICA and ERP averages.
- Default high-pass is 1 Hz as stated in the paper. Low-pass 40 Hz is an implementation choice, not a reported paper setting. The original 30 Hz cutoff cannot retain the paper's full 3–40 Hz analysis range.
- Read line frequency from metadata (50 Hz for sub-001/run-1). Skip notch when line frequency is outside the low-pass range.
- Read stimulus conditions directly from events.tsv's value column. trial_type contains stimulus/responses, not standard/oddball labels. Include both oddball and oddball_with_reponse by default; button presses, noise and glitches do not enter the target ERP. Noise has its own epochs.
- Optional --correct-only keeps responded-to targets and standards without a response; state this policy when comparing results.
- Keep -200 to +800 ms epochs and -200 to 0 ms baseline. Export retained counts and drop reasons. EEG rejection thresholds use volts internally.
- Preserve the input Raw; interpolate marked bad channels after ICA and re-reference after interpolation.
- Fix download resumption: an existing directory alone does not establish a completed download.

## Run a recording

Install requirements in your Python environment. Supply actual OpenNeuro EEG files, not Git-annex symlinks from a GitHub checkout:

```powershell
python -m pip install -r requirements.txt
python main.py --data "D:\data\ds003061" --subject 001 --run 1 --out results/sub001_run1
python -m unittest test_pipeline -v
```

First run exports a decomposition with no components removed. Inspect component maps, sources and spectra before supplying --exclude-ica. The dataset has no channels typed EOG; EXG channels are miscellaneous. Do not assume a particular EXG channel or frontal EEG channel is a validated eye reference.

```python
import mne
ica = mne.preprocessing.read_ica("results/sub001_run1/decomposition-ica.fif")
raw = mne.io.read_raw_fif("results/sub001_run1/clean-raw.fif", preload=True)
ica.plot_components()
ica.plot_sources(raw)
# Inspect candidate components using ica.plot_properties(raw, picks=[index]).
```

After review, rerun the same recording with documented --exclude-ica indices and --bad channel names. A final pipeline also needs visually reviewed BAD time annotations. This CLI does not yet ingest an external annotation file; add reviewed annotations to raw before preprocessing in main.py. No automatic ICLabel classification is claimed. The paper uses ICLabel >=90% non-brain probability plus visual confirmation.

Exported outputs: ERP at Fz/Cz/Pz/P1, oddball and standard topographies at 20/50/100/200/350/500 ms, trial/channel averaged Welch PSD, FIF data and JSON audit. Figures are generated only when actual EEG is supplied.

## Why the PI's P1 plot may differ

The actual sub-001/run-1 events.tsv contains 522 standards, 112 responded-to oddballs, only 1 oddball without a response, 112 noise stimuli, 113 button-press markers and 3 ignore markers. If the original annotation matching finds both target labels and chooses oddball first, its target ERP is based on just one trial instead of 113. This is a major concrete candidate for the discrepancy. Counts are before artifact rejection.

P1 could mean participant 1, the P1 electrode (which exists in this recording), or an early positive ERP component. The original plot uses Pz, not P1. Obtain the PI's graph and channel, participant/run, event policy, reference, filter settings, artifact decisions, epoch/baseline, polarity and units. Compare the same raw recording and retained trials first. Never tune filters or deletions just to make a curve resemble the paper.

## Figure-by-figure replication work still needed

| Paper figure | Current support | Remaining requirement |
| --- | --- | --- |
| 1: task schematic | Protocol available in dataset README | Draw a labeled protocol schematic |
| 2: participant amplitude/latency | Epochs and evokeds exported | Confirm electrode/ROI, peak window, participant selection and definition of amplitude |
| 3: group ERP | Individual ERP supported | Process matched recordings for all selected subjects; average subjects equally |
| 4–5: 20/50 ms group topography | Individual maps at these times supported | Confirm subject/run inclusion; compute group evoked with common channels/reference |
| 6: group PSD | Individual condition PSD supported | Confirm paper's segment, electrode and PSD estimator choices; average linear power across subjects before dB |
| 7: RF ROC | Not implemented | Establish valid condition labels and subject-separated evaluation before training |
| Supplementary/ERSP/ITC | Not implemented | Confirm wavelet cycles, frequency grid, padding, baseline method and permutation design |
| Entropy/coherence/PLV | Not implemented | Confirm definitions, estimator/window, trial aggregation and valid condition mapping |

The dataset README describes three identical auditory oddball runs. It does not establish that a run is meditation and another is cognition. Standard and oddball are stimulus categories, not evidence of separate mental states. A meditation classifier requires verified labels.

The article does not fully specify low-pass/notch settings, excluded channels/components, selected raw subject identifiers, ERP peak definition, all PSD settings or wavelet parameters. Exact numerical reproduction cannot be established from the prose alone. Its methods/results/discussion also use condition and figure references inconsistently. Keep missing choices documented rather than claiming exact replication.

## Sources

- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12417730/
- Original code: https://github.com/PR-06-des/EEG-Data-Analyisis-Practice
- Dataset protocol: https://github.com/OpenNeuroDatasets/ds003061/blob/master/README
- Event definitions: https://github.com/OpenNeuroDatasets/ds003061/blob/master/task-P300_events.json
- Participant 1 acquisition: https://github.com/OpenNeuroDatasets/ds003061/blob/master/sub-001/eeg/sub-001_task-P300_run-1_eeg.json
- MNE ICA reference: https://mne.tools/stable/generated/mne.preprocessing.ICA.html

Validation: synthetic tests check stimulus/response selection, responded-target inclusion, correct-only policy, reference rank, input preservation and absence of arbitrary ICA removal. Actual participant EEG and the PI's graph were not available during this audit; physiological output has not been verified.
