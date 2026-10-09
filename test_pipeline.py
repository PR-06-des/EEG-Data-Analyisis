import tempfile
import unittest
from pathlib import Path
import numpy as np
import mne
from src.analysis import events_from_tsv, make_epochs
from src.preprocessing import preprocess_eeg


class PipelineTest(unittest.TestCase):
    def test_stimulus_mapping_and_baseline(self):
        raw = mne.io.RawArray(np.zeros((3, 1800)),
                              mne.create_info(["Fz", "Cz", "Pz"], 100, "eeg"))
        text = ("onset\ttrial_type\tvalue\n"
                "2\tstimulus\tstandard\n"
                "4\tstimulus\toddball_with_reponse\n"
                "4.3\tresponses\tresponse\n"
                "6\tstimulus\toddball\n"
                "8\tstimulus\tstandard_with_reponse\n"
                "10\tstimulus\tcondition_5\n")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "events.tsv"
            path.write_text(text)
            events, _ = events_from_tsv(raw, path)
            self.assertEqual(events[:, 2].tolist(), [1, 2, 2, 1])
            epochs = make_epochs(raw, events)
            self.assertEqual(len(epochs["oddball"]), 2)
            correct, _ = events_from_tsv(raw, path, correct_only=True)
            self.assertEqual(correct[:, 2].tolist(), [1, 2])

    def test_no_unverified_component_removed_and_input_unchanged(self):
        rng = np.random.default_rng(10)
        raw = mne.io.RawArray(rng.normal(size=(4, 3000)) * 1e-6,
                             mne.create_info(["Fz", "Cz", "Pz", "P1"], 100, "eeg"))
        original = raw.get_data().copy()
        clean, ica = preprocess_eeg(raw, return_ica=True)
        self.assertEqual(ica.exclude, [])
        self.assertLessEqual(ica.n_components_, 3)
        np.testing.assert_array_equal(raw.get_data(), original)
        np.testing.assert_allclose(clean.get_data().mean(axis=0), 0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
