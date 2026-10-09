"""Regression tests using the public Figure 2 processed DAS sample."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import h5py
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = REPOSITORY_ROOT / "tests" / "data"
INPUT_FILE = TEST_DATA / "fig2_fk_filtered.h5"
REFERENCE_FILE = TEST_DATA / "fig2_spatial_median_reference.h5"
sys.path.insert(0, str(REPOSITORY_ROOT))

from spatial_median_denoiser import DASDenoiser, read_waveform  # noqa: E402


class Figure2RegressionTests(unittest.TestCase):
    def test_fixture_files_contain_finite_waveforms(self) -> None:
        for path in (INPUT_FILE, REFERENCE_FILE):
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file(), f"Missing test fixture: {path}")
                with h5py.File(path, "r") as h5:
                    self.assertIn("Acquisition/Raw[0]/RawData", h5)
                    waveform = h5["Acquisition/Raw[0]/RawData"]
                    self.assertEqual(waveform.ndim, 2)
                    self.assertGreater(waveform.size, 0)
                    self.assertTrue(np.isfinite(waveform[:]).all())

    def test_filter_reproduces_figure2_reference(self) -> None:
        input_data, sampling_frequency, _ = read_waveform(INPUT_FILE)
        reference, reference_frequency, _ = read_waveform(REFERENCE_FILE)

        self.assertAlmostEqual(sampling_frequency, reference_frequency, places=8)
        self.assertEqual(input_data.shape, reference.shape)

        reproduced = DASDenoiser(
            n_samples_stft=128,
            n_aperture_channels=50,
            threshold_multiplier=0.8,
            verbose=False,
        ).denoise(
            input_data,
            sampling_frequency=sampling_frequency,
            block_size=500,
        )

        relative_l2_error = np.linalg.norm(reproduced - reference) / max(
            np.linalg.norm(reference), np.finfo(float).tiny
        )
        self.assertLessEqual(
            relative_l2_error,
            1e-3,
            f"Reproduced output differs from the Figure 2 reference: "
            f"relative L2 error={relative_l2_error:.3g}",
        )


if __name__ == "__main__":
    unittest.main()
