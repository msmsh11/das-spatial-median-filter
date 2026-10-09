"""Spatial-median amplitude replacement for DAS STFTs.

This is the DASDenoiser algorithm supplied with the study. The median is
applied along the channel axis in the time-frequency amplitude cube; it is not
a median filter applied directly to the waveform samples.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import istft, stft


class DASDenoiser:
    """Replace selected STFT amplitudes by their spatial-channel median."""

    def __init__(
        self,
        n_samples_stft: int = 128,
        n_aperture_channels: int = 50,
        threshold_multiplier: float = 0.8,
        use_start_times: bool = False,
        verbose: bool = True,
    ) -> None:
        self.n_samples_stft = n_samples_stft
        self.n_aperture_channels = n_aperture_channels
        self.threshold_multiplier = threshold_multiplier
        self.use_start_times = use_start_times
        self.verbose = verbose

    @staticmethod
    def next_power_of_two(n: int) -> int:
        return 1 << (n - 1).bit_length()

    def denoise(
        self,
        data: np.ndarray,
        sampling_frequency: float,
        start_times: np.ndarray | None = None,
        block_size: int = 500,
    ) -> np.ndarray:
        n_channels, n_samples = data.shape
        if self.n_samples_stft >= n_samples:
            raise ValueError("STFT window length must be shorter than the time series")

        if self.n_aperture_channels >= n_channels:
            self.n_aperture_channels = max(1, n_channels // 4)

        nperseg = self.next_power_of_two(self.n_samples_stft)
        noverlap = nperseg // 2
        filtered_data = np.zeros_like(data)
        n_blocks = int(np.ceil(n_channels / block_size))
        if self.verbose:
            print(f"DAS denoising: {n_channels} channels x {n_samples} samples; "
                  f"{sampling_frequency:g} Hz; {n_blocks} block(s)")

        for block_idx in range(n_blocks):
            start_ch = block_idx * block_size
            end_ch = min((block_idx + 1) * block_size, n_channels)
            block_start_times = (
                start_times[start_ch:end_ch] if start_times is not None else None
            )
            filtered_data[start_ch:end_ch] = self._process_block(
                data[start_ch:end_ch], sampling_frequency, nperseg, noverlap,
                block_start_times,
            )
        return filtered_data

    def _process_block(
        self,
        block_data: np.ndarray,
        fs: float,
        nperseg: int,
        noverlap: int,
        start_times: np.ndarray | None,
    ) -> np.ndarray:
        n_ch, n_samp = block_data.shape
        stfts = [
            stft(
                block_data[i], fs=fs, window="hann", nperseg=nperseg,
                noverlap=noverlap, boundary="even", padded=True,
            )[2]
            for i in range(n_ch)
        ]
        stft_array = np.stack(stfts, axis=0)
        amplitudes = np.abs(stft_array)
        phases = np.angle(stft_array)
        median_per_freq = np.median(amplitudes, axis=0)
        threshold = np.median(median_per_freq) * self.threshold_multiplier
        mask = amplitudes > threshold

        if self.use_start_times and start_times is not None:
            if np.isscalar(start_times):
                start_times = np.full(n_ch, start_times)
            else:
                start_times = np.asarray(start_times)
            _, frame_times, _ = stft(
                block_data[0], fs=fs, window="hann", nperseg=nperseg,
                noverlap=noverlap, boundary="even", padded=True,
            )
            for i in range(n_ch):
                start_frame = np.searchsorted(frame_times, start_times[i], side="left")
                mask[i, :, :start_frame] = False

        filter_size = (2 * self.n_aperture_channels + 1, 1, 1)
        median_filtered_amp = median_filter(amplitudes, size=filter_size, mode="nearest")
        amplitudes[mask] = median_filtered_amp[mask]

        filtered_block = np.zeros_like(block_data)
        for i in range(n_ch):
            _, reconstructed = istft(
                amplitudes[i] * np.exp(1j * phases[i]), fs=fs, window="hann",
                nperseg=nperseg, noverlap=noverlap, boundary=True, time_axis=-1,
            )
            filtered_block[i] = reconstructed[:n_samp]
        return filtered_block


def read_waveform(path: Path) -> tuple[np.ndarray, float, dict]:
    with h5py.File(path, "r") as h5:
        dataset = h5["Acquisition/Raw[0]/RawData"]
        data = dataset[:].T
        attrs = dict(h5["Acquisition/Raw[0]"].attrs)
        fs = float(attrs.get("OutputDataRate", 50.0))
    return data, fs, attrs


def write_waveform(path: Path, data: np.ndarray, fs: float, attrs: dict,
                   aperture: int, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        group = h5.create_group("/Acquisition/Raw[0]")
        group.create_dataset("RawData", data=data.T)
        for key, value in attrs.items():
            group.attrs[key] = value
        group.attrs["OutputDataRate"] = fs
        group.attrs["ProcessingSteps"] = (
            str(attrs.get("ProcessingSteps", "")) + "->tf_denoise_spatial_median"
        )
        group.attrs["TFDenoiserEnabled"] = True
        group.attrs["TFDenoiser_STFT_Samples"] = 128
        group.attrs["TFDenoiser_Aperture_Channels"] = aperture
        group.attrs["TFDenoiser_Threshold_Multiplier"] = threshold
        h5["/Acquisition"].attrs["SpatialSamplingInterval"] = attrs.get(
            "EffectiveDx", np.nan
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5", type=Path, help="FK-filtered HDF5 waveform")
    parser.add_argument("output_h5", type=Path, help="Destination HDF5 waveform")
    parser.add_argument("--aperture", type=int, default=50,
                        help="Half-width in channels; release metadata records 50")
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    data, fs, attrs = read_waveform(args.input_h5)
    denoiser = DASDenoiser(
        n_samples_stft=128,
        n_aperture_channels=args.aperture,
        threshold_multiplier=args.threshold,
        verbose=True,
    )
    result = denoiser.denoise(data, sampling_frequency=fs, block_size=500)
    write_waveform(args.output_h5, result, fs, attrs, args.aperture, args.threshold)


if __name__ == "__main__":
    main()
