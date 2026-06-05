import numpy as np
import torch
import os

class RANDataStreamer:
    def __init__(self, filepath, num_gnbs, grid_size=(14, 12), fft_size=64):
        self.data = np.fromfile(filepath, dtype=np.float32)
        self.num_gnbs, self.grid_size, self.fft_size = num_gnbs, grid_size, fft_size
        self.total_samples = len(self.data) // 2
        self.current_ptr, self.has_wrapped = 0, False

    def _get_complex_samples(self, n_samples):
        if self.current_ptr + n_samples > self.total_samples:
            self.current_ptr, self.has_wrapped = 0, True
        start, end = self.current_ptr * 2, (self.current_ptr + n_samples) * 2
        raw = np.nan_to_num(self.data[start:end], nan=0.0, posinf=1.0, neginf=-1.0)
        complex_chunk = raw[0::2] + 1j * raw[1::2]
        self.current_ptr += n_samples
        return complex_chunk

    def process_fft(self, samples):
        n_ffts = len(samples) // self.fft_size
        if n_ffts == 0: return np.zeros((1, self.fft_size))
        samples = samples[:n_ffts * self.fft_size].reshape(n_ffts, self.fft_size)
        return np.abs(np.fft.fft(samples, axis=1))

    def normalize_and_clean(self, data):
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
        max_val = np.max(np.abs(data))
        if max_val > 0: data /= max_val
        return np.clip(data, -1.0, 1.0)

    def get_next_batch(self, num_agents=None):
        self.has_wrapped = False
        num_agents = num_agents or self.num_gnbs
        batch = {}
        time_steps, subcarriers = self.grid_size
        for i in range(num_agents):
            raw = self._get_complex_samples(time_steps * self.fft_size)
            grid = self.normalize_and_clean(self.process_fft(raw)[:time_steps, :subcarriers])
            batch[f"gnb_{i}"] = torch.from_numpy(grid.astype(np.float32)).contiguous()
        return batch
