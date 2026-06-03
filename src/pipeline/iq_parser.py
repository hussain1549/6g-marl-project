import torch
import numpy as np
import os

class RANDataStreamer:
    """
    High-performance data streamer for raw wireless I/Q data.
    Segments continuous I/Q streams into time-frequency resource grids per gNB agent.
    """

    def __init__(self, file_path, num_gnbs=3, fft_size=128, frame_length=10, overlap=0):
        """
        Initialize the RANDataStreamer.

        Args:
            file_path (str): Path to the binary file containing raw complex I/Q data.
            num_gnbs (int): Number of base station agents (gNBs).
            fft_size (int): Size of the FFT for spectrum calculation.
            frame_length (int): Number of FFT segments per resource grid frame.
            overlap (int): Overlap between frames (in number of samples).
        """
        if not os.path.exists(file_path):
            self.complex_data = np.array([], dtype=np.complex64)
        else:
            raw_data = np.fromfile(file_path, dtype=np.float32)
            real = raw_data[0::2]
            imag = raw_data[1::2]
            min_len = min(len(real), len(imag))
            self.complex_data = (real[:min_len] + 1j * imag[:min_len]).astype(np.complex64)
        
        self.num_gnbs = num_gnbs
        self.fft_size = fft_size
        self.frame_length = frame_length
        self.overlap = overlap
        
        self.current_pos = 0
        self.samples_per_frame = fft_size * frame_length
        self.agent_names = [f"gnb_{i}" for i in range(num_gnbs)]

    def _process_frame(self, data_segment):
        """
        Calculates the magnitude spectrum and normalizes it.
        """
        # Reshape and FFT
        segments = data_segment.reshape(self.frame_length, self.fft_size)
        spectrum = np.fft.fft(segments, axis=-1)
        spectrum = np.fft.fftshift(spectrum, axes=-1)
        magnitude = np.abs(spectrum)
        
        tensor = torch.from_numpy(magnitude).float()
        
        # Safety checks: Clean NaN and Inf
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalization to [-1, 1]
        # Using a simple min-max or tanh-like normalization for proxy CSI
        # Here we'll use: 2 * (x - min) / (max - min) - 1, with safety for div by zero
        t_min = tensor.min()
        t_max = tensor.max()
        
        if t_max > t_min:
            normalized_tensor = 2 * (tensor - t_min) / (t_max - t_min) - 1
        else:
            normalized_tensor = torch.zeros_like(tensor)
            
        return normalized_tensor

    def get_next_batch(self, num_agents=None):
        """
        Yields a dictionary of normalized PyTorch tensors for each agent.
        
        Returns:
            dict: {agent_name: torch.Tensor}
        """
        if num_agents is None:
            num_agents = self.num_gnbs
            
        batch = {}
        
        for i in range(num_agents):
            agent_name = self.agent_names[i]
            
            # Check if we have enough data left
            if self.current_pos + self.samples_per_frame > len(self.complex_data):
                # Wrap around or stop? Let's wrap for simplicity in streamer
                self.current_pos = 0
                
            if len(self.complex_data) < self.samples_per_frame:
                # Not enough data at all, return zeros
                batch[agent_name] = torch.zeros((self.frame_length, self.fft_size))
                continue

            segment = self.complex_data[self.current_pos : self.current_pos + self.samples_per_frame]
            processed_frame = self._process_frame(segment)
            
            # Optimize for GPU layout (channels_last if 3D, but here 2D)
            # Ensure contiguous for standard performance
            batch[agent_name] = processed_frame.contiguous()
            
            # Advance position (simple non-overlapping for agents in this batch)
            self.current_pos += self.samples_per_frame
            
        return batch

