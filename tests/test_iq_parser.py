import numpy as np
import torch
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from pipeline.iq_parser import RANDataStreamer

def generate_mock_iq_data(filepath, n_samples=10000):
    data = np.random.randn(n_samples * 2).astype(np.float32)
    data[100], data[101] = np.nan, np.inf
    data.tofile(filepath)

def test_iq_parser():
    mock_file = "tests/mock_iq.bin"
    generate_mock_iq_data(mock_file)
    
    try:
        streamer = RANDataStreamer(mock_file, num_gnbs=2)
        batch = streamer.get_next_batch()
        assert len(batch) == 2
        assert not torch.isnan(batch["gnb_0"]).any()
        print("RANDataStreamer signal processing test successful.")
    finally:
        if os.path.exists(mock_file):
            os.remove(mock_file)

if __name__ == "__main__":
    test_iq_parser()
