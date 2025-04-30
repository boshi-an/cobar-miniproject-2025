import numpy as np
import os

class DataSaver:
    def __init__(self, path):
        self.path = path
        self.file_index = 0  # Initialize file index

    def append(self, data_dict):
        self._save(data_dict)

    def _save(self, data):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        save_path = os.path.join(f"{self.path}", f"{self.file_index}.npz")  # Append file index to path
        with open(save_path, 'wb') as f:
            np.savez(f, **data)
        self.file_index += 1  # Increment file index

    def close(self):
        pass
