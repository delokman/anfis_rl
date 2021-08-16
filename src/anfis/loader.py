import os

import numpy as np
import torch.utils.data
from torch.utils.data import TensorDataset


def load_training_data(test_folder, data_folder='./data/', batch_size=64, num_workers=2, dtype=torch.float):
    delimiter = ','

    inputs = np.genfromtxt(os.path.join(data_folder, test_folder, 'inputs.csv'), delimiter=delimiter)
    outputs = np.genfromtxt(os.path.join(data_folder, test_folder, 'outputs.csv'), delimiter=delimiter)

    outputs = np.expand_dims(outputs, -1)

    inputs = torch.tensor(inputs, dtype=dtype)
    outputs = torch.tensor(outputs, dtype=dtype)

    dataset = torch.utils.data.DataLoader(TensorDataset(inputs, outputs), batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, persistent_workers=True)

    return dataset


if __name__ == '__main__':
    dataset = load_training_data('test1', data_folder='../data/')
    (x, y) = next(dataset)

    print(x, y)
