import numpy as np
from torch.utils.data.dataset import Dataset
import torch
from utils import matrix_to_pixel_frame_target
from torch import nn


class MyDataset(Dataset):
    def __init__(self, dataset):

        self.__dataset = dataset

    def __getitem__(self, index):
        data = self.__dataset[index]

        pixel_id = np.array(data[0], dtype=int)
        frame_id = np.array(data[1], dtype=int)
        target = np.array(data[2], dtype=np.float32)

        return torch.from_numpy(pixel_id), torch.from_numpy(frame_id), torch.from_numpy(target)

    def __len__(self):
        self.__size = self.__dataset.shape[0]
        return self.__size


def load_dataset(matrix2d, batch_size, num_workers, train_test_split=None, valve=None):
    print('getting ind matrix')
    ind_mat = matrix_to_pixel_frame_target(matrix2d)
    dataset = MyDataset(ind_mat)
    if not train_test_split and not valve:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                             drop_last=False)
        loader = (loader, None)
    elif train_test_split:
        tot_num_samples = len(dataset)
        n_train = int(tot_num_samples*train_test_split)
        n_val = int(tot_num_samples*(1-train_test_split))
        train_set, val_set = torch.utils.data.random_split(dataset, (n_train, n_val))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers,
                                                   drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers,
                                                 drop_last=False)
        loader = (train_loader, val_loader)
    else:
        valve_frames = [int(list(v.keys())[0])-1 for v in valve]
        ind_mat_train = ind_mat[ind_mat[:, 1] < (valve_frames[-1]+valve_frames[-2])/2]
        ind_mat_val = ind_mat[ind_mat[:, 1] >= (valve_frames[-1] + valve_frames[-2]) / 2]
        dataset_train = MyDataset(ind_mat_train)
        dataset_val = MyDataset(ind_mat_val)
        loader = (torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              drop_last=False),
                  torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers,
                                              drop_last=False))
    return loader


class Swish(nn.Module):
    '''
    Implementation of swish.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - beta - trainable parameter
    '''

    def __init__(self, beta=None):
        '''
        Initialization.
        INPUT:
            - beta: trainable parameter
            beta is initialized with zero value by default
        '''
        super(Swish, self).__init__()

        # initialize beta
        if not beta:
            beta = 0.

        beta_tensor = torch.tensor(beta**2, dtype=torch.float, requires_grad=True)
        self.beta = torch.nn.Parameter(beta_tensor**2, requires_grad=True)  # create a tensor out of beta

    def forward(self, input):
        return input*torch.sigmoid(self.beta*input)


class EarlyStopping:
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)