import os
from pathlib import Path

import numpy as np

from tqdm import tqdm as tq

import torch

__all__ = ['DVS128DataSet']

class DVS128DataSet(torch.utils.data.Dataset):
    def __init__(self, data_dir, cnn_win, tcn_win, window_stride, verbose=False, include_subjects=None, single_out : bool = False, cnn_stride=None, file_suffix : str = "", transform=None):
        self.cnn_win = cnn_win
        self.tcn_win = tcn_win
        self.window_stride = window_stride
        if cnn_stride is None:
            cnn_stride = cnn_win

        self.cnn_stride = cnn_stride
        self.classes = [c for c in range(1, 12)]
        self.verbose = verbose
        self.transform = transform
        self.single_out = single_out
        cd = Path(__file__).parent.resolve()
        prep_file_pre = os.path.join(data_dir, "dataset")
        # dataset is always the same if we leave out the same subjects - not so
        # nice that the `file_suffix` is not calculated locally but whatever
        prepared_data_file = prep_file_pre+file_suffix+"_data.npy"
        prep_file_pre = prep_file_pre + "_cnn{}_tcn{}_stride{}".format(self.cnn_win, self.tcn_win, self.window_stride)
        if self.cnn_stride != cnn_win:
            prep_file_pre += "_cnn_stride{}".format(self.cnn_stride)
        prepared_index_file = prep_file_pre+file_suffix+"_indices.npy"
        prepared_label_file = prep_file_pre+file_suffix
        if not single_out:
            prepared_label_file += "_multi"
        else:
            prepared_label_file += "_single"
        prepared_label_file += "_labels.npy"

        if os.path.exists(prepared_data_file):
            if verbose:
                print("Found prepared dataset at {}".format(prepared_data_file))
            self.dat = np.load(prepared_data_file)
            load_data = False
        else:
            load_data = True
        if os.path.exists(prepared_index_file) and os.path.exists(prepared_label_file) and not load_data:
            self.indices = np.load(prepared_index_file)
            self.labels = np.load(prepared_label_file)
            return
        else:
            print("No prepared indices found, generating...")
        if load_data:
            self.dat = []
        self.indices = []
        self.labels = []
        if verbose:
            print("Loading Data.. ")
        curr_len = 0
        for i, cl in enumerate(self.classes):
            data, idxs, len = self.get_data_from_folder(os.path.join(data_dir, "class_{}".format(cl)), incl_subjects=include_subjects, ld_data=load_data)
            if self.single_out:
                curr_labels = np.full((idxs.shape[0]), cl-1, dtype=int)
            else:
                curr_labels = np.full((idxs.shape[0], self.tcn_win), cl-1, dtype=int)
            self.indices.append(idxs+curr_len)
            curr_len += len
            if load_data:
                self.dat.append(data)
            self.labels.append(curr_labels)
        if load_data:
            self.dat = np.concatenate(self.dat, axis=0)
        self.indices = np.concatenate(self.indices, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        if load_data:
            if verbose:
                print("Saving data at {}".format(prepared_data_file))
            np.save(prepared_data_file, self.dat)
        if verbose:
            print("Saving labels at {}".format(prepared_label_file))
        np.save(prepared_label_file, self.labels)
        if verbose:
            print("Saving indices at {}".format(prepared_index_file))
        np.save(prepared_index_file, self.indices)
        print("Dataset generation complete.")

    def get_data_from_folder(self, folder, incl_subjects=None, ld_data=True):
        np_files = [os.path.join(folder, f) for f in  os.listdir(folder) if f.endswith(".npy")]
        data = None
        if incl_subjects is not None:
            incl_user_names = ["user{:02d}".format(i) for i in incl_subjects]
            np_files = [f for f in np_files if os.path.basename(f)[0:6] in incl_user_names]
        if self.verbose:
            print("Loading {} files from folder: {}".format(len(np_files), folder))
            np_file_it = tq(np_files)
        else:
            np_file_it = np_files

        frames_tot = 0
        for i, f in enumerate(np_file_it):
            d = np.load(f)
            n_frames = d.shape[0]
            # if not self.overlap_cnn_wins:
            #     n_windows = int(np.ceil((n_frames-self.window_size+1)/self.window_stride))
            #     in_idxs = np.repeat(np.arange(self.window_size, dtype=np.int)[None, :], n_windows, axis=0) + (np.repeat(np.arange(n_windows, dtype=np.int)[:, None], self.window_size, axis=1) * self.window_stride)
            # else:
            n_windows = int(np.ceil((n_frames-self.unique_frames_in_window+1)/self.window_stride))
            indices_single_stack = np.concatenate([np.arange(i*self.cnn_stride, i*self.cnn_stride+self.cnn_win) for i in range(self.tcn_win)])
            in_idxs = np.repeat(indices_single_stack[None, :], n_windows, axis=0) + (np.repeat(np.arange(n_windows, dtype=np.int)[:, None], self.window_size, axis=1) * self.window_stride)
            if i == 0:
                idxs = in_idxs
                if ld_data:
                    data = d
            else:
                idxs = np.concatenate((idxs, in_idxs+frames_tot), axis=0)
                if ld_data:
                    data = np.concatenate((data, d), axis=0)
            frames_tot += n_frames

        if self.verbose:
            print("Processed folder {}!".format(folder))
        return data, idxs, frames_tot

    @property
    def window_size(self):
        return self.tcn_win*self.cnn_win

    @property
    def unique_frames_in_window(self):
        return self.cnn_win + (self.tcn_win-1)*self.cnn_stride

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, key):
        #print("Key: ", key)
        #print("Indices: ", self.indices[key])
        if self.transform:
            return (torch.tensor(self.transform(self.dat[self.indices[key]])).float(), torch.tensor(self.labels[key].squeeze()).long())
        else:
            return (torch.tensor(self.dat[self.indices[key]]).float(), torch.tensor(self.labels[key].squeeze()).long())
