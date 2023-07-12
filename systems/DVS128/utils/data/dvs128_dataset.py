import os
from pathlib import Path

import numpy as np
import sparse

from tqdm import tqdm as tq

import torch
from quantlib.QTensor import QTensor

__all__ = ['DVS128DataSet']

class DVS128DataSet(torch.utils.data.Dataset):
    def __init__(self, data_dir, cnn_win, tcn_win, window_stride, verbose=False, include_subjects=None, single_out : bool = False, cnn_stride=None, file_suffix : str = "", transform=None, fps : int = 30, **kwargs):
        self.cnn_win = cnn_win
        self.tcn_win = tcn_win
        self.window_stride = window_stride
        if cnn_stride is None:
            cnn_stride = cnn_win
        if include_subjects is None:
            include_subjects = list(range(1, 30))
        self.cnn_stride = cnn_stride
        self.classes = [c for c in range(1, 12)]
        self.verbose = verbose
        self.transform = transform
        self.single_out = single_out
        data_dir = os.path.join(data_dir, f"{fps}FPS")
        # dataset is always the same if we leave out the same subjects - not so
        # nice that the `file_suffix` is not calculated locally but whatever
        prepared_data_file = os.path.join(data_dir, "data.npz")
        prep_file_pre = os.path.join(data_dir, "cnn{}_tcn{}_stride{}".format(self.cnn_win, self.tcn_win, self.window_stride))
        if self.cnn_stride != cnn_win:
            prep_file_pre += "_cnn_stride{}".format(self.cnn_stride)
        prepared_index_file = prep_file_pre+file_suffix+"_indices.npz"
        prepared_label_file = prep_file_pre+file_suffix
        if not single_out:
            prepared_label_file += "_multi"
        else:
            prepared_label_file += "_single"
        prepared_label_file += "_labels.npz"
        data_by_user = {}
        idxs_by_user = {}
        labels_by_user = {}
        if os.path.exists(prepared_data_file):
            if verbose:
                print("Found prepared dataset at {}".format(prepared_data_file))
            save_data = False
            all_data = np.load(prepared_data_file)

            for u in all_data.files:
                data_by_user[int(u)] = all_data[u]

            all_data.close()
        else:
            save_data = True
        if os.path.exists(prepared_index_file) and os.path.exists(prepared_label_file) and not save_data:
            all_idxs = np.load(prepared_index_file)
            for u in all_idxs.files:
                idxs_by_user[int(u)] = all_idxs[u]
            all_idxs.close()
            all_labels = np.load(prepared_label_file)
            for u in all_labels.files:
                labels_by_user[int(u)] = all_labels[u]
            all_labels.close()
            self.extract_data(data_by_user, idxs_by_user, labels_by_user, include_subjects)
            return
        else:
            print("No prepared indices found, generating...")

        if verbose:
            print("Loading Data.. ")

        for i, cl in enumerate(self.classes):

            cur_data_by_user = self.get_data_from_folder(os.path.join(data_dir, "class_{}".format(cl)), incl_subjects=None)
            n_labels = {u: v['idxs'].shape[0] for u, v in cur_data_by_user.items()}
            if self.single_out:
                cur_labels = {u: np.full((n_labels[u],), cl-1, dtype=int) for u in cur_data_by_user.keys()}
            else:
                cur_labels = {u: np.full((n_labels[u], self.tcn_win), cl-1, dtype=int) for u in cur_data_by_user.keys()}

            for u, v in cur_data_by_user.items():
                if u not in labels_by_user.keys():
                    data_by_user[u] = v['data']
                    labels_by_user[u] = cur_labels[u]
                    idxs_by_user[u] = v['idxs']
                else:
                    idxs_by_user[u] = np.concatenate((idxs_by_user[u], v['idxs'] + data_by_user[u].shape[0]), axis=0)
                    data_by_user[u] = np.concatenate((data_by_user[u], v['data']), axis=0)
                    labels_by_user[u] = np.concatenate((labels_by_user[u], cur_labels[u]), axis=0)

        if save_data:
            if verbose:
                print("Saving data at {}".format(prepared_data_file))
            np.savez_compressed(prepared_data_file, **{str(u): v for u,v in data_by_user.items()})
        if verbose:
            print("Saving labels at {}".format(prepared_label_file))

        np.savez(prepared_label_file,  **{str(u): v for u,v in labels_by_user.items()})
        if verbose:
            print("Saving indices at {}".format(prepared_index_file))
        np.savez(prepared_index_file, **{str(u): v for u,v in idxs_by_user.items()})

        self.extract_data(data_by_user, idxs_by_user, labels_by_user, include_subjects)
        print("Dataset generation complete.")

    def extract_data(self, data, idxs, labels, include_subjects):
        self.dat = np.concatenate([data[u] for u in include_subjects], axis=0).astype(np.float32)
        start_idxs = np.cumsum([0] + [int(data[u].shape[0]) for u in include_subjects[:-1]])
        self.indices = np.concatenate([idxs[u] + start_idxs[i] for i, u in enumerate(include_subjects)], axis=0)
        self.labels = np.concatenate([labels[u] for u in include_subjects], axis=0)



    def get_data_from_folder(self, folder, incl_subjects=None):
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

        data_by_user = {}
        for i, f in enumerate(np_file_it):
            d = np.load(f)
            cur_user = int(os.path.basename(f)[4:6])
            n_frames = d.shape[0]
            n_windows = int(np.ceil((n_frames-self.unique_frames_in_window+1)/self.window_stride))
            indices_single_stack = np.concatenate([np.arange(i*self.cnn_stride, i*self.cnn_stride+self.cnn_win) for i in range(self.tcn_win)])
            in_idxs = np.repeat(indices_single_stack[None, :], n_windows, axis=0) + (np.repeat(np.arange(n_windows, dtype=np.int)[:, None], self.window_size, axis=1) * self.window_stride)
            if cur_user not in data_by_user.keys():

                data_by_user[cur_user] = {'data': d, 'idxs': in_idxs}
            else:
                cd = data_by_user[cur_user]
                data_by_user[cur_user] = {'data':np.concatenate((cd['data'], d), axis=0),
                                          'idxs':np.concatenate((cd['idxs'], in_idxs + cd['data'].shape[0]), axis=0)}
            

        if self.verbose:
            print("Processed folder {}!".format(folder))
        return data_by_user

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
            return (QTensor(self.transform(self.dat[self.indices[key]]), eps=1.).float(), torch.tensor(self.labels[key].squeeze()).long())
        else:
            return (QTensor(self.dat[self.indices[key]], eps=1.).float(), torch.tensor(self.labels[key].squeeze()).long())
