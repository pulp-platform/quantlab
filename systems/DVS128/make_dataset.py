# USE THIS

import os
from dv import LegacyAedatFile
import numpy as np
import os
import itertools
import cv2
from pathlib import Path
import pandas
import matplotlib.pyplot as plt
import numba
from tqdm import tqdm as tq

# download the dataset from
# https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794
# into this folder
DATASET_DIR = './dvs128'
# point the ./data folder to this folder (with a symlink)
GEN_DATA_DIR = './dvs128/processed'
# change if you want :)
FPS = 30

def get_filenames():
    users = ["user{:02d}".format(i) for i in range(1, 30)]
    lightings = ["fluorescent", "natural", "fluorescent_led", "lab", "led"]
    def realdir(f):
        return os.path.join(DATASET_DIR, f)
    filenames = [(realdir(u+"_"+l+".aedat"), realdir(u+"_"+l+"_labels.csv")) for u,l in itertools.product(users, lightings)]
    return [f for f in filenames if Path(f[0]).exists()]

def tern_to_greyscale(img):
    return np.uint8((img*127)+127)

def make_frames(evts : list, fps : int, prefix : str, out_mode : str = "np"):
    frametime_usec = 1000000.0//fps
    Path(prefix).parent.mkdir(exist_ok=True, parents=True)
    if out_mode == "cv2":
        fcc = cv2.VideoWriter_fourcc((*"XVID"))
        writer = cv2.VideoWriter(prefix+".avi", fcc, float(fps), (128, 128), 0)
        out_file = prefix+".avi"
    curr_frame = np.zeros((128, 128), dtype=np.int8)

    frame_idx = 0
    for idx, e in enumerate(evts):
        if idx == 0:
            frame_start = e.timestamp
        if e.timestamp - frame_start > frametime_usec:
            if out_mode == "cv2":
                frm = tern_to_greyscale(curr_frame)
                writer.write(frm)

            elif frame_idx == 0:
                video = curr_frame[None, :, :]
                #print("Max of curr_frame: {}".format(np.max(curr_frame)))
                #print("Min of curr_frame: {}".format(np.min(curr_frame)))
            else:
                video = np.concatenate((video, curr_frame[None, :, :]))
            curr_frame = np.zeros((128, 128), dtype=np.int8)
            frame_start = e.timestamp
            frame_idx += 1
        curr_frame[e.y, e.x] = np.int8(e.polarity*2-1)
    if out_mode == "cv2":
        writer.release()
        return out_file
    else:
        try:
            np.save(prefix+".npy", video)
            print("Saving {}".format(prefix+".npy"))
        except:
            print("Saving video failed:")
            print("len(evts): {}".format(len(evts)))

def gen_data(in_files : tuple, out_prefix : str, fps : int, out_mode : str):
    anns = pandas.read_csv(in_files[1])
    out_files = []
    with LegacyAedatFile(in_files[0]) as f:
        evt_time = -1
        for cl, st, et in zip(anns["class"], anns["startTime_usec"], anns["endTime_usec"]):
            while evt_time < st:
                evt_time = next(f).timestamp
            valid_evts = []
            while evt_time < et:
                evt = next(f)
                evt_time = evt.timestamp
                valid_evts.append(evt)
            if len(valid_evts) == 0:
                print("No valid events - start time: {}, end time: {}".format(st, et))
                print("File: {}".format(in_files[0]))
                print("Class: {}".format(cl))
            wr_prefix = os.path.join(out_prefix, 'class_{}'.format(cl), os.path.basename(in_files[0][:-6]))
            out_files.append(make_frames(valid_evts, fps, wr_prefix, out_mode))
    return out_files


if __name__ == "__main__":
    files = get_filenames()
    for f in tq(files):
        _ = gen_data(f, GEN_DATA_DIR, FPS, "np")
