import mbt_pyspt as mbt
import numpy as np
import json
from mbt_pyspt.models.eegdata import EEGData
from mbt_pyspt.modules.preprocessingflow import PreprocessingFlow


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    path = 'data/JULFIS/RAW/2017-11-13_13-14-19-MBT_XpertMode-VPro12-JULFIS-Bas.json'
    eeg_data = load_data(path)
    channel_data = eeg_data['recording']['channelData']
    sampling_rate = eeg_data['header']['sampRate']
    channel_locations = eeg_data['header']['acquisitionLocation']
    eeg_data = EEGData(channel_data, sampling_rate, channel_locations)

    pp_flow = PreprocessingFlow()


    # Preprocesing pipeline
    #
    # 1. Load the data
    # 2. Split the data: 8 songs (1 minute) and 8 white noise (15 seconds)
    # 3. Preprocess the data
    #   a.Handle extreme high/low values (interpolate with median)
    #   b.Handle continuous losses
    #       I. Decide a Max time period of losses allowed (1 second)
    #       II. Interpolate continuous NaN values below the threshold
    #       III. Remove continuous NaN values above the threshold
    #   c. Remove the mean from each channel
    #   d. Compute notch filter on 50Hz and 100Hz
    #   e. Compute band-pass filter on range 2-40Hz
    #   f. Calculate the thresholds for the outliers on the whole signal
    #      and remove the outliers too
    #   g. Clean artifacts using Quality Index method (or another automated method)
    # 4. Compute power bands with the Welch method
    # 5. Split the signals into interesting power bands (alpha, theta and beta) and compute AI indexes, z-scores etc







