import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.fft import fft, fftfreq

from preprocessing.em_preprocessing import participant_avg_annotation_windows, compute_avg_annotation_windows


def plot_annotations(data):
    plt.rcParams["figure.figsize"] = [15, 7.50]
    fig, axs = plt.subplots(nrows=1, ncols=2)

    # scatter plot for A trials
    x = data['trials']['EO/class_1_A']['annotations']['x']
    y = data['trials']['EO/class_1_A']['annotations']['y']
    axs[0].scatter(x, y, c='green', marker='^', label='HAHV')

    x = data['trials']['EO/class_2_A']['annotations']['x']
    y = data['trials']['EO/class_2_A']['annotations']['y']
    axs[0].scatter(x, y, c='cyan', marker='>', label='LAHV')

    x = data['trials']['EO/class_3_A']['annotations']['x']
    y = data['trials']['EO/class_3_A']['annotations']['y']
    axs[0].scatter(x, y, c='blue', marker='v', label='LALV')

    x = data['trials']['EO/class_4_A']['annotations']['x']
    y = data['trials']['EO/class_4_A']['annotations']['y']
    axs[0].scatter(x, y, c='orange', marker='<', label='HALV')

    axs[0].legend(loc='upper left')
    axs[0].set_xlim(-0.5, 0.5)
    axs[0].set_ylim(-0.5, 0.5)
    axs[0].set_xlabel('Valence')
    axs[0].set_ylabel('Arousal')
    axs[0].set_title("Valence-Arousal annotations for A trials - participant " + data['participant'])
    axs[0].grid(True)

    # scatter plot for B trials
    x = data['trials']['EO/class_1_B']['annotations']['x']
    y = data['trials']['EO/class_1_B']['annotations']['y']
    axs[1].scatter(x, y, c='green', marker='^', label='HAHV')

    x = data['trials']['EO/class_2_B']['annotations']['x']
    y = data['trials']['EO/class_2_B']['annotations']['y']
    axs[1].scatter(x, y, c='cyan', marker='>', label='LAHV')

    x = data['trials']['EO/class_3_B']['annotations']['x']
    y = data['trials']['EO/class_3_B']['annotations']['y']
    axs[1].scatter(x, y, c='blue', marker='v', label='LALV')

    x = data['trials']['EO/class_4_B']['annotations']['x']
    y = data['trials']['EO/class_4_B']['annotations']['y']
    axs[1].scatter(x, y, c='orange', marker='<', label='HALV')

    axs[1].legend(loc='upper left')
    axs[1].set_xlim(-0.5, 0.5)
    axs[1].set_ylim(-0.5, 0.5)
    axs[1].set_xlabel('Valence')
    axs[1].set_ylabel('Arousal')
    axs[1].set_title("Valence-Arousal annotations for B trials - participant " + data['participant'])
    axs[1].grid(True)

    fig.tight_layout()
    plt.show()


def plot_avg_annotations(data):
    plt.rcParams["figure.figsize"] = [15, 7.50]
    fig, axs = plt.subplots(nrows=1, ncols=2)

    # scatter plot for A trials
    x = data['trials']['EO/class_1_A']['annotations']['avg_x']
    y = data['trials']['EO/class_1_A']['annotations']['avg_y']
    axs[0].scatter(x, y, c='green', marker='^', label='HAHV')

    x = data['trials']['EO/class_2_A']['annotations']['avg_x']
    y = data['trials']['EO/class_2_A']['annotations']['avg_y']
    axs[0].scatter(x, y, c='cyan', marker='>', label='LAHV')

    x = data['trials']['EO/class_3_A']['annotations']['avg_x']
    y = data['trials']['EO/class_3_A']['annotations']['avg_y']
    axs[0].scatter(x, y, c='blue', marker='v', label='LALV')

    x = data['trials']['EO/class_4_A']['annotations']['avg_x']
    y = data['trials']['EO/class_4_A']['annotations']['avg_y']
    axs[0].scatter(x, y, c='orange', marker='<', label='HALV')

    axs[0].legend(loc='upper left')
    axs[0].set_xlim(-0.5, 0.5)
    axs[0].set_ylim(-0.5, 0.5)
    axs[0].set_xlabel('Valence')
    axs[0].set_ylabel('Arousal')
    axs[0].set_title("Valence-Arousal annotations for A trials - participant " + data['participant'])
    axs[0].grid(True)

    # scatter plot for B trials
    x = data['trials']['EO/class_1_B']['annotations']['avg_x']
    y = data['trials']['EO/class_1_B']['annotations']['avg_y']
    axs[1].scatter(x, y, c='green', marker='^', label='HAHV')

    x = data['trials']['EO/class_2_B']['annotations']['avg_x']
    y = data['trials']['EO/class_2_B']['annotations']['avg_y']
    axs[1].scatter(x, y, c='cyan', marker='>', label='LAHV')

    x = data['trials']['EO/class_3_B']['annotations']['avg_x']
    y = data['trials']['EO/class_3_B']['annotations']['avg_y']
    axs[1].scatter(x, y, c='blue', marker='v', label='LALV')

    x = data['trials']['EO/class_4_B']['annotations']['avg_x']
    y = data['trials']['EO/class_4_B']['annotations']['avg_y']
    axs[1].scatter(x, y, c='orange', marker='<', label='HALV')

    axs[1].legend(loc='upper left')
    axs[1].set_xlim(-0.5, 0.5)
    axs[1].set_ylim(-0.5, 0.5)
    axs[1].set_xlabel('Valence')
    axs[1].set_ylabel('Arousal')
    axs[1].set_title("Valence-Arousal annotations for B trials - participant " + data['participant'])
    axs[1].grid(True)

    fig.tight_layout()
    plt.show()


def plot_avg_annotations_all_participants(dataset):
    plt.rcParams["figure.figsize"] = [15, 7.50]
    fig, axs = plt.subplots(nrows=1, ncols=2)

    # Calculate average annotations for each participant
    n_windows = 1
    for participant_id in dataset:
        dataset[participant_id] = participant_avg_annotation_windows(dataset[participant_id], n_windows)

    # Scatter plot for average annotations of A trials for each participant
    x = []
    y = []
    for participant_id in dataset:
        x.append(dataset[participant_id]['trials']['EO/class_1_A']['annotations']['avg_x'])
        y.append(dataset[participant_id]['trials']['EO/class_1_A']['annotations']['avg_y'])
    axs[0].scatter(x, y, c='green', marker='^', label='HAHV')

    x = []
    y = []
    for participant_id in dataset:
        x.append(dataset[participant_id]['trials']['EO/class_2_A']['annotations']['avg_x'])
        y.append(dataset[participant_id]['trials']['EO/class_2_A']['annotations']['avg_y'])
    axs[0].scatter(x, y, c='cyan', marker='>', label='LAHV')

    x = []
    y = []
    for participant_id in dataset:
        x.append(dataset[participant_id]['trials']['EO/class_3_A']['annotations']['avg_x'])
        y.append(dataset[participant_id]['trials']['EO/class_3_A']['annotations']['avg_y'])
    axs[0].scatter(x, y, c='blue', marker='v', label='LALV')

    x = []
    y = []
    for participant_id in dataset:
        x.append(dataset[participant_id]['trials']['EO/class_4_A']['annotations']['avg_x'])
        y.append(dataset[participant_id]['trials']['EO/class_4_A']['annotations']['avg_y'])
    axs[0].scatter(x, y, c='orange', marker='<', label='HALV')

    axs[0].legend(loc='upper left')
    axs[0].set_xlim(-0.5, 0.5)
    axs[0].set_ylim(-0.5, 0.5)
    axs[0].set_xlabel('Valence')
    axs[0].set_ylabel('Arousal')
    axs[0].set_title("Valence-Arousal annotations for A trials - Average for each participant")
    axs[0].grid(True)



    # Scatter plot for average annotations of B trials for each participant
    x = []
    y = []
    for participant_id in dataset:
        x.append(dataset[participant_id]['trials']['EO/class_1_B']['annotations']['avg_x'])
        y.append(dataset[participant_id]['trials']['EO/class_1_B']['annotations']['avg_y'])
    axs[1].scatter(x, y, c='green', marker='^', label='HAHV')

    x = []
    y = []
    for participant_id in dataset:
        x.append(dataset[participant_id]['trials']['EO/class_2_B']['annotations']['avg_x'])
        y.append(dataset[participant_id]['trials']['EO/class_2_B']['annotations']['avg_y'])
    axs[1].scatter(x, y, c='cyan', marker='>', label='LAHV')

    x = []
    y = []
    for participant_id in dataset:
        x.append(dataset[participant_id]['trials']['EO/class_3_B']['annotations']['avg_x'])
        y.append(dataset[participant_id]['trials']['EO/class_3_B']['annotations']['avg_y'])
    axs[1].scatter(x, y, c='blue', marker='v', label='LALV')

    x = []
    y = []
    for participant_id in dataset:
        x.append(dataset[participant_id]['trials']['EO/class_4_B']['annotations']['avg_x'])
        y.append(dataset[participant_id]['trials']['EO/class_4_B']['annotations']['avg_y'])
    axs[1].scatter(x, y, c='orange', marker='<', label='HALV')

    axs[1].legend(loc='upper left')
    axs[1].set_xlim(-0.5, 0.5)
    axs[1].set_ylim(-0.5, 0.5)
    axs[1].set_xlabel('Valence')
    axs[1].set_ylabel('Arousal')
    axs[1].set_title("Valence-Arousal annotations for A trials - Average for each participant")
    axs[1].grid(True)

    fig.tight_layout()
    plt.show()


def plot_trial_annotations(x, y, trial='', animated=False):
    plt.rcParams["figure.figsize"] = [7.50, 7.50]
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.set_xlim(-0.5, 0.5)
    axs.set_ylim(-0.5, 0.5)
    axs.set_xlabel('Valence')
    axs.set_ylabel('Arousal')
    axs.set_title("Valence-Arousal averaged annotations for single trial " + trial)
    axs.grid(True)

    fig.tight_layout()
    axs.scatter(x, y, c='red', marker='x', label='Annotations')
    axs.legend(loc='upper left')
    plt.show()


def plot_signal(channel_data, duration, sampling_rate=250, raw_xlim=[0, None], raw_ylim=[-0.0001, 0.0001]):
    dt = 1 / sampling_rate  # sampling interval
    Fs = 1 / dt  # sampling frequency of 250Hz
    t = np.arange(0, duration)
    F4 = np.array(channel_data[0])
    F3 = np.array(channel_data[1])

    plt.rcParams["figure.figsize"] = [20, 5]
    fig, axs = plt.subplots(1, 1)
    axs.set_title("EEG Signal")
    axs.plot(t, F4, t, F3)
    axs.set_xlim(raw_xlim)
    axs.set_ylim(raw_ylim)
    axs.set_xlabel("Time")
    axs.set_ylabel("F4 and F3")
    axs.grid(True)
    fig.tight_layout()
    plt.show()


def plot_study(channel_data, duration, sampling_rate=250, raw_xlim=[0, None]):
    dt = 1 / sampling_rate  # sampling interval
    Fs = 1 / dt  # sampling frequency of 250Hz
    t = np.arange(0, duration)
    F4 = np.array(channel_data[0])
    F3 = np.array(channel_data[1])

    plt.rcParams["figure.figsize"] = [15, 20]
    fig, axs = plt.subplots(4, 1)
    axs[0].set_title("EEG Signal")
    axs[0].plot(t, F4, t, F3)
    axs[0].set_xlim(raw_xlim[0], raw_xlim[1])
    axs[0].set_ylim(-0.0001, 0.0001)
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("F4 and F3")
    axs[0].grid(True)

    cxy, f = axs[1].cohere(F4, F3, 125, Fs)
    axs[1].set_ylabel('coherence')

    N = int(duration)

    yfF4 = fft(F4)
    yfF3 = fft(F3)
    xf = fftfreq(N, 1/250)

    axs[2].plot(xf, np.abs(yfF3), color='blue')
    axs[2].set_xlim(0, 55)
    axs[2].set_ylabel('Amplitude')
    axs[2].set_xlabel('Frequency')
    axs[2].set_title('FFT F3')

    axs[3].plot(xf, np.abs(yfF4), color='orange' )
    axs[3].set_xlim(0, 55)
    axs[3].set_ylabel('Amplitude')
    axs[3].set_xlabel('Frequency')
    axs[3].set_title('FFT F4')

    fig.tight_layout()
    plt.show()


def plot_linear_regression(x_test, y_test, y_pred, d_label, i_label):
    plt.ylabel(d_label)
    plt.xlabel(i_label)
    plt.rcParams["figure.figsize"] = [15, 7.50]
    plt.scatter(x_test, y_test, color='black')
    plt.plot(x_test, y_pred, color='blue', linewidth=3)
    plt.title("Predicting " + d_label + " over " + i_label)

    plt.xticks()
    plt.yticks()

    plt.tight_layout()
    plt.show()