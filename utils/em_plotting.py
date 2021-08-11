import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq


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

