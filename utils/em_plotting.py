import matplotlib.pyplot as plt


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

