import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.fft import fft, fftfreq
from sklearn.metrics import RocCurveDisplay, auc, confusion_matrix, ConfusionMatrixDisplay

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
    x = data['trials']['EO/class_1_A']['features']['avg_x']
    y = data['trials']['EO/class_1_A']['features']['avg_y']
    axs[0].scatter(x, y, c='green', marker='^', label='HAHV')

    x = data['trials']['EO/class_2_A']['features']['avg_x']
    y = data['trials']['EO/class_2_A']['features']['avg_y']
    axs[0].scatter(x, y, c='cyan', marker='>', label='LAHV')

    x = data['trials']['EO/class_3_A']['features']['avg_x']
    y = data['trials']['EO/class_3_A']['features']['avg_y']
    axs[0].scatter(x, y, c='blue', marker='v', label='LALV')

    x = data['trials']['EO/class_4_A']['features']['avg_x']
    y = data['trials']['EO/class_4_A']['features']['avg_y']
    axs[0].scatter(x, y, c='orange', marker='<', label='HALV')

    axs[0].legend(loc='upper left')
    axs[0].set_xlim(-0.5, 0.5)
    axs[0].set_ylim(-0.5, 0.5)
    axs[0].set_xlabel('Valence')
    axs[0].set_ylabel('Arousal')
    axs[0].set_title("Valence-Arousal annotations for A trials - participant " + data['participant'])
    axs[0].grid(True)

    # scatter plot for B trials
    x = data['trials']['EO/class_1_B']['features']['avg_x']
    y = data['trials']['EO/class_1_B']['features']['avg_y']
    axs[1].scatter(x, y, c='green', marker='^', label='HAHV')

    x = data['trials']['EO/class_2_B']['features']['avg_x']
    y = data['trials']['EO/class_2_B']['features']['avg_y']
    axs[1].scatter(x, y, c='cyan', marker='>', label='LAHV')

    x = data['trials']['EO/class_3_B']['features']['avg_x']
    y = data['trials']['EO/class_3_B']['features']['avg_y']
    axs[1].scatter(x, y, c='blue', marker='v', label='LALV')

    x = data['trials']['EO/class_4_B']['features']['avg_x']
    y = data['trials']['EO/class_4_B']['features']['avg_y']
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
        x.append(dataset[participant_id]['trials']['EO/class_1_A']['features']['avg_x'])
        y.append(dataset[participant_id]['trials']['EO/class_1_A']['features']['avg_y'])
    axs[0].scatter(x, y, c='green', marker='^', label='HAHV')

    x = []
    y = []
    for participant_id in dataset:
        x.append(dataset[participant_id]['trials']['EO/class_2_A']['features']['avg_x'])
        y.append(dataset[participant_id]['trials']['EO/class_2_A']['features']['avg_y'])
    axs[0].scatter(x, y, c='cyan', marker='>', label='LAHV')

    x = []
    y = []
    for participant_id in dataset:
        x.append(dataset[participant_id]['trials']['EO/class_3_A']['features']['avg_x'])
        y.append(dataset[participant_id]['trials']['EO/class_3_A']['features']['avg_y'])
    axs[0].scatter(x, y, c='blue', marker='v', label='LALV')

    x = []
    y = []
    for participant_id in dataset:
        x.append(dataset[participant_id]['trials']['EO/class_4_A']['features']['avg_x'])
        y.append(dataset[participant_id]['trials']['EO/class_4_A']['features']['avg_y'])
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
        x.append(dataset[participant_id]['trials']['EO/class_1_B']['features']['avg_x'])
        y.append(dataset[participant_id]['trials']['EO/class_1_B']['features']['avg_y'])
    axs[1].scatter(x, y, c='green', marker='^', label='HAHV')

    x = []
    y = []
    for participant_id in dataset:
        x.append(dataset[participant_id]['trials']['EO/class_2_B']['features']['avg_x'])
        y.append(dataset[participant_id]['trials']['EO/class_2_B']['features']['avg_y'])
    axs[1].scatter(x, y, c='cyan', marker='>', label='LAHV')

    x = []
    y = []
    for participant_id in dataset:
        x.append(dataset[participant_id]['trials']['EO/class_3_B']['features']['avg_x'])
        y.append(dataset[participant_id]['trials']['EO/class_3_B']['features']['avg_y'])
    axs[1].scatter(x, y, c='blue', marker='v', label='LALV')

    x = []
    y = []
    for participant_id in dataset:
        x.append(dataset[participant_id]['trials']['EO/class_4_B']['features']['avg_x'])
        y.append(dataset[participant_id]['trials']['EO/class_4_B']['features']['avg_y'])
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


def plot_roc_curve(fpr, tpr, roc_auc=0, label=""):
    plt.plot(fpr, tpr, color='orange', label='ROC curve %.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(label + ' Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


def plot_kfold_roc_curve(classifier, X, y, cv, pos_label, title=''):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X[test],
            y[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
            pos_label=pos_label
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic - " + title,
    )
    ax.legend(loc="lower right")
    plt.show()


def plot_labels_distribution(participant, arousal_labels, valence_labels, va_labels):
    plt.rcParams["figure.figsize"] = [15, 4]
    fig, axs = plt.subplots(nrows=1, ncols=3)

    n, bins, patches = axs[0].hist(arousal_labels, color='orange')
    axs[0].set_title("Distribution of labels for Arousal "  + participant)
    axs[0].set_xlabel('Classes')
    axs[0].set_ylabel('Samples')

    n, bins, patches = axs[1].hist(valence_labels, color='green')
    axs[1].set_title("Distribution of labels for Valence " + participant)
    axs[1].set_xlabel('Classes')
    axs[1].set_ylabel('Samples')

    n, bins, patches = axs[2].hist(va_labels, color='red')
    axs[2].set_title("Distribution of labels for VA" + participant)
    axs[2].set_xlabel('Classes')
    axs[2].set_ylabel('Samples')
    plt.show()


def plot_confusion_matrix_for_classifier(clf, X_test, y_test, labels):
    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=labels)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions,  labels=labels).ravel()
    print("TN",tn, "FP", fp, "FN", fn, "TP", tp)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()


def plot_accuracy_vs_mcc(df, sort_criterion, color_acc, color_mcc, color_chance, title, path=""):
    # df.set_index('Participant', inplace=True)
    bar_width = 0.3
    df.sort_values(by=[sort_criterion], inplace=True, ascending=False)
    x = np.arange(len(df.index))
    y_val = 'Test Accuracy'
    ylim = (-0.5, 1)
    fig, ax = plt.subplots()
    ax.bar(x=x - bar_width, height=df["MCC"], align='center', alpha=0.5, width=bar_width, capsize=10, color=color_mcc,
           label='Test MCC')
    ax.bar(x=x, height=df["Test Accuracy"], align='center', alpha=0.5, width=bar_width, capsize=10, color=color_acc,
           label='Test Accuracy')
    ax.bar(x=x + bar_width, height=df["Chance Level"], align='center', alpha=0.6, width=bar_width, capsize=10,
           color=color_chance, label='Default Guessing')
    ax.legend()
    ax.set_ylabel(y_val)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Participant'])
    ax.set_ylim(ylim)
    ax.set_title(title + ', sorted by ' + sort_criterion)
    ax.yaxis.grid(True)
    # Save the figure and show
    plt.tight_layout()
    plt.xticks(rotation=75)
    plt.savefig(path)
    plt.show()


def plot_cv_accuracy_vs_mcc(df, sort_criterion, color_acc, color_mcc, color_chance, title, path=""):
    # df.set_index('Participant', inplace=True)
    bar_width = 0.3
    df.sort_values(by=[sort_criterion], inplace=True, ascending=False)
    x = np.arange(len(df.index))
    y_val = 'CV Accuracy'
    ylim = (-0.5, 1)
    fig, ax = plt.subplots()
    error_kw = dict(ecolor=color_mcc, lw=1, capsize=3, capthick=1)
    ax.bar(x=x - bar_width, height=df["CV MCC"], yerr=df["CV MCC Std"], error_kw=error_kw, align='center', alpha=0.5,
           width=bar_width, capsize=10, color=color_mcc, label='Test MCC')
    error_kw = dict(ecolor=color_acc, lw=1, capsize=3, capthick=1)
    ax.bar(x=x, height=df["Mean"], yerr=df["Std"], error_kw=error_kw, align='center', alpha=0.5, width=bar_width,
           capsize=10, color=color_acc, label='Test Accuracy')
    ax.bar(x=x + bar_width, height=df["Chance Level"], align='center', alpha=0.6, width=bar_width, capsize=10,
           color=color_chance, label='Default Guessing')
    ax.legend()
    ax.set_ylabel(y_val)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Participant'])
    ax.set_ylim(ylim)
    ax.set_title(title + ', sorted by ' + sort_criterion)
    ax.yaxis.grid(True)
    # Save the figure and show
    plt.tight_layout()
    plt.xticks(rotation=75)
    plt.savefig(path)
    plt.show()


def plot_experiment_recap(df, path=""):
    # df.set_index('Participant', inplace=True)
    # df.sort_values(by=[sort_criterion], inplace=True, ascending=False)
    x = np.arange(len(df.index))
    y_val = 'Test Accuracy'
    ylim = (-0.25, 1)
    fig, ax = plt.subplots()
    error_kw = dict(ecolor='red', lw=1, capsize=3, capthick=1)
    ax.bar(x=x - 0.2, height=df["Avg MCC"], align='center', yerr=df["Avg MCC Std"], error_kw=error_kw, alpha=0.5,
           width=0.2, capsize=10, color='red', label='Avg MCC')
    error_kw = dict(ecolor='blue', lw=1, capsize=3, capthick=1)
    ax.bar(x=x, height=df["Avg Test Accuracy"], yerr=df["Avg Test Accuracy Std"], error_kw=error_kw, align='center',
           alpha=0.5, width=0.2, capsize=10, color='blue', label='Avg Test Accuracy')
    error_kw = dict(ecolor='blue', lw=1, capsize=3, capthick=1)
    ax.bar(x=x + 0.2, height=df["Avg Chance Level"], yerr=df["Avg Chance Level Std"], error_kw=error_kw, align='center',
           alpha=0.3, width=0.2, capsize=10, color='blue', label='Default Guessing')
    ax.legend()
    ax.set_ylabel(y_val)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Classifiers'])
    ax.set_ylim(ylim)
    ax.set_title("Test Accuracy of optimized classifiers vs default guessing")
    ax.yaxis.grid(True)
    # Save the figure and show
    plt.tight_layout()
    plt.xticks(rotation=75)
    plt.savefig(path)
    plt.show()


def plot_cv_recap(df, path=""):
    # df.set_index('Participant', inplace=True)
    # df.sort_values(by=[sort_criterion], inplace=True, ascending=False)
    x = np.arange(len(df.index))
    y_val = 'CV Accuracy'
    ylim = (-0.25, 1)
    fig, ax = plt.subplots()
    error_kw = dict(ecolor='red', lw=1, capsize=3, capthick=1)
    ax.bar(x=x - 0.2, height=df["Avg CV MCC"], align='center', yerr=df["Avg CV MCC Std"], error_kw=error_kw, alpha=0.5,
           width=0.2, capsize=10, color='red', label='Avg CV MCC')
    error_kw = dict(ecolor='purple', lw=1, capsize=3, capthick=1)
    ax.bar(x=x, height=df["Avg CV Score"], yerr=df["Avg CV Std"], align='center', error_kw=error_kw, alpha=0.5,
           width=0.2, capsize=10, color='purple', label='Avg CV Accuracy')
    error_kw = dict(ecolor='blue', lw=1, capsize=3, capthick=1)
    ax.bar(x=x + 0.2, height=df["Avg Chance Level"], yerr=df["Avg Chance Level Std"], error_kw=error_kw, align='center',
           alpha=0.3, width=0.2, capsize=10, color='blue', label='Default Guessing')
    ax.legend()
    ax.set_ylabel(y_val)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Classifiers'])
    ax.set_ylim(ylim)
    ax.set_title("Cross-Validated Accuracy of optimized classifiers vs default guessing")
    ax.yaxis.grid(True)
    # Save the figure and show
    plt.tight_layout()
    plt.xticks(rotation=75)
    plt.savefig(path)
    plt.show()


