# TODO: read pickle file
# TODO:
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import colorsys

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import decomposition

import seaborn as sn


def dump_data():
    experiment_data = pd.read_pickle("./data/haptic_geometry_dataset.pickle")
    rokubimini = experiment_data['rokubimini']
    filehandler = open("./data/rokubimini.pickle", 'wb')
    pickle.dump(rokubimini, filehandler)
    filehandler.close()
    print("dumped roku")

def visualize_signal(signals, labels):
    for sig,lab in zip(signals,labels):
        df = pd.DataFrame(sig, columns=['fx', 'fy', 'fz', 'tx', 'ty', 'tz'])
        df.plot()
        plt.title(lab)
        plt.show()

def colored_labels(y):
    N = 9
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    distinct_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

    y_rgb = list()
    cnt = 0
    y_copy = []
    for i in range(len(y)):
        if i > 0 and y[i] not in y_copy:
            cnt += 1
        y_rgb.append(distinct_colors[cnt])
        y_copy.append(y[i])

    c = np.asarray(y_rgb) / np.max(y_rgb)
    return c

def pca(data: np.ndarray, labels: np.ndarray):
    X = data.copy()
    y = labels.copy()
    n_components = 3

    fig = plt.figure(1, figsize=(20, 13))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .9, 0.9], elev=50, azim=45)
    plt.cla()

    pca = decomposition.PCA(n_components=n_components)
    pca.fit(X)
    x = pca.transform(X)
    c = colored_labels(y)

    _, class_idxs, class_cnts = np.unique(labels, return_index=True, return_counts=True)
    for start, size in zip(class_idxs, class_cnts):
        stop = start + size
        colors = c[start:stop, :]
        ax.scatter(x[start:stop, 0], x[start:stop, 1], x[start:stop, 2], c=colors, cmap='Dark2', s=150, edgecolors='k',
                   label=labels[start])

    ax.legend(loc="best", title="Classes", fontsize="x-large", title_fontsize="x-large")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    ax.grid(True)
    plt.show()
    #plt.savefig('./images/pca.png', bbox_inches='tight')

def preprocess_data(data):
    # filter data
    signals_train = [sub['signal'] for sub in data['train_ds']]
    labels_train = [sub['label'] for sub in data['train_ds']]

    signals_val = [sub['signal'] for sub in data['val_ds']]
    labels_val = [sub['label'] for sub in data['val_ds']]

    # normalize data
    signals_train_norm = [((sig - sig.mean(0)) / sig.std(0)) for sig in signals_train]
    signals_val_norm = [((sig - sig.mean(0)) / sig.std(0)) for sig in signals_val]

    # remove outliners
    sig_train_len = [sig.shape[0] for sig in signals_train_norm]
    idx_out = [length_idx for length_idx, length in enumerate(sig_train_len) if (length > 600 and length < 1000)]
    signals_train_norm = [sig for sig_idx, sig in enumerate(signals_train_norm) if sig_idx in idx_out]
    labels_train = [label for label_idx, label in enumerate(labels_train) if label_idx in idx_out]

    sig_val_len = [sig.shape[0] for sig in signals_val_norm]
    idx_out = [length_idx for length_idx, length in enumerate(sig_val_len) if (length > 600 and length < 1000)]
    signals_val_norm = [sig for sig_idx, sig in enumerate(signals_val_norm) if sig_idx in idx_out]
    labels_val = [label for label_idx, label in enumerate(labels_val) if label_idx in idx_out]

    # pad every signal to 1000
    signals_train_padded = []
    for sig in signals_train_norm:
        sig = np.pad(sig, [(0, 1000-len(sig)), (0, 0)], mode='constant', constant_values=0)
        signals_train_padded.append(sig)

    signals_val_padded = []
    for sig in signals_val_norm:
        sig = np.pad(sig, [(0, 1000-len(sig)), (0, 0)], mode='constant', constant_values=0)
        signals_val_padded.append(sig)

    # flatten
    signals_train_flat = [arr.flatten() for arr in signals_train_padded]
    signals_val_flat = [arr.flatten() for arr in signals_val_padded]

    return signals_train_flat, labels_train, signals_val_flat, labels_val

def read_roku():
    roku = pd.read_pickle("./data/rokubimini.pickle")
    return roku

def create_models():
    models = []

    models.append(("SVC", SVC()))
    models.append(("KNeighborsClassifier", KNeighborsClassifier()))
    # models.append(("KNeighborsRegressor", KNeighborsRegressor()))
    models.append(("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, max_features=None)))
    models.append(("RandomForestRegressor",RandomForestRegressor(n_estimators=100)))

    return models

def run_experiments(dataset,models):
    X_train, y_train, X_test, y_test = dataset

    for name, model in models:
        print("processing ",name)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        # evaluate model
        conf_mat = confusion_matrix(y_test,y_pred)
        acc = accuracy_score(y_test, y_pred, normalize=True)

        df_cm = pd.DataFrame(conf_mat, index=[i for i in ['-1', '0', '1', '2', '3', '4', '5', '6', '7']],
                             columns=[i for i in ['-1', '0', '1', '2', '3', '4', '5', '6', '7']])
        plt.figure(figsize=(9, 7))
        sns_conf_mat = sn.heatmap(df_cm, annot=True)
        print(acc)
        sns_conf_mat.set_title(str(name)+" accuracy_score: " + str(acc))

        sns_conf_mat.figure.savefig(str(name)+".png")
        print("processed ",name)

def main():
    #dump_data()
    roku = read_roku()
    X_train,y_train, X_test,y_test = preprocess_data(roku)
    dataset = X_train, y_train, X_test, y_test
    models = create_models()
    run_experiments(dataset,models)

if __name__ == "__main__":
    main()