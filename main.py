# TODO: read pickle file
# TODO:
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def dump_data():
    experiment_data = pd.read_pickle("./data/haptic_geometry_dataset.pickle")
    rokubimini = experiment_data['rokubimini']
    filehandler = open("./data/rokubimini.pickle", 'wb')
    pickle.dump(rokubimini, filehandler)
    filehandler.close()
    print("dumped roku")


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

    # for sig,lab in zip(signals_train,labels_train):
    #     df = pd.DataFrame(sig, columns=['fx', 'fy', 'fz', 'tx', 'ty', 'tz'])
    #     df.plot()
    #     plt.title(lab)
    #     plt.show()
    #
    return roku





def main():
    #dump_data()
    roku = read_roku()
    X_train,y_train, X_test,y_test = preprocess_data(roku)
    # pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
    # pipe.fit(X_train, y_train)
    #
    # print(pipe.score(X_test, y_test))
    clf = RandomForestClassifier(max_depth=100, random_state=69)
    clf.fit(X_train, y_train)

    print(clf.score(X_test,y_test))

if __name__ == "__main__":
    main()