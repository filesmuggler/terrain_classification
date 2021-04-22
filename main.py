# TODO: read pickle file
# TODO:
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


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

    #models.append(("SVC", SVC()))
    #models.append(("KNeighbors", KNeighborsClassifier()))
    models.append(("RandomForest", RandomForestClassifier(n_estimators=100, criterion='gini',
                                max_depth=10, random_state=0, max_features=None)))

    return models

def run_experiments(dataset,models):
    X_train, y_train, X_test, y_test = dataset

    results = []
    names = []

    for name, model in models:
        names.append(name)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        # evaluate model
        conf_mat = confusion_matrix(y_test,y_pred)
        model_score = model.score(X_test,y_test)
        results.append((conf_mat,model_score))

    for i in range(len(names)):
        print(names[i], results[i])

def main():
    #dump_data()
    roku = read_roku()
    X_train,y_train, X_test,y_test = preprocess_data(roku)
    dataset = X_train, y_train, X_test, y_test
    models = create_models()
    run_experiments(dataset,models)

if __name__ == "__main__":
    main()