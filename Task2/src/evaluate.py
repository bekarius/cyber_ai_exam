from pathlib import Path
import numpy as np
from joblib import load
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def evaluate_model(model_path: Path, X, y):
    clf = load(model_path)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    return acc, cm

def save_confusion_matrix_plot(cm, class_names, outpath):
    import itertools
    plt.figure(figsize=(4.5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()
