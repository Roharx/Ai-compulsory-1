import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------
# Fix the random seed, so we get the same results:
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------------------------------------------------------
# 1) Load MNIST
# --------------------------------------------------------------------
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
# "mnist_784": each 28×28 image is “unrolled” into 784 features (28*28).
# as_frame=False gives a NumPy array instead of a pandas DataFrame.

X = mnist.data.astype(np.uint8)
# Casting to uint8 keeps pixel values as integers 0–255, saving memory.
y = mnist.target.astype(np.int64)
# Labels are cast to 64-bit integers for classification.

print("X shape:", X.shape, "y shape:", y.shape)

# --------------------------------------------------------------------
# 2) Stratified split (20% test)
# --------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
# Stratify=y makes sure we have the same amount of data for each label in y
# This is important for small data sets, maybe not that important for this one

# --- Quick run: downsample training to ~20k examples ---------------
# This is for testing only on my slower laptop
# quick_idx = np.random.choice(len(X_train), size=20000, replace=False)
# X_train_small = X_train[quick_idx]
# y_train_small = y_train[quick_idx]
# print("Quick-run training set:", X_train_small.shape)

# --------------------------------------------------------------------
# 3) Random Forest + fast GridSearch
# --------------------------------------------------------------------
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
param_grid = {
    "n_estimators": [250, 500],
    "max_depth": [20, 40],
    "max_features": ["sqrt"],
}
# n_estimators: number of forests, the input amount is the number of trees
# max_depth: [20, 40] compares the 20 and 40 together, then picks the more accurate one
# max_features: ["sqrt"]: tells GridSearchCV to test forests where each tree node considers about 28 (√784) randomly
# chosen pixel features when deciding its split.

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="accuracy",
    cv=2,
    n_jobs=-1, # all CPU cores
    verbose=1
)
# scoring="accuracy": Metric used to evaluate each model on the validation folds: proportion of correct predictions

# cv=2: Number of cross-validation folds: the training set is split into 2 roughly equal parts.
# For each parameter combo:
#   Train on fold 1, validate on fold 2.
#   Train on fold 2, validate on fold 1.
# Mean of the 2 validation scores is the CV score.

print("\nRunning quick GridSearchCV…")
grid.fit(X_train, y_train)

print("\nBest params:", grid.best_params_)
print("Best CV accuracy: {:.4f}".format(grid.best_score_))

best_rf = grid.best_estimator_

# --------------------------------------------------------------------
# 4) Evaluate on held-out
# --------------------------------------------------------------------
y_pred = best_rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro")
rec = recall_score(y_test, y_pred, average="macro")

print("\n=== Test Set Performance (Quick RF) ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall    (macro): {rec:.4f}")

print("\nPer-class report:")
print(classification_report(y_test, y_pred, digits=4))

# --------------------------------------------------------------------
# 5) Confusion-matrix plots
# --------------------------------------------------------------------
# This is just for visualization, not a requirement. I've generated it via ChatGPT
def plot_confusion_matrices(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = ["Confusion Matrix (counts)", "Confusion Matrix (row-normalized)"]
    matrices = [cm, cm_norm]
    cmaps = ["Blues", "Greens"]

    for ax, mat, title, cmap in zip(axes, matrices, titles, cmaps):
        im = ax.imshow(mat, interpolation="nearest", cmap=cmap)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ticks = np.arange(len(labels))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        fmt = ".2f" if title.endswith("normalized)") else "d"
        thresh = mat.max() / 2.0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, format(mat[i, j], fmt),
                        ha="center", va="center",
                        color="white" if mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)

plot_confusion_matrices(y_test, y_pred, labels=list(range(10)))
