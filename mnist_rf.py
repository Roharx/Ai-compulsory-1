from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from pathlib import Path

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# 1) Load MNIST dataset: 70,000 images of size 28x28 flattened into 784 columns
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(np.int64)

print("MNIST loaded:", X.shape, y.shape)

# 2) Split into Train+Validation and Test
# X_test, y_test → final test set of 10,000 examples
# X_totaltrain, y_totaltrain → remaining 60,000 examples for training + validation
X_totaltrain, X_test, y_totaltrain, y_test = train_test_split(
    X, y, test_size=10000, stratify=y, random_state=RANDOM_STATE
)

# Split total train further into Train and Validation
X_train, X_validation, y_train, y_validation = train_test_split(
    X_totaltrain, y_totaltrain, test_size=10000, stratify=y_totaltrain, random_state=RANDOM_STATE
)

# 3) Baseline Random Forest
randomforest_baseline = RandomForestClassifier(
    n_estimators=800,   # number of trees to build (hyperparameter)
    max_depth=20,       # maximum depth of each tree
    n_jobs=-1,          # use all CPU cores
    random_state=RANDOM_STATE
)
randomforest_baseline.fit(X_train, y_train)
validation_predict_base = randomforest_baseline.predict(X_validation)
print("Baseline RF - Validation accuracy:", accuracy_score(y_validation, validation_predict_base))

# 4) Manual fine-tuning
candidates_n_estimators = [500, 800, 1000]  # candidate values for number of trees
candidates_max_depth   = [10, 20, 30, 40]   # candidate values for tree depth

best_accuracy = -1.0  # best accuracy found so far
best_configuration = None  # stores the best (n_estimators, max_depth)

for n in candidates_n_estimators:
    for d in candidates_max_depth:
        randomforest_try = RandomForestClassifier(
            n_estimators=n, max_depth=d, n_jobs=-1, random_state=RANDOM_STATE
        )
        randomforest_try.fit(X_train, y_train)  # train on training set
        acc = accuracy_score(y_validation, randomforest_try.predict(X_validation))  # evaluate on validation
        print(f"Validation accuracy @ n_estimators={n}, max_depth={d}: {acc:.4f}")
        if acc > best_accuracy:  # update best config if current model is better
            best_accuracy = acc
            best_configuration = (n, d)

print("\nBest RF config on validation:", best_configuration, "val_acc=", best_accuracy)

# 5) Retrain final model on Train + Validation (60,000 images instead of 50,000)
n_best, d_best = best_configuration
randomforest_final = RandomForestClassifier(
    n_estimators=n_best, max_depth=d_best, n_jobs=-1, random_state=RANDOM_STATE
)
randomforest_final.fit(
    np.vstack([X_train, X_validation]),  # combine Train and Validation sets
    np.hstack([y_train, y_validation])
)

test_pred = randomforest_final.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
print("\nRandom Forest — Test accuracy:", test_acc)
print("\nRandom Forest — Classification report:\n", classification_report(y_test, test_pred))

# 6) Confusion matrix — visual table with errors
cnf_matrix = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6,5))
plt.imshow(cnf_matrix, interpolation='nearest')
plt.title(f"RF Confusion Matrix (n_estimators={n_best}, max_depth={d_best})")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.colorbar(); plt.tight_layout()
rf_cm_path = OUT_DIR / "confusion_matrix_rf.png"
plt.savefig(rf_cm_path, dpi=150)
plt.close()
print(f"[saved] {rf_cm_path.resolve()}")

# 7) Misclassified examples — real images where the model predicted incorrectly
mis_idx = np.where(y_test != test_pred)[0][:25]
if mis_idx.size > 0:
    rows, cols = 5, 5
    plt.figure(figsize=(10, 8))
    for i, idx in enumerate(mis_idx[:rows*cols]):
        plt.subplot(rows, cols, i+1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"T:{y_test[idx]} P:{test_pred[idx]}")
        plt.axis('off')
    plt.suptitle("Random Forest — Misclassified examples")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    rf_mis_path = OUT_DIR / "misclassified_examples_rf.png"
    plt.savefig(rf_mis_path, dpi=150)
    plt.close()
    print(f"[saved] {rf_mis_path.resolve()}")
else:
    print("No RF misclassified examples (very unlikely on MNIST).")

# 8) Histogram-Based Gradient Boosting — optional second model for comparison
try_hgb = True
if try_hgb:
    hgb = HistGradientBoostingClassifier(
        max_depth=20,     # recommended range: 10–40
        max_iter=800,     # recommended range: 500–1000 iterations
        random_state=RANDOM_STATE
    )
    hgb.fit(X_train, y_train)  # train on Train, validate on Validation
    print("\nHGB - Validation accuracy:", accuracy_score(y_validation, hgb.predict(X_validation)))

    # Retrain on Train + Validation and evaluate on Test
    hgb.fit(
        np.vstack([X_train, X_validation]),
        np.hstack([y_train, y_validation])
    )
    hgb_test_pred = hgb.predict(X_test)
    hgb_test_acc = accuracy_score(y_test, hgb_test_pred)
    print("HGB — Test accuracy:", hgb_test_acc)
    print("\nHGB — Classification report:\n", classification_report(y_test, hgb_test_pred))

    cm_hgb = confusion_matrix(y_test, hgb_test_pred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm_hgb, interpolation='nearest')
    plt.title("HGB Confusion Matrix (max_iter=800, max_depth=20)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    hgb_cm_path = OUT_DIR / "confusion_matrix_hgb.png"
    plt.savefig(hgb_cm_path, dpi=150)
    plt.close()
    print(f"[saved] {hgb_cm_path.resolve()}")
