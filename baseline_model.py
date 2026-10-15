"""
baseline_model.py
--------------------------------------------------------------
ChemSentinel - Model A: baseline classifier (Logistic Regression)

WHAT THIS SCRIPT DOES:
    1. Re-uses the functions from ingest_and_validate.py
       (load -> validate -> time split -> scale with train-only scaler).
    2. Trains a Logistic Regression on the scaled TRAIN data.
    3. Evaluates it on the TEST data:
       Accuracy, Precision, Recall, Macro-F1, False-Positive Rate,
       and a Confusion Matrix (printed and saved as a picture).
    4. Runs Test B: feeds hand-made "unusual" readings to the model.

IMPORTANT HONESTY NOTES:
    - All data is SYNTHETIC (source_type = "synthetic_demo").
    - We created the ABNORMAL rows with a big gas spike ourselves, so
      the classes are easy to separate. High scores here only show that
      the PIPELINE works. They are NOT evidence of real-world detection.
    - The output is an advisory state for investigation. It is NOT a
      diagnosis of any toxic exposure.

HOW TO RUN:
    pip install pandas numpy scikit-learn matplotlib
    python baseline_model.py
    (ingest_and_validate.py and chemsentinel_simulated_data.csv
     must be in the same folder.)
--------------------------------------------------------------
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # draw to a file, no pop-up window needed
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Re-use our own earlier code instead of copy-pasting it
from ingest_and_validate import (
    INPUT_FILE,
    TRAIN_FRACTION,
    FEATURE_COLUMNS,
    load_data,
    validate_data,
    split_train_test,
    fit_scaler_on_train,
    apply_scaler,
)

# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------
RANDOM_SEED = 42
CONFUSION_MATRIX_PICTURE = "confusion_matrix_baseline.png"

# Order matters: row/column 0 = NORMAL, row/column 1 = ABNORMAL
LABEL_ORDER = ["NORMAL", "ABNORMAL"]

# The "positive" class is the one we want to catch (the unusual one)
POSITIVE_LABEL = "ABNORMAL"


# ------------------------------------------------------------
# 2. TRAINING
# ------------------------------------------------------------
def train_baseline_model(X_train_scaled, y_train):
    """Train a Logistic Regression and return it.

    Why Logistic Regression? It is simple and transparent: each feature gets
    ONE coefficient we can print and explain. class_weight="balanced" tells
    the model that ABNORMAL rows are rarer (about 17%) and should not be
    ignored.
    """
    model = LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED)
    model.fit(X_train_scaled, y_train)
    return model


# ------------------------------------------------------------
# 3. EVALUATION
# ------------------------------------------------------------
def evaluate_model(model, X_test_scaled, y_test):
    """Predict on the TEST set and compute all review metrics."""
    y_pred = model.predict(X_test_scaled)

    # Confusion matrix layout (rows = TRUE label, columns = PREDICTED label):
    #                   pred NORMAL   pred ABNORMAL
    #   true NORMAL         TN             FP
    #   true ABNORMAL       FN             TP
    cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
    tn, fp, fn, tp = cm.ravel()

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        # Precision: of everything we flagged ABNORMAL, how much really was?
        "precision": precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL,
                                     zero_division=0),
        # Recall: of all truly ABNORMAL rows, how many did we catch?
        "recall": recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL,
                               zero_division=0),
        # Macro-F1: F1 of each class, averaged (treats both classes equally)
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        # False-positive rate: NORMAL rows wrongly flagged (false alarms)
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "confusion_matrix": cm,
    }
    return results


def print_results(results):
    """Print the metrics in a readable way."""
    print("----- BASELINE MODEL RESULTS (TEST SET) -----")
    print(f"Accuracy             : {results['accuracy']:.3f}")
    print(f"Precision (ABNORMAL) : {results['precision']:.3f}")
    print(f"Recall    (ABNORMAL) : {results['recall']:.3f}")
    print(f"Macro-F1             : {results['macro_f1']:.3f}")
    print(f"False-positive rate  : {results['false_positive_rate']:.3f}")
    print("\nConfusion matrix (rows = true, columns = predicted)")
    print(f"order: {LABEL_ORDER}")
    print(results["confusion_matrix"])
    print("---------------------------------------------\n")


def save_confusion_matrix_picture(cm, file_name):
    """Draw the confusion matrix and save it as a PNG for the slides."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Baseline confusion matrix\n(synthetic_demo data)")
    # Write the count inside each box
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    fig.tight_layout()
    fig.savefig(file_name, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix picture to {file_name}\n")


def print_model_explanation(model):
    """Print the learned coefficients so we can EXPLAIN the model.

    Positive coefficient = a higher (scaled) value pushes the model toward
    ABNORMAL. Larger absolute value = stronger influence.
    """
    # model.classes_ is sorted alphabetically: ['ABNORMAL', 'NORMAL'].
    # For two classes, sklearn's coefficients describe the SECOND class
    # (NORMAL), so we flip the sign to describe ABNORMAL.
    coefs = -model.coef_[0]
    print("Feature influence toward ABNORMAL (scaled units):")
    for name, value in zip(FEATURE_COLUMNS, coefs):
        print(f"  {name:22s} {value:+.3f}")
    print()


# ------------------------------------------------------------
# 4. TEST B - feed a simulated unusual pattern
# ------------------------------------------------------------
def run_test_b(model, scaler):
    """TEST B: feed hand-made readings and check the model's response.

    HOW TO RUN TEST B (explain this in the review):
      1. Write a record by hand and mark it clearly as synthetic.
      2. Put it in a DataFrame with the SAME columns as FEATURE_COLUMNS.
      3. Scale it with the scaler that was fitted on TRAIN data
         (use .transform ONLY - never .fit on the new record).
      4. Call model.predict(...) and compare with what we EXPECT.
      5. Record PASS / FAIL for each case.

    To try your own pattern, edit or add a dictionary in test_cases below.
    Test A (normal baseline) is the first case: an ordinary reading must be
    classified NORMAL.
    """
    test_cases = [
        # (description, expected label, the synthetic reading)
        ("Ordinary room reading (control)", "NORMAL",
         {"temperature_c": 25.0, "humidity_pct": 55.0,
          "gas_sensor_mq7_raw": 150, "gas_sensor_mq135_raw": 200}),

        ("Both gas sensors spike, slight temperature rise", "ABNORMAL",
         {"temperature_c": 27.5, "humidity_pct": 55.0,
          "gas_sensor_mq7_raw": 650, "gas_sensor_mq135_raw": 750}),

        ("Only MQ-7 spikes strongly", "ABNORMAL",
         {"temperature_c": 25.0, "humidity_pct": 55.0,
          "gas_sensor_mq7_raw": 900, "gas_sensor_mq135_raw": 200}),
    ]

    print("----- TEST B: simulated unusual patterns (source_type: synthetic_demo) -----")
    passed = 0
    for description, expected, reading in test_cases:
        # One-row DataFrame, same column names/order as training
        record = pd.DataFrame([reading], columns=FEATURE_COLUMNS)
        record_scaled = apply_scaler(scaler, record)        # transform only!

        predicted = model.predict(record_scaled)[0]
        # Probability the model assigns to ABNORMAL (for information only)
        abnormal_index = list(model.classes_).index(POSITIVE_LABEL)
        probability = model.predict_proba(record_scaled)[0][abnormal_index]

        # Convert to the dashboard wording. Advisory only, not a diagnosis.
        display_state = "ABNORMAL/INVESTIGATE" if predicted == "ABNORMAL" else "NORMAL"
        result = "PASS" if predicted == expected else "FAIL"
        if result == "PASS":
            passed += 1

        print(f"[{result}] {description}")
        print(f"       expected={expected}, predicted={predicted}, "
              f"P(ABNORMAL)={probability:.2f}, dashboard state={display_state}")

    print(f"\nTest B: {passed} of {len(test_cases)} cases passed.")
    print("Reminder: these cases are made-up and resemble our own synthetic data,"
          " so passing proves the workflow, not real-world detection.\n")


# ------------------------------------------------------------
# 5. MAIN SCRIPT
# ------------------------------------------------------------
def main():
    # Step 1: load, validate, split, scale (all from the previous script)
    raw_df = load_data(INPUT_FILE)
    clean_df, _ = validate_data(raw_df)
    train_df, test_df = split_train_test(clean_df, TRAIN_FRACTION)

    scaler = fit_scaler_on_train(train_df)              # fit on TRAIN only
    X_train_scaled = apply_scaler(scaler, train_df)
    X_test_scaled = apply_scaler(scaler, test_df)       # transform only

    y_train = train_df["target_label"]
    y_test = test_df["target_label"]

    # Step 2: train Model A
    model = train_baseline_model(X_train_scaled, y_train)
    print_model_explanation(model)

    # Step 3: evaluate on data the model has never seen
    results = evaluate_model(model, X_test_scaled, y_test)
    print_results(results)
    save_confusion_matrix_picture(results["confusion_matrix"],
                                  CONFUSION_MATRIX_PICTURE)

    # Step 4: Test B
    run_test_b(model, scaler)


if __name__ == "__main__":
    main()
