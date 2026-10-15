"""
ingest_and_validate.py
--------------------------------------------------------------
ChemSentinel - Data ingestion, validation and scaling (Test C)

WHAT THIS SCRIPT DOES (in order):
    1. Load the CSV file.
    2. Validate it: find missing (NaN) and impossible values,
       remove the bad rows, and report what was removed.
    3. Split the clean data into TRAIN and TEST (in time order).
    4. Fit a StandardScaler on the TRAIN data ONLY, then use it
       to transform both TRAIN and TEST.   <-- prevents data leakage

All data is SYNTHETIC (source_type = "synthetic_demo").
Nothing here is a real measurement.

HOW TO RUN:
    pip install pandas numpy scikit-learn
    python ingest_and_validate.py
--------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------
INPUT_FILE = "chemsentinel_simulated_data.csv"
TRAIN_FRACTION = 0.70       # first 70% of rows (by time) = training data

# The numeric columns the model will use as input features
FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "gas_sensor_mq7_raw",
    "gas_sensor_mq135_raw",
]

# Every column we expect to find in the CSV (our agreed data schema)
REQUIRED_COLUMNS = ["timestamp"] + FEATURE_COLUMNS + ["source_type", "target_label"]

# Physical limits. A value outside these is "impossible".
# (min, max) for each feature column.
VALID_RANGES = {
    "temperature_c": (-273.15, 85.0),    # below absolute zero is impossible;
                                         # 85 C is a generous upper limit for a
                                         # room/plant sensor and is OUR assumption
    "humidity_pct": (0.0, 100.0),        # a percentage cannot be negative or > 100
    "gas_sensor_mq7_raw": (0, 4095),     # 12-bit ADC counts (assumption: confirm with ECE)
    "gas_sensor_mq135_raw": (0, 4095),
}

VALID_LABELS = ["NORMAL", "ABNORMAL"]


# ------------------------------------------------------------
# 2. LOADING
# ------------------------------------------------------------
def load_data(file_path):
    """Read the CSV into a DataFrame. Stops with a clear message if the
    file or a required column is missing (instead of a confusing crash)."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: could not find '{file_path}'. "
                         f"Run generate_synthetic_data.py first.")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"ERROR: the CSV is missing columns: {missing_cols}")

    return df


# ------------------------------------------------------------
# 3. VALIDATION
# ------------------------------------------------------------
def validate_data(df):
    """Check the data and return (clean_df, report).

    Each rule marks bad rows. We then drop those rows, so the rest of the
    pipeline never sees them. We do NOT crash on bad data.

    NOTE: we DROP bad rows instead of filling them with the column average.
    Filling with the average would use information from the whole dataset
    (including the future test rows), which is a form of data leakage.
    """
    df = df.copy()                       # never change the caller's data
    report = {}                          # rule name -> number of bad rows

    # Step A: turn text into numbers. Anything unreadable (e.g. "abc")
    # becomes NaN so the missing-value check below catches it.
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Step B: missing values (NaN) in any required column
    is_missing = df[REQUIRED_COLUMNS].isna().any(axis=1)
    report["missing_value"] = int(is_missing.sum())

    # Step C: impossible values, one column at a time
    is_impossible = pd.Series(False, index=df.index)
    for col, (low, high) in VALID_RANGES.items():
        out_of_range = (df[col] < low) | (df[col] > high)   # NaN compares as False
        report[f"out_of_range_{col}"] = int(out_of_range.sum())
        is_impossible = is_impossible | out_of_range

    # Step D: unknown labels (we only allow NORMAL / ABNORMAL)
    is_bad_label = ~df["target_label"].isin(VALID_LABELS)
    report["unknown_label"] = int(is_bad_label.sum())

    # Step E: honesty check - every row must be tagged as synthetic
    is_not_synthetic = df["source_type"] != "synthetic_demo"
    report["source_type_not_synthetic_demo"] = int(is_not_synthetic.sum())

    # Combine all rules: a row is bad if ANY rule flagged it
    is_bad_row = is_missing | is_impossible | is_bad_label | is_not_synthetic
    report["total_rows_before"] = len(df)
    report["rows_removed"] = int(is_bad_row.sum())

    clean_df = df[~is_bad_row].copy()
    report["total_rows_after"] = len(clean_df)

    # Keep rows in time order (needed for the time-based split)
    clean_df = clean_df.sort_values("timestamp").reset_index(drop=True)

    return clean_df, report


def print_report(report):
    """Print the validation report in a readable way."""
    print("----- VALIDATION REPORT -----")
    for name, value in report.items():
        print(f"{name:40s} {value}")
    print("-----------------------------\n")


# ------------------------------------------------------------
# 4. SPLITTING (done BEFORE scaling)
# ------------------------------------------------------------
def split_train_test(df, train_fraction):
    """Split by TIME: earlier rows = train, later rows = test.

    Why not a random split? Neighbouring readings are very similar, so a
    random split lets near-copies of test rows leak into training and makes
    results look better than they are. A time-ordered split is stricter.
    """
    cut = int(len(df) * train_fraction)
    train_df = df.iloc[:cut].copy()
    test_df = df.iloc[cut:].copy()

    # Safety check: both parts should contain both labels, otherwise the
    # later evaluation (precision/recall) would not make sense.
    for name, part in [("train", train_df), ("test", test_df)]:
        if part["target_label"].nunique() < 2:
            print(f"WARNING: the {name} set has only one label. "
                  f"Consider changing TRAIN_FRACTION.")
    return train_df, test_df


# ------------------------------------------------------------
# 5. SCALING (fit on TRAIN only)
# ------------------------------------------------------------
def fit_scaler_on_train(train_df):
    """Learn the mean and standard deviation from TRAINING data only."""
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])      # <-- .fit() sees ONLY train
    return scaler


def apply_scaler(scaler, df):
    """Apply an ALREADY-FITTED scaler. Never calls .fit() here, so test
    data can never influence the mean/std."""
    scaled_array = scaler.transform(df[FEATURE_COLUMNS])
    return pd.DataFrame(scaled_array, columns=FEATURE_COLUMNS, index=df.index)


# ------------------------------------------------------------
# 6. TEST C: deliberately corrupt some data and check we survive
# ------------------------------------------------------------
def make_corrupted_copy(df):
    """Return a copy of the data with a few deliberately broken values.
    This lets us PROVE the validator handles bad input (Test C).
    This is still synthetic test data, only broken on purpose."""
    bad = df.copy()
    bad = bad.astype({"temperature_c": float, "humidity_pct": float})
    bad.loc[5, "temperature_c"] = np.nan        # missing value
    bad.loc[10, "humidity_pct"] = -20.0         # negative humidity
    bad.loc[15, "temperature_c"] = -300.0       # below absolute zero
    bad.loc[20, "gas_sensor_mq7_raw"] = 99999   # beyond 12-bit ADC range
    bad.loc[25, "gas_sensor_mq135_raw"] = np.nan
    bad.loc[30, "target_label"] = "UNKNOWN"     # label not allowed
    return bad


# ------------------------------------------------------------
# 7. MAIN SCRIPT
# ------------------------------------------------------------
def main():
    # ---- Part 1: Test C - bad data must not crash the pipeline ----
    print("=== TEST C: validating a deliberately corrupted copy ===")
    raw_df = load_data(INPUT_FILE)
    corrupted_df = make_corrupted_copy(raw_df)
    _, test_c_report = validate_data(corrupted_df)
    print_report(test_c_report)
    print("Expected: 6 bad rows removed, and no crash.\n")

    # ---- Part 2: the normal pipeline on the real (clean) file ----
    print("=== NORMAL PIPELINE on the original file ===")
    clean_df, report = validate_data(raw_df)
    print_report(report)

    train_df, test_df = split_train_test(clean_df, TRAIN_FRACTION)
    print(f"Train rows: {len(train_df)}   Test rows: {len(test_df)}")
    print("Train labels:\n", train_df["target_label"].value_counts().to_string())
    print("Test labels:\n", test_df["target_label"].value_counts().to_string(), "\n")

    # Fit on TRAIN only, then transform BOTH
    scaler = fit_scaler_on_train(train_df)
    X_train_scaled = apply_scaler(scaler, train_df)
    X_test_scaled = apply_scaler(scaler, test_df)

    # Proof that there is no leakage:
    #  - train columns have mean ~0 and std ~1 (the scaler was fitted on them)
    #  - test columns do NOT (they were only transformed, not fitted)
    print("Scaled TRAIN mean (should be ~0):")
    print(X_train_scaled.mean().round(3).to_string(), "\n")
    print("Scaled TEST mean (NOT exactly 0 - that is correct):")
    print(X_test_scaled.mean().round(3).to_string())


if __name__ == "__main__":
    main()
