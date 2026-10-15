"""
generate_synthetic_data.py
--------------------------------------------------------------
ChemSentinel - Initial synthetic dataset generator (30% review)

PURPOSE:
    Create a small CSV that proves our data schema works BEFORE
    any real sensor data exists.

HONESTY NOTE:
    Every row is SYNTHETIC (made up by this script).
    Every row is tagged source_type = "synthetic_demo".
    The values are NOT measurements and NOT real chemical exposures.
    The "ABNORMAL" label is simply our own rule for this demo.

HOW TO RUN:
    pip install pandas numpy
    python generate_synthetic_data.py
--------------------------------------------------------------
"""

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. SETTINGS (change these numbers in one place)
# ------------------------------------------------------------
RANDOM_SEED = 42            # fixed seed -> same data every run (reproducible)
N_NORMAL = 150              # number of NORMAL rows
N_ABNORMAL = 30             # number of ABNORMAL rows
SAMPLE_INTERVAL_SEC = 10    # pretend a reading arrives every 10 seconds
START_TIME = "2025-01-01 09:00:00"   # fake start time (also synthetic)
OUTPUT_FILE = "chemsentinel_simulated_data.csv"

# Where the abnormal "events" start (row index) and how long each lasts.
# 3 events x 10 rows = 30 ABNORMAL rows. Real events last for a while,
# so we use bursts of consecutive rows instead of random single rows.
EVENT_STARTS = [40, 100, 150]
EVENT_LENGTH = 10

# Raw gas values are "ADC counts". We assume a 12-bit ADC (0 to 4095),
# which is common on microcontrollers such as the ESP32.
ADC_MIN = 0
ADC_MAX = 4095

# The final column order of the CSV (our data schema)
SCHEMA_COLUMNS = [
    "timestamp",
    "temperature_c",
    "humidity_pct",
    "gas_sensor_mq7_raw",
    "gas_sensor_mq135_raw",
    "source_type",
    "target_label",
]


# ------------------------------------------------------------
# 2. HELPER FUNCTIONS (one small job each)
# ------------------------------------------------------------
def make_normal_reading(rng):
    """Return one NORMAL reading as a dictionary.

    NORMAL = comfortable room, stable humidity, low gas readings.
    rng.normal(mean, std) draws a random number around 'mean'
    with a spread of 'std'.
    """
    return {
        "temperature_c": rng.normal(25.0, 0.5),         # room temp ~25 C
        "humidity_pct": rng.normal(55.0, 2.0),          # humidity ~55 %
        "gas_sensor_mq7_raw": rng.normal(150, 10),      # low baseline
        "gas_sensor_mq135_raw": rng.normal(200, 12),    # low baseline
        "target_label": "NORMAL",
    }


def make_abnormal_reading(rng):
    """Return one ABNORMAL reading as a dictionary.

    ABNORMAL = big spike in both gas sensors plus a slight
    temperature rise. Humidity stays close to normal.
    """
    return {
        "temperature_c": rng.normal(27.5, 0.8),         # slight shift up
        "humidity_pct": rng.normal(55.0, 2.5),          # roughly unchanged
        "gas_sensor_mq7_raw": rng.normal(600, 80),      # spike
        "gas_sensor_mq135_raw": rng.normal(700, 90),    # spike
        "target_label": "ABNORMAL",
    }


def build_abnormal_index_set():
    """Return the set of row numbers that belong to abnormal events."""
    abnormal_rows = set()
    for start in EVENT_STARTS:
        for i in range(start, start + EVENT_LENGTH):
            abnormal_rows.add(i)
    return abnormal_rows


# ------------------------------------------------------------
# 3. MAIN SCRIPT
# ------------------------------------------------------------
def main():
    rng = np.random.default_rng(RANDOM_SEED)

    total_rows = N_NORMAL + N_ABNORMAL
    abnormal_rows = build_abnormal_index_set()

    # Safety check: our event settings must match the row counts above.
    if len(abnormal_rows) != N_ABNORMAL:
        raise ValueError(
            f"EVENT_STARTS x EVENT_LENGTH gives {len(abnormal_rows)} rows, "
            f"but N_ABNORMAL is {N_ABNORMAL}. Please make them match."
        )
    if max(abnormal_rows) >= total_rows:
        raise ValueError("An abnormal event runs past the last row.")

    # Create evenly spaced fake timestamps, one per row.
    timestamps = pd.date_range(
        start=START_TIME,
        periods=total_rows,
        freq=f"{SAMPLE_INTERVAL_SEC}s",
    )

    # Build the rows one by one (simple loop, easy to explain).
    rows = []
    for i in range(total_rows):
        if i in abnormal_rows:
            reading = make_abnormal_reading(rng)
        else:
            reading = make_normal_reading(rng)

        reading["timestamp"] = timestamps[i]
        reading["source_type"] = "synthetic_demo"   # academic honesty tag
        rows.append(reading)

    df = pd.DataFrame(rows)

    # Realistic cleanup:
    #  - ADC readings are whole numbers and must stay inside 0..4095
    #  - temperature and humidity are rounded to 2 decimals
    #  - humidity is a percentage, so keep it between 0 and 100
    for col in ["gas_sensor_mq7_raw", "gas_sensor_mq135_raw"]:
        df[col] = df[col].clip(ADC_MIN, ADC_MAX).round().astype(int)
    df["temperature_c"] = df["temperature_c"].round(2)
    df["humidity_pct"] = df["humidity_pct"].clip(0, 100).round(2)

    # Put the columns in the agreed schema order.
    df = df[SCHEMA_COLUMNS]

    # Export to CSV (index=False so we don't save the row numbers).
    df.to_csv(OUTPUT_FILE, index=False)

    # Quick summary so we can check the result immediately.
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}\n")
    print("Label counts:")
    print(df["target_label"].value_counts(), "\n")
    print("Average values per label:")
    print(df.groupby("target_label")[
        ["temperature_c", "humidity_pct",
         "gas_sensor_mq7_raw", "gas_sensor_mq135_raw"]
    ].mean().round(1), "\n")
    print("First 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()