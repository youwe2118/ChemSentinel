"""
dashboard.py
--------------------------------------------------------------
ChemSentinel - Software dashboard proof-of-concept (Streamlit)

WHAT THIS APP SHOWS:
    1. A large status box: green NORMAL or red INVESTIGATE (model output).
    2. The most recent reading (the "latest incoming row").
    3. A time-series graph of the two gas sensors.
    4. A log table of recent readings and the model's prediction.

HOW IT WORKS (important for the review):
    - There is NO live sensor and NO real-time telemetry.
    - The app REPLAYS rows from chemsentinel_simulated_data.csv.
      You move the slider to choose which reading is "the latest one".
    - Every row is SYNTHETIC (source_type = "synthetic_demo").
    - The model is the Logistic Regression from baseline_model.py, trained
      on the first 70% of rows. The status box is an ADVISORY software
      output, not a diagnosis of any toxic exposure.

HOW TO RUN:
    pip install streamlit pandas numpy scikit-learn
    streamlit run dashboard.py
    (ingest_and_validate.py, baseline_model.py and the CSV must be in
     the same folder.)
--------------------------------------------------------------
"""

import pandas as pd
import streamlit as st

# Re-use the functions we already wrote and explained
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
from baseline_model import train_baseline_model

DISCLAIMER = ("Educational prototype only; not a certified detector or "
              "replacement for industrial safety systems.")

LOG_ROWS_TO_SHOW = 10      # how many recent rows appear in the log table


# ------------------------------------------------------------
# 1. PREPARE DATA AND MODEL (runs once, then Streamlit remembers it)
# ------------------------------------------------------------
@st.cache_data
def prepare_replay_data():
    """Load, validate, train the model, and predict a state for every row.

    Returns a DataFrame with extra columns:
        model_prediction : "NORMAL" or "ABNORMAL" (from the model)
        data_split       : "train" or "test"  (which part the row belongs to)
    """
    raw_df = load_data(INPUT_FILE)
    clean_df, _ = validate_data(raw_df)
    train_df, test_df = split_train_test(clean_df, TRAIN_FRACTION)

    # Fit scaler and model on TRAIN rows only (same as baseline_model.py)
    scaler = fit_scaler_on_train(train_df)
    model = train_baseline_model(apply_scaler(scaler, train_df),
                                 train_df["target_label"])

    # Predict for every row so the dashboard can replay them
    all_scaled = apply_scaler(scaler, clean_df)
    clean_df["model_prediction"] = model.predict(all_scaled)

    # Remember which rows the model was trained on (honesty!)
    clean_df["data_split"] = "test"
    clean_df.loc[train_df.index, "data_split"] = "train"
    return clean_df


# ------------------------------------------------------------
# 2. SMALL DISPLAY HELPERS
# ------------------------------------------------------------
def show_status_box(prediction):
    """Draw a big coloured box: green NORMAL or red INVESTIGATE."""
    if prediction == "ABNORMAL":
        colour, text = "#c62828", "INVESTIGATE"       # red
        small_text = "Model output: ABNORMAL pattern"
    else:
        colour, text = "#2e7d32", "NORMAL"            # green
        small_text = "Model output: NORMAL pattern"

    # st.markdown with HTML lets us control size and colour
    st.markdown(
        f"""
        <div style="background-color:{colour}; padding:30px;
                    border-radius:12px; text-align:center;">
            <div style="color:white; font-size:60px; font-weight:bold;">{text}</div>
            <div style="color:white; font-size:20px;">{small_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# 3. THE PAGE
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="ChemSentinel Dashboard", layout="wide")
    st.title("ChemSentinel - Early-Warning Dashboard (Proof of Concept)")
    st.caption("Replaying SYNTHETIC data (source_type: synthetic_demo). "
               "No live sensors are connected.")

    df = prepare_replay_data()

    # ---- Replay control: choose which reading is "the latest" ----
    last_index = len(df) - 1
    chosen = st.slider("Replay position (reading number)",
                       min_value=0, max_value=last_index, value=0)

    history = df.iloc[: chosen + 1]       # everything received so far
    latest = df.iloc[chosen]              # the most recent row

    # ---- Status indicator + warning message ----
    show_status_box(latest["model_prediction"])
    st.write("")                          # small blank space
    if latest["model_prediction"] == "ABNORMAL":
        st.error("WARNING: unusual sensor pattern detected. "
                 "Please investigate. This is an advisory software alert, "
                 "not proof of exposure.")
    else:
        st.success("No unusual pattern detected in the latest reading.")

    # ---- Latest reading ----
    st.subheader("Most recent reading")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperature (C)", f"{latest['temperature_c']:.2f}")
    col2.metric("Humidity (%)", f"{latest['humidity_pct']:.2f}")
    col3.metric("MQ-7 raw", int(latest["gas_sensor_mq7_raw"]))
    col4.metric("MQ-135 raw", int(latest["gas_sensor_mq135_raw"]))
    st.caption(f"Timestamp: {latest['timestamp']}  |  "
               f"source_type: {latest['source_type']}  |  "
               f"this row was used for: {latest['data_split']}")

    # ---- Time-series graph ----
    st.subheader("Gas sensor readings so far")
    chart_data = history.set_index("timestamp")[
        ["gas_sensor_mq7_raw", "gas_sensor_mq135_raw"]
    ]
    st.line_chart(chart_data)

    # ---- Log table ----
    st.subheader(f"Reading log (last {LOG_ROWS_TO_SHOW} readings)")
    log_columns = ["timestamp"] + FEATURE_COLUMNS + [
        "model_prediction", "target_label", "data_split", "source_type"]
    # Newest first, so the latest reading is at the top
    st.dataframe(history[log_columns].tail(LOG_ROWS_TO_SHOW).iloc[::-1])
    st.caption("target_label is the label our own synthetic generator "
               "assigned. model_prediction is what the model decided.")

    # ---- Disclaimer (always at the bottom) ----
    st.divider()
    st.caption(DISCLAIMER)


main()
