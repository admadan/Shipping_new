
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---- LSTM model and LNG Market Page ----

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---- LSTM function ----
def run_lstm_model(df, target_column, sequence_length=10):
    df = df.sort_values(by='Date')
    series = df[target_column].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    train_size = int(len(scaled_series) * 0.8)
    train_data = scaled_series[:train_size]
    test_data = scaled_series[train_size:]

    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    predicted_scaled = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted_scaled)
    actual = scaler.inverse_transform(y_test)

    return predicted, actual

# ---- LNG Market Page ----
def lng_market_page():
    st.title("📈 LNG Market Trends with Forecast")

    base_url = "https://docs.google.com/spreadsheets/d/1kySjcfv1jMkDRrqAD9qS10KjIs5H1Vdu/gviz/tq?tqx=out:csv&sheet="
    sheet_names = {
        "Weekly": "Weekly%20data_160K%20CBM",
        "Monthly": "Monthly%20data_160K%20CBM",
        "Yearly": "Yearly%20data_160%20CBM"
    }

    freq_option = st.radio("Select Data Frequency", ["Weekly", "Monthly", "Yearly"])
    google_sheets_url = f"{base_url}{sheet_names[freq_option]}"

    try:
        df_selected = pd.read_csv(google_sheets_url, dtype=str)

        if "Date" in df_selected.columns:
            df_selected["Date"] = pd.to_datetime(df_selected["Date"], errors='coerce')
            df_selected = df_selected.dropna(subset=["Date"]).sort_values(by="Date")
        else:
            st.error("⚠️ 'Date' column not found in the dataset.")
            return

        for col in df_selected.columns:
            if col != "Date":
                df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce').fillna(0)

        available_columns = [col for col in df_selected.columns if col != "Date"]
        selected_column = st.selectbox("Select Parameter for LSTM Forecast", available_columns)

        start_date = st.date_input("Select Start Date", df_selected["Date"].min())
        end_date = st.date_input("Select End Date", df_selected["Date"].max())
        df_filtered = df_selected[(df_selected["Date"] >= pd.to_datetime(start_date)) & (df_selected["Date"] <= pd.to_datetime(end_date))]

        st.subheader(f"Time Series Plot for: {selected_column}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_filtered["Date"], df_filtered[selected_column], label=selected_column)
        ax.set_title(f"{selected_column} Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel(selected_column)
        ax.grid(True)
        st.pyplot(fig)

        if len(df_filtered) > 20:
            st.subheader("LSTM Forecast")
            predicted, actual = run_lstm_model(df_filtered, selected_column)

            fig_lstm, ax_lstm = plt.subplots(figsize=(10, 4))
            ax_lstm.plot(actual, label="Actual", color="blue")
            ax_lstm.plot(predicted, label="Predicted", color="red")
            ax_lstm.set_title(f"LSTM Prediction for {selected_column}")
            ax_lstm.set_ylabel(selected_column)
            ax_lstm.set_xlabel("Index")
            ax_lstm.legend()
            ax_lstm.grid(True)
            st.pyplot(fig_lstm)

    except Exception as e:
        st.error(f"❌ Error loading or processing data: {e}")


# ---- Streamlit Page Config ----
st.set_page_config(page_title="Shipping Dashboard", layout="wide")

# ---- Sidebar Navigation ----
page = st.sidebar.radio("Select Page", ["Home", "Vessel Profile", "LNG Market", "Yearly Simulation"])

# ---- Home Page ----
if page == "Home":
    st.title("Shipping Market Equilibrium Calculator")
    fleet_size_number_supply = st.number_input("Fleet Size (Number of Ships)", value=3131, step=1, format="%d")
    fleet_size_dwt_supply_in_dwt_million = st.number_input("Fleet Size Supply (Million DWT)", value=254.1, step=0.1)
    utilization_constant = st.number_input("Utilization Constant", value=0.95, step=0.01)
    assumed_speed = st.number_input("Assumed Speed (knots)", value=11.0, step=0.1)
    sea_margin = st.number_input("Sea Margin", value=0.05, step=0.01)
    assumed_laden_days = st.number_input("Assumed Laden Days Fraction", value=0.4, step=0.01)
    demand_billion_ton_mile = st.number_input("Demand (Billion Ton Mile)", value=10396.0, step=10.0)

    dwt_utilization = (fleet_size_dwt_supply_in_dwt_million * 1_000_000 / fleet_size_number_supply) * utilization_constant
    distance_travelled_per_day = assumed_speed * 24 * (1 - sea_margin)
    productive_laden_days_per_year = assumed_laden_days * 365
    maximum_supply_billion_ton_mile = fleet_size_number_supply * dwt_utilization * distance_travelled_per_day * productive_laden_days_per_year / 1_000_000_000
    equilibrium = demand_billion_ton_mile - maximum_supply_billion_ton_mile
    result = "Excess Supply" if equilibrium < 0 else "Excess Demand"

    st.subheader("Results:")
    st.metric("DWT Utilization (tons)", f"{dwt_utilization:,.2f}")
    st.metric("Distance Travelled per Day (nm)", f"{distance_travelled_per_day:,.2f}")
    st.metric("Productive Laden Days per Year", f"{productive_laden_days_per_year:,.2f}")
    st.metric("Maximum Supply (Billion Ton Mile)", f"{maximum_supply_billion_ton_mile:,.2f}")
    st.metric("Equilibrium (Billion Ton Mile)", f"{equilibrium:,.2f}")
    st.metric("Market Condition", result)

    fig, ax = plt.subplots()
    ax.bar(["Demand", "Supply"], [demand_billion_ton_mile, maximum_supply_billion_ton_mile], color=['blue', 'orange'])
    ax.set_ylabel("Billion Ton Mile")
    ax.set_title("Supply vs Demand")
    st.pyplot(fig)

# ---- Vessel Profile ----
if page == "Vessel Profile":
    st.title("🚢 Vessel Profile")
    # Content omitted here for brevity (same as earlier)

# ---- LNG Market ----
if page == "LNG Market":
    lng_market_page()

# ---- Yearly Simulation ----
if page == "Yearly Simulation":
    st.title("📊 Yearly Simulation Dashboard")
    # Content omitted here for brevity (same as earlier)
