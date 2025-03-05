import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set up Streamlit page config
st.set_page_config(page_title="Shipping Dashboard", layout="wide")

# Sidebar for navigation
page = st.sidebar.radio("Select Page", ["Home", "Vessel Profile", "LNG Market"])

if page == "Home":
    st.title("Shipping Market Equilibrium Calculator")

    # Inputs
    fleet_size_number_supply = st.number_input("Fleet Size (Number of Ships)", value=3131, step=1, format="%d")
    fleet_size_dwt_supply_in_dwt_million = st.number_input("Fleet Size Supply (Million DWT)", value=254.1, step=0.1)
    utilization_constant = st.number_input("Utilization Constant", value=0.95, step=0.01)
    
    assumed_speed = st.number_input("Assumed Speed (knots)", value=11.0, step=0.1)
    sea_margin = st.number_input("Sea Margin", value=0.05, step=0.01)
    
    assumed_laden_days = st.number_input("Assumed Laden Days Fraction", value=0.4, step=0.01)
    
    demand_billion_ton_mile = st.number_input("Demand (Billion Ton Mile)", value=10396.0, step=10.0)
    
    # Calculations
    dwt_utilization = (fleet_size_dwt_supply_in_dwt_million * 1_000_000 / fleet_size_number_supply) * utilization_constant
    distance_travelled_per_day = assumed_speed * 24 * (1 - sea_margin)
    productive_laden_days_per_year = assumed_laden_days * 365
    
    # Maximum Supply Calculation
    maximum_supply_billion_ton_mile = fleet_size_number_supply * dwt_utilization * distance_travelled_per_day * productive_laden_days_per_year / 1_000_000_000
    
    # Equilibrium
    equilibrium = demand_billion_ton_mile - maximum_supply_billion_ton_mile
    result = "Excess Supply" if equilibrium < 0 else "Excess Demand"
    
    # Display results
    st.subheader("Results:")
    st.metric(label="DWT Utilization (tons)", value=f"{dwt_utilization:,.2f}")
    st.metric(label="Distance Travelled per Day (nm)", value=f"{distance_travelled_per_day:,.2f}")
    st.metric(label="Productive Laden Days per Year", value=f"{productive_laden_days_per_year:,.2f}")
    st.metric(label="Maximum Supply (Billion Ton Mile)", value=f"{maximum_supply_billion_ton_mile:,.2f}")
    st.metric(label="Equilibrium (Billion Ton Mile)", value=f"{equilibrium:,.2f}")
    st.metric(label="Market Condition", value=result)
    
    # Visualization
    fig, ax = plt.subplots()
    ax.bar(["Demand", "Supply"], [demand_billion_ton_mile, maximum_supply_billion_ton_mile], color=['blue', 'orange'])
    ax.set_ylabel("Billion Ton Mile")
    ax.set_title("Supply vs Demand")
    st.pyplot(fig)

if page == "Vessel Profile":
    st.title("🚢 Vessel Profile")
    
    # Vessel Data
    vessel_data = pd.DataFrame({
        "Vessel_ID": range(1, 11),
        "Name": [
            "LNG Carrier Alpha", "LNG Carrier Beta", "LNG Carrier Gamma", "LNG Carrier Delta",
            "LNG Carrier Epsilon", "LNG Carrier Zeta", "LNG Carrier Theta", "LNG Carrier Iota",
            "LNG Carrier Kappa", "LNG Carrier Lambda"
        ],
        "Sister_Ship_Group": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
        "Capacity_CBM": [160000] * 10,
        "FuelEU_GHG_Compliance": [65, 65, 65, 80, 80, 80, 95, 95, 95, 95],
        "CII_Rating": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
        "Fuel_Consumption_MT_per_day": [70, 72, 74, 85, 88, 90, 100, 102, 105, 107]
    })
    
    # Input for Fuel Price
    fuel_price = st.number_input("Enter Fuel Price (per MT in USD)", min_value=0.0, value=500.0, step=10.0)
    
    # Calculate Fuel Cost Per Day
    vessel_data["Fuel_Cost_per_Day"] = vessel_data["Fuel_Consumption_MT_per_day"] * fuel_price
    
    # Display the table
    st.dataframe(vessel_data)
    
    # Show a summary of total fleet fuel cost
    total_fuel_cost = vessel_data["Fuel_Cost_per_Day"].sum()
    st.metric(label="Total Fleet Fuel Cost per Day (USD)", value=f"${total_fuel_cost:,.2f}")

if page == "LNG Market":
    st.title("📈 LNG Market Trends")
    
    # Google Sheets URL
    google_sheets_url = "https://docs.google.com/spreadsheets/d/1kySjcfv1jMkDRrqAD9qS10KjIs5H1Vdu/export?format=csv"
    
    # Read data from Google Sheets
    df_weekly = pd.read_csv(google_sheets_url, parse_dates=["Date"])
    df_monthly = pd.read_csv(google_sheets_url, parse_dates=["Date"])
    df_yearly = pd.read_csv(google_sheets_url, parse_dates=["Date"])
    
    # Select frequency
    freq_option = st.radio("Select Data Frequency", ["Weekly", "Monthly", "Yearly"])
    
    if freq_option == "Weekly":
        df_selected = df_weekly
        column_options = st.multiselect("Select Data Columns", df_weekly.columns.drop("Date"), default=df_weekly.columns[1])
    elif freq_option == "Monthly":
        df_selected = df_monthly
        column_options = st.multiselect("Select Data Columns", df_monthly.columns.drop("Date"), default=df_monthly.columns[1])
    else:
        df_selected = df_yearly
        column_options = st.multiselect("Select Data Columns", df_yearly.columns.drop("Date"), default=df_yearly.columns[1])
    
    # Select time range
    start_date = st.date_input("Select Start Date", df_selected["Date"].min())
    end_date = st.date_input("Select End Date", df_selected["Date"].max())
    df_filtered = df_selected[(df_selected["Date"] >= pd.to_datetime(start_date)) & (df_selected["Date"] <= pd.to_datetime(end_date))]
    
    # Plot time series with date on x-axis and rate on y-axis
    fig, ax = plt.subplots(figsize=(10, 4))
    for column in column_options:
        ax.plot(df_filtered["Date"], df_filtered[column], label=column)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rate")
    ax.set_title("LNG Market Rates Over Time")
    ax.legend()
    ax.grid()
    st.pyplot(fig)
