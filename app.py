import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set up Streamlit page config
st.set_page_config(page_title="Shipping Dashboard", layout="wide")

# Sidebar for navigation
page = st.sidebar.radio("Select Page", ["Home", "Vessel Profile", "LNG Market", "Yearly Simulation"])

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
    
    # Input for Fuel Price and Voyage Days
    fuel_price = st.number_input("Enter Fuel Price (per MT in USD)", min_value=0.0, value=500.0, step=10.0)
    voyage_days = st.number_input("Enter Voyage Days", min_value=1, value=10, step=1)
    
    # Calculate Fuel Cost Per Day and Total Cost
    vessel_data["Fuel_Cost_per_Day"] = vessel_data["Fuel_Consumption_MT_per_day"] * fuel_price
    vessel_data["Total_Voyage_Cost"] = vessel_data["Fuel_Cost_per_Day"] * voyage_days
    
    # Display the table
    st.dataframe(vessel_data)
    
    # Show a summary of total fleet fuel cost
    total_fuel_cost = vessel_data["Fuel_Cost_per_Day"].sum()
    total_voyage_cost = vessel_data["Total_Voyage_Cost"].sum()
    st.metric(label="Total Fleet Fuel Cost per Day (USD)", value=f"${total_fuel_cost:,.2f}")
    st.metric(label="Total Voyage Cost (USD)", value=f"${total_voyage_cost:,.2f}")



if page == "LNG Market":
    st.title("📈 LNG Market Trends")
    
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
        
        for col in df_selected.columns:
            if col != "Date":
                df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce').fillna(0)

        available_columns = [col for col in df_selected.columns if col != "Date"]
        column_options = st.multiselect("Select Data Columns", available_columns, default=available_columns[:2] if available_columns else [])
        
        if "Date" in df_selected.columns:
            start_date = st.date_input("Select Start Date", df_selected["Date"].min())
            end_date = st.date_input("Select End Date", df_selected["Date"].max())
            df_filtered = df_selected[(df_selected["Date"] >= pd.to_datetime(start_date)) & (df_selected["Date"] <= pd.to_datetime(end_date))]
            
            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax1.set_xlabel("Date")
            ax1.set_title("LNG Market Rates Over Time")
            ax1.tick_params(axis='x', rotation=45)
            
            if len(column_options) > 1:
                ax2 = ax1.twinx()
                ax1.set_ylabel(column_options[0])
                ax2.set_ylabel(column_options[1])
                ax1.plot(df_filtered["Date"], df_filtered[column_options[0]], label=column_options[0], color='blue')
                ax2.plot(df_filtered["Date"], df_filtered[column_options[1]], label=column_options[1], color='orange')
                ax1.set_ylim(df_filtered[column_options[0]].min(), df_filtered[column_options[0]].max())
                ax2.set_ylim(df_filtered[column_options[1]].min(), df_filtered[column_options[1]].max())
            else:
                ax1.set_ylabel(column_options[0])
                ax1.plot(df_filtered["Date"], df_filtered[column_options[0]], label=column_options[0], color='blue')
                ax1.set_ylim(df_filtered[column_options[0]].min(), df_filtered[column_options[0]].max())
            
            ax1.legend()
            ax1.grid()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")

if page == "Yearly Simulation":
    st.title("📊 Yearly Simulation Dashboard")
    
    base_url = "https://docs.google.com/spreadsheets/d/1kySjcfv1jMkDRrqAD9qS10KjIs5H1Vdu/gviz/tq?tqx=out:csv&sheet=Yearly%20equilibrium"
    
    try:
        df_yearly_sim = pd.read_csv(base_url, dtype=str)
        
        if "Year" in df_yearly_sim.columns:
            df_yearly_sim["Year"] = pd.to_datetime(df_yearly_sim["Year"], format="%Y", errors='coerce').dt.year
            df_yearly_sim = df_yearly_sim.dropna(subset=["Year"]).sort_values(by="Year")
        else:
            st.error("⚠️ 'Year' column not found in the dataset.")
        
        for col in df_yearly_sim.columns:
            if col != "Year":
                df_yearly_sim[col] = pd.to_numeric(df_yearly_sim[col], errors='coerce').fillna(0)

        available_columns = [col for col in df_yearly_sim.columns if col != "Year"]
        variable_option = st.multiselect("Select Data Columns", available_columns, default=available_columns[:2] if available_columns else [])
        
        start_year = st.number_input("Select Start Year", int(df_yearly_sim["Year"].min()), int(df_yearly_sim["Year"].max()), int(df_yearly_sim["Year"].min()))
        end_year = st.number_input("Select End Year", int(df_yearly_sim["Year"].min()), int(df_yearly_sim["Year"].max()), int(df_yearly_sim["Year"].max()))
        df_filtered = df_yearly_sim[(df_yearly_sim["Year"] >= start_year) & (df_yearly_sim["Year"] <= end_year)]
        
        # User selects a specific year
        selected_year = st.number_input("Select a Year to Display Data", int(df_yearly_sim["Year"].min()), int(df_yearly_sim["Year"].max()), int(df_yearly_sim["Year"].max()))
        
        if variable_option:
            selected_data = df_yearly_sim[df_yearly_sim["Year"] == selected_year][variable_option]
            st.subheader(f"Data for {selected_year}")
            st.write(selected_data)
        
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.set_xlabel("Year")
        ax1.set_title("Yearly Simulation Trends")
        ax1.tick_params(axis='x', rotation=45)
        
        if len(variable_option) > 1:
            ax2 = ax1.twinx()
            ax1.set_ylabel(variable_option[0])
            ax2.set_ylabel(variable_option[1])
            ax1.plot(df_filtered["Year"], df_filtered[variable_option[0]], label=variable_option[0], color='blue')
            ax2.plot(df_filtered["Year"], df_filtered[variable_option[1]], label=variable_option[1], color='orange')
            ax1.set_ylim(df_filtered[variable_option[0]].min(), df_filtered[variable_option[0]].max())
            ax2.set_ylim(df_filtered[variable_option[1]].min(), df_filtered[variable_option[1]].max())
        else:
            ax1.set_ylabel(variable_option[0])
            ax1.plot(df_filtered["Year"], df_filtered[variable_option[0]], label=variable_option[0], color='blue')
            ax1.set_ylim(df_filtered[variable_option[0]].min(), df_filtered[variable_option[0]].max())
            if '%' in variable_option[0]:
                ax1.yaxis.set_major_formatter(PercentFormatter())
        
        ax1.legend()
        ax1.grid()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")