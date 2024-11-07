import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple

# Page config
st.set_page_config(page_title='RV Value Prediction', page_icon=":car:", layout="wide")

# Title and description
st.title('Residual Value Prediction')
st.markdown("""
* Dataset: Data from 2010-2021
""")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load and cache the dataset with optimized dtypes"""
    # Specify dtypes for better memory usage
    dtypes = {
        'Make_motto': 'category',
        'Model_I': 'category',
        'Make_jato': 'category',
        'Model_jato': 'category',
        'Version_Name_jato': 'category',
        'Local_Version_Name': 'category',
        'SalesStatus': 'category',
        'Color': 'category',
        'Make_Group': 'category',
        'Condition': 'category',
        'Taxi': 'category',
        'Year': 'int32',
        'YearDatePriced': 'int32',
        'MSRP': 'float32',
        'Mileage': 'float32',
        'Age': 'float32',
        'SoldPrice': 'float32',
        'RV_value': 'float32'
    }
    
    # Read only needed columns
    needed_columns = list(dtypes.keys()) + ['DatePriced']
    
    df = pd.read_csv(
        path,
        usecols=needed_columns,
        dtype=dtypes
    )
    
    return df

@st.cache_data
def filter_makes_models(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Cache the unique makes and their corresponding models"""
    makes_models = {}
    for make in df['Make_jato'].unique():
        models = df[df['Make_jato'] == make]['Model_I'].unique().tolist()
        makes_models[make] = sorted(models)
    return makes_models

@st.cache_data
def filter_models_trims(df: pd.DataFrame, make: str, model: str) -> List[str]:
    """Cache the unique trims for each make/model combination"""
    mask = (df['Make_jato'] == make) & (df['Model_I'] == model)
    return sorted(df[mask]['Version_Name_jato'].unique().tolist())

@st.cache_data
def filter_trim_years(df: pd.DataFrame, make: str, model: str, trim: str) -> List[int]:
    """Cache the unique years for each make/model/trim combination"""
    mask = (df['Make_jato'] == make) & (df['Model_I'] == model) & (df['Version_Name_jato'] == trim)
    return sorted(df[mask]['Year'].unique().tolist())

@st.cache_data
def calculate_prediction_lines(results, filtered_data) -> Tuple[List[int], List[float]]:
    """Calculate prediction lines with proper integer handling for x_range"""
    coef = pd.DataFrame(
        results.iloc[0]["px_fit_results"].summary().tables[1].data[1:3],
        columns=results.iloc[0]["px_fit_results"].summary().tables[1].data[0]
    )[['coef', '[0.025', '0.975]']]

    coef_mean = coef.transpose().loc['coef', :].astype('float').tolist()
    
    # Convert float values to integers for range
    x_min = int(np.ceil(filtered_data.Age.min()))  # Round up to nearest integer
    x_max = int(np.floor(filtered_data.Age.max()))  # Round down to nearest integer
    
    if x_max <= x_min:
        x_max = x_min + 1  # Ensure at least one point
        
    x_range = list(range(x_min, x_max + 1))  # Include the last point
    base_pred = [coef_mean[0] + coef_mean[1] * np.log10(max(1, i)) for i in x_range]  # Avoid log(0)
    
    return x_range, base_pred

try:
    # Load data
    file_path = "/Users/chanont/Desktop/Desktop/Carro/Script/Handover Doc mac/Pulldata from S3 py/merged_data_for_streamlit_new_15112021_filtered.csv"
    graphdata = load_data(file_path)
    
    # Sidebar filters using cached functions
    st.sidebar.title('Select Parameters')
    
    # Get cached makes and models
    makes_models = filter_makes_models(graphdata)
    make_selection = st.sidebar.selectbox('Select Brand', sorted(makes_models.keys()))
    
    # Get models for selected make
    model_selection = st.sidebar.selectbox('Select Model', makes_models[make_selection])
    
    # Get trims for selected make/model
    trims = filter_models_trims(graphdata, make_selection, model_selection)
    trim_selection = st.sidebar.selectbox('Select Sub Model', trims)
    
    # Get years for selected make/model/trim
    years = filter_trim_years(graphdata, make_selection, model_selection, trim_selection)
    modelyear_selection = st.sidebar.selectbox('Select Model Year', years)

    # Filter data based on selections - now more efficient
    mask = (
        (graphdata.Model_I == model_selection) &
        (graphdata.Version_Name_jato == trim_selection) &
        (graphdata.Year == modelyear_selection)
    )
    filtered_data = graphdata[mask].copy()

    # Clean data - more efficient with mask operations
    filtered_data = filtered_data.dropna(subset=['Mileage', 'Age', 'SoldPrice'])
    
    if not filtered_data.empty:
        # Display header information
        st.title('🚗 Residual Value Prediction')
        st.subheader(f"{make_selection} {model_selection} {trim_selection} {modelyear_selection}")
        st.subheader(f"MSRP: {filtered_data.MSRP.iloc[0]:,.2f}")

        # Create scatter plot with optimized settings
        fig = px.scatter(
            filtered_data,
            x="Age",
            y="RV_value",
            color="Mileage",
            hover_name="Version_Name_jato",
            hover_data=["Year", "DatePriced", "Condition", "MSRP", "Mileage", "SoldPrice"],
            color_continuous_scale="Viridis",
            trendline="ols",
            trendline_options=dict(log_x=True),
            width=1100,
            height=600,
            title="Price Guide Model based on Transactions (2010-2021)"
        )

        # Configure x-axis ticks
        years = range(modelyear_selection + 1, modelyear_selection + 17)
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[52 * i for i in range(1, 17)],
                ticktext=list(years)
            ),
            xaxis_title="Year Sold",
            yaxis_title="Residual Value (%)",
            # Optimize rendering
            uirevision=True,
            template="plotly_white"
        )

        # Add prediction intervals
        results = px.get_trendline_results(fig)
        if not results.empty:
            try:
                x_range, base_pred = calculate_prediction_lines(results, filtered_data)
                
                for multiplier, color in [(1.1, "red"), (0.9, "red")]:
                    pred_vals = [v * multiplier for v in base_pred]
                    fig.add_trace(
                        go.Scatter(
                            x=list(x_range),
                            y=pred_vals,
                            mode="lines",
                            line=dict(color=color, dash='dash'),
                            showlegend=False
                        )
                    )
            except Exception as e:
                st.warning(f"Could not calculate prediction intervals: {str(e)}")

        # Display plot and data with optimized settings
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data with optimized settings
        st.dataframe(
            filtered_data,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("No data available for the selected filters.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check your data path and ensure all required columns are present in the dataset.")