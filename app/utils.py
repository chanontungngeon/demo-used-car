import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, List

def create_scatter_plot(data: pd.DataFrame, modelyear: int) -> go.Figure:
    """Create scatter plot with trend lines"""
    fig = px.scatter(
        data,
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

    # Create x-axis ticks without using range()
    tick_values = [52 * i for i in range(1, 17)]  # This range is fine as it uses integers
    tick_text = [str(modelyear + i) for i in range(1, 17)]
    
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=tick_values,
            ticktext=tick_text
        ),
        xaxis_title="Year Sold",
        yaxis_title="Residual Value (%)"
    )

    return fig

def add_prediction_intervals(fig: go.Figure, filtered_data: pd.DataFrame) -> go.Figure:
    """Add prediction intervals to the plot"""
    try:
        # Get trendline results
        results = px.get_trendline_results(fig)
        
        if not results.empty:
            # Get coefficients
            coef = pd.DataFrame(
                results.iloc[0]["px_fit_results"].summary().tables[1].data[1:3],
                columns=results.iloc[0]["px_fit_results"].summary().tables[1].data[0]
            )[['coef', '[0.025', '0.975]']]

            coef_mean = coef.transpose().loc['coef', :].astype('float').tolist()
            
            # Get min and max values for x
            x_min = np.ceil(filtered_data.Age.min())
            x_max = np.floor(filtered_data.Age.max())
            
            # Create array of x values
            x_values = np.linspace(start=max(1, x_min), stop=x_max, num=100)
            
            # Calculate base predictions
            base_pred = coef_mean[0] + coef_mean[1] * np.log10(x_values)
            
            # Add prediction interval lines
            for multiplier, color in [(1.1, "red"), (0.9, "red")]:
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=base_pred * multiplier,
                        mode="lines",
                        line=dict(color=color, dash='dash'),
                        showlegend=False,
                        name=f"{multiplier*100}% Prediction Interval"
                    )
                )
                
    except Exception as e:
        print(f"Error adding prediction intervals: {e}")
    
    return fig