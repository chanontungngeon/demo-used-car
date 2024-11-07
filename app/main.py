import streamlit as st
from data_handler import DataHandler
from utils import create_scatter_plot, add_prediction_intervals

# Page config
st.set_page_config(page_title='RV Value Prediction', page_icon=":car:", layout="wide")

# Title and description
st.title('Residual Value Prediction')
st.markdown("""
* Dataset: Data from 2010-2021
""")

def main():
    try:
        # Load data
        file_path = "merged_data_for_streamlit_new_15112021_filtered.csv"  # Update with your actual file path
        graphdata = DataHandler.load_data(file_path)

        # Sidebar filters
        st.sidebar.title('Select Parameters')
        
        make_selection = st.sidebar.selectbox(
            'Select Brand', 
            sorted(graphdata.Make_jato.unique().tolist())
        )
        
        model_mask = graphdata.Make_jato == make_selection
        model_selection = st.sidebar.selectbox(
            'Select Model',
            sorted(graphdata[model_mask].Model_I.unique().tolist())
        )
        
        trim_mask = model_mask & (graphdata.Model_I == model_selection)
        trim_selection = st.sidebar.selectbox(
            'Select Sub Model',
            sorted(graphdata[trim_mask].Version_Name_jato.unique().tolist())
        )
        
        year_mask = trim_mask & (graphdata.Version_Name_jato == trim_selection)
        modelyear_selection = st.sidebar.selectbox(
            'Select Model Year',
            sorted(graphdata[year_mask].Year.unique().tolist())
        )

        # Filter data
        filtered_data = DataHandler.filter_data(
            graphdata, 
            make_selection,
            model_selection, 
            trim_selection, 
            modelyear_selection
        )

        if not filtered_data.empty:
            # Display header information
            st.title('🚗 Residual Value Prediction')
            st.subheader(f"{make_selection} {model_selection} "
                        f"{trim_selection} {modelyear_selection}")
            st.subheader(f"MSRP: ${filtered_data.MSRP.iloc[0]:,.2f}")

            # Create and display plot
            try:
                fig = create_scatter_plot(filtered_data, modelyear_selection)
                fig = add_prediction_intervals(fig, filtered_data)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")
            
            # Display data
            st.dataframe(filtered_data, use_container_width=True)

        else:
            st.warning("No data available for the selected filters.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your data path and ensure all required columns are present in the dataset.")

if __name__ == "__main__":
    main()