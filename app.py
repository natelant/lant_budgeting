import streamlit as st
import pandas as pd
from function import process_csv_files, build_bottom_line, build_bar_chart
import os
from tempfile import mkdtemp
import shutil

st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.title("Financial Dashboard")

# Cache the data processing
@st.cache_data
def process_data(temp_folder):
    return process_csv_files(temp_folder)

# File uploader section
st.subheader("Upload Your Financial Data")
uploaded_files = st.file_uploader(
    "Upload your CSV files (Checking and Credit Card statements)",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:
    # Create a temporary directory to store uploaded files
    temp_dir = mkdtemp()
    
    try:
        # Save uploaded files to temporary directory
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Process the data
        merged_df = process_data(temp_dir)
        
        # Create two columns for the dashboard
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Monthly Summary")
            # Create and display the bar chart
            bar_chart = build_bar_chart(merged_df)
            st.plotly_chart(bar_chart, use_container_width=True)
        
        with col2:
            st.subheader("Detailed Breakdown")
            # Create and display the summary table
            bottom_line_df = build_bottom_line(merged_df)
            st.dataframe(bottom_line_df, use_container_width=True)
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
else:
    st.info("Please upload your CSV files to begin analysis.")
