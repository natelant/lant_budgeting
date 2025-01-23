import streamlit as st
import pandas as pd
from function import process_csv_files, build_bottom_line, build_bar_chart, build_scatterplot, build_expense_tracker, build_expense_trends, build_top_expenses_rank
import os
import plotly.express as px
from tempfile import mkdtemp
import shutil
import numpy as np

st.set_page_config(
    page_title="Financial Dashboard", 
    page_icon=":money_with_wings:",
    layout="wide"
)
st.title("Lant Family Financial Dashboard")

# Cache the data processing
@st.cache_data
def process_data(temp_folder):
    return process_csv_files(temp_folder)

# File uploader section
st.subheader("Upload JP Morgan Files")
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

        # filter out transactions by ID
        st.subheader("Filter Transactions by ID")
        id_filter = st.text_input("Enter IDs (comma-separated)")
        
        if id_filter:
            # Split the input string into a list and strip whitespace
            id_list = [int(id.strip()) for id in id_filter.split(',')]
            filtered_df = merged_df[~merged_df['ID'].isin(id_list)]
        else:
            filtered_df = merged_df.copy()  # Show all data if no filter
        
        # Display elements vertically
        st.subheader("All Transactions")
        scatterplot_expenses = build_scatterplot(filtered_df, 'Expense')
        scatterplot_income = build_scatterplot(filtered_df, 'Income')
        st.plotly_chart(scatterplot_expenses, use_container_width=True)
        st.plotly_chart(scatterplot_income, use_container_width=True)
        
        st.subheader("Monthly Summary")
        bar_chart = build_bar_chart(filtered_df)
        st.plotly_chart(bar_chart, use_container_width=True)
        
        st.subheader("Detailed Breakdown")
        bottom_line_df = build_bottom_line(filtered_df)
        
        # Calculate averages excluding the most recent month and outliers
        most_recent_month = bottom_line_df.columns[-1]  # Last column is most recent
        historical_df = bottom_line_df.drop(columns=[most_recent_month])
        
        # Function to calculate average excluding outliers for a series
        def get_average_without_outliers(series):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return series[(series >= lower_bound) & (series <= upper_bound)].mean()
        
        # Calculate averages excluding outliers for each category
        category_averages = historical_df.apply(lambda row: get_average_without_outliers(row), axis=1)
        
        # Function to color cells based on comparison with average
        def color_cells(val):
            if isinstance(val, pd.Series):
                return [''] * len(val)
            if pd.isna(val):
                return ''
            try:
                # Convert value to float for comparison
                val = float(val)
                # Get the category (row) name from the current cell's index
                category = bottom_line_df.index[bottom_line_df.eq(val).any(axis=1)].values[0]
                avg = category_averages[category]
                
                # Calculate percentage difference from average
                pct_diff = (val - avg) / avg if avg != 0 else 0
                
                # Determine if this is an income category (checking the first element of the tuple)
                is_income = category[0] == 'Income' or (category[0] == 'Monthly Totals' and category[1] == 'Income')
                
                # Convert to RGB color
                # For Income: green for above average, red for below
                # For Expenses: red for above average, green for below
                if (pct_diff > 0) != is_income:  # XOR operation
                    intensity = min(abs(pct_diff) * 0.5, 1)  # Scale the intensity
                    return f'background-color: rgba(255, 0, 0, {intensity})'
                else:
                    intensity = min(abs(pct_diff) * 0.5, 1)  # Scale the intensity
                    return f'background-color: rgba(0, 255, 0, {intensity})'
            except Exception as e:
                return ''
        
        # Debug: Print unique categories
        print("Categories in DataFrame:", bottom_line_df.index.tolist())
        
        # Apply styling and format numbers
        styled_df = bottom_line_df.style\
            .format("${:,.2f}")\
            .applymap(color_cells)
        
        st.dataframe(styled_df, use_container_width=True)

        

        # Budget analysis
        st.subheader("Expense Tracker")
        col1, col2 = st.columns(2)  # Create two equal-width columns
        with col1:
            months = sorted(filtered_df['Month'].unique(), reverse=True)
            month_selection = st.selectbox(
                "Select a month",
                months,
                index=0  # Most recent month
            )

        with col2:
            categories = filtered_df['Category'].unique()
            default_category_idx = np.where(categories == 'Groceries')[0][0]
            expense_category = st.selectbox(
                "Select an expense category", 
                categories,
                index=int(default_category_idx)
            )
            
        expense_tracker = build_expense_tracker(filtered_df, expense_category, month_selection)
        st.plotly_chart(expense_tracker, use_container_width=True)


        # Expense Trends
        st.subheader("Expense Trends")
        expense_trends = build_expense_trends(filtered_df, expense_category, month_selection)
        st.plotly_chart(expense_trends, use_container_width=True)

        # Top Expenses Rank
        st.subheader("Top Expenses Rank")
        top_expenses_rank = build_top_expenses_rank(filtered_df, month_selection)
        st.plotly_chart(top_expenses_rank, use_container_width=True)

        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
else:
    st.info("Please upload your CSV files to begin analysis.")
