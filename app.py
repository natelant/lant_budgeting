import streamlit as st
import pandas as pd
from function import process_csv_files, build_bottom_line, build_bar_chart, build_scatterplot, build_expense_tracker, build_expense_trends, build_top_expenses_rank, build_bottom_line_per_day, get_expense_categories, calculate_budget_comparison, build_budget_vs_actual_chart, style_budget_table, get_budget_summary_stats, get_default_budgets, get_default_budget_for_category
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

        # Create two columns for filters
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            # filter out transactions by ID
            st.subheader("Filter Transactions by ID")
            id_filter = st.text_input("Enter IDs (comma-separated)")
        
        with filter_col2:
            # filter out transactions by Description
            st.subheader("Filter Transactions by Description")
            desc_filter = st.text_input("Enter descriptions (comma-separated)")

        # Apply both filters
        filtered_df = merged_df.copy()  # Start with all data
        
        if id_filter:
            # Split the input string into a list and strip whitespace
            id_list = [int(id.strip()) for id in id_filter.split(',')]
            filtered_df = filtered_df[~filtered_df['ID'].isin(id_list)]
            
        if desc_filter:
            # Split the input string into a list and strip whitespace
            desc_list = [desc.strip() for desc in desc_filter.split(',')]
            # Filter out rows where Description contains any of the entered texts
            filtered_df = filtered_df[~filtered_df['Description'].str.contains('|'.join(desc_list), case=False, na=False)]
        
        # Display elements vertically
        st.subheader("All Transactions")
        scatterplot_expenses = build_scatterplot(filtered_df, 'Expense')
        scatterplot_income = build_scatterplot(filtered_df, 'Income')
        st.plotly_chart(scatterplot_expenses, use_container_width=True)
        st.plotly_chart(scatterplot_income, use_container_width=True)
        
        st.subheader("Monthly Summary")
        bar_chart = build_bar_chart(filtered_df)
        st.plotly_chart(bar_chart, use_container_width=True)

        st.subheader("Budget Analysis and Opportunity")
        
        # Budget Tracking Section
        st.subheader("Budget vs Actual Spending")
        
        # Month selection for budget analysis
        budget_months = sorted(filtered_df['Month'].unique(), reverse=True)
        budget_month = st.selectbox(
            "Select month for budget analysis",
            budget_months,
            index=0,
            key="budget_month_selector"
        )
        
        # Get expense categories for budget tracking
        expense_categories = get_expense_categories(filtered_df)
        
        # Create budget inputs
        st.write("Set your monthly budget for each category:")
        
        # Create columns for budget inputs
        budget_cols = st.columns(3)
        budget_values = {}
        
        for i, category in enumerate(expense_categories):
            col_idx = i % 3
            with budget_cols[col_idx]:
                # Get default value for this category
                default_value = get_default_budget_for_category(category)
                
                budget_values[category] = st.number_input(
                    f"{category} Budget",
                    min_value=0.0,
                    value=default_value,
                    step=100.0,
                    key=f"budget_{category}"
                )
        
        # Calculate budget comparison
        budget_df = calculate_budget_comparison(filtered_df, budget_month, budget_values)
        
        # Create and display budget vs actual chart
        if not budget_df.empty:
            budget_chart = build_budget_vs_actual_chart(budget_df, budget_month)
            if budget_chart:
                st.plotly_chart(budget_chart, use_container_width=True)
            
            # Display budget summary table
            st.subheader("Budget Summary")
            styled_budget_df = style_budget_table(budget_df)
            st.dataframe(styled_budget_df, use_container_width=True)
            
            # Summary statistics
            stats = get_budget_summary_stats(budget_df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Budget", f"${stats['total_budget']:,.2f}")
            with col2:
                st.metric("Total Actual", f"${stats['total_actual']:,.2f}")
            with col3:
                st.metric("Total Difference", f"${stats['total_difference']:,.2f}", 
                         delta_color="inverse" if stats['total_difference'] > 0 else "normal")
        
        st.subheader("Detailed Breakdown (Dollars Per Day)")
        # Add toggle button for dollars display type
        display_type = st.toggle("Show Total Dollars", value=False, help="Toggle between Total Dollars and Per Day Dollars")
        
        # Choose which function to call based on toggle state
        if display_type:  # If True, show total dollars
            bottom_line_df = build_bottom_line(filtered_df)
        else:  # If False (default), show per day dollars
            bottom_line_df = build_bottom_line_per_day(filtered_df)
        
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
            # Get unique expense categories only
            categories = filtered_df[filtered_df['Sorting Type'] == 'Expense']['Category'].unique()
            categories = sorted(categories[~np.isin(categories, ['INTERNAL_TRANSFER'])])
            
            # Set default index (first category if Groceries not found)
            try:
                default_category_idx = categories.index('Groceries')
            except ValueError:
                default_category_idx = 0
                
            expense_category = st.selectbox(
                "Select an expense category", 
                categories,
                index=default_category_idx
            )
            
        expense_tracker = build_expense_tracker(filtered_df, expense_category, month_selection)
        st.plotly_chart(expense_tracker, use_container_width=True)


        # Expense Trends
        st.subheader("Expense Trends")
        expense_trends = build_expense_trends(filtered_df, expense_category, month_selection)
        st.plotly_chart(expense_trends, use_container_width=True)

        # Top Expenses Rank
        st.subheader("Top Expenses Rank")
        top_expenses_rank = build_top_expenses_rank(filtered_df, month_selection, expense_category)
        st.plotly_chart(top_expenses_rank, use_container_width=True)

        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
else:
    st.info("Please upload your CSV files to begin analysis.")
