import streamlit as st
import pandas as pd
from function import process_csv_files, build_bottom_line, build_bar_chart, build_scatterplot, build_expense_tracker, build_expense_trends, build_top_expenses_rank, build_bottom_line_per_day, get_expense_categories, calculate_budget_comparison, build_budget_vs_actual_chart, style_budget_table, get_budget_summary_stats, get_default_budgets, get_default_budget_for_category, calculate_ytd_summary, search_vendor_transactions, search_category_transactions, identify_monthly_subscriptions, debug_vendor_analysis
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

        # Budget Inputs Section
        st.subheader("Set Your Monthly Budgets")
        st.write("Configure your monthly budget for each expense category:")
        
        # Initialize budget values in session state if not already present
        if 'budget_values' not in st.session_state:
            st.session_state.budget_values = {}
        
        # Get expense categories from the uploaded data
        expense_categories = get_expense_categories(merged_df)
        
        # Create columns for budget inputs
        budget_cols = st.columns(3)
        
        for i, category in enumerate(expense_categories):
            col_idx = i % 3
            with budget_cols[col_idx]:
                # Get default value for this category
                default_value = get_default_budget_for_category(category)
                
                # Use session state to persist values
                key = f"budget_{category}"
                if key not in st.session_state:
                    st.session_state[key] = default_value
                
                st.session_state.budget_values[category] = st.number_input(
                    f"{category} Budget",
                    min_value=0.0,
                    value=st.session_state[key],
                    step=100.0,
                    key=key
                )
        
        st.divider()

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
        
        # Use budget values from session state
        budget_values = st.session_state.budget_values
        
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
        
        # Category Transaction Search
        st.subheader("Category Transaction Details")
        st.write("Select a category to view all transactions for the selected month:")
        
        # Get expense categories for the dropdown
        expense_categories = get_expense_categories(filtered_df)
        
        # Create two columns for category selection
        col1, col2 = st.columns(2)
        with col1:
            selected_category = st.selectbox(
                "Select Category",
                expense_categories,
                key="category_transaction_selector"
            )
        
        with col2:
            # Show the selected month (read-only, matches budget month)
            st.text_input(
                "Selected Month",
                value=str(budget_month),
                disabled=True,
                key="category_month_display"
            )
        
        if selected_category:
            category_results = search_category_transactions(filtered_df, selected_category, budget_month)
            
            if not category_results.empty:
                st.write(f"Found {len(category_results)} transactions for '{selected_category}' in {budget_month}:")
                
                # Calculate summary statistics
                total_amount = category_results['Amount ($)'].sum()
                avg_amount = category_results['Amount ($)'].mean()
                
                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Transactions", len(category_results))
                with col2:
                    st.metric("Total Amount", f"${total_amount:,.2f}")
                with col3:
                    st.metric("Average Transaction", f"${avg_amount:,.2f}")
                
                # Display the transactions table
                st.dataframe(
                    category_results,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info(f"No transactions found for '{selected_category}' in {budget_month}.")
        
        # Budget analysis
        st.subheader("Expense Tracker")
        st.write(f"Tracking expenses for '{selected_category}' in {budget_month}:")
        
        # Get the budget value for the selected category
        category_budget = st.session_state.budget_values.get(selected_category, 0)
        
        expense_tracker = build_expense_tracker(filtered_df, selected_category, budget_month, category_budget)
        st.plotly_chart(expense_tracker, use_container_width=True)


        # Expense Trends
        st.subheader("Expense Trends")
        expense_trends = build_expense_trends(filtered_df, selected_category, budget_month)
        st.plotly_chart(expense_trends, use_container_width=True)

        # Top Expenses Rank
        st.subheader("Top Expenses Rank")
        top_expenses_rank = build_top_expenses_rank(filtered_df, budget_month, selected_category)
        st.plotly_chart(top_expenses_rank, use_container_width=True)

        # Vendor Search Section
        st.subheader("Vendor Search")
        st.write("Search for transactions by vendor name:")
        
        vendor_search = st.text_input(
            "Enter vendor name to search",
            placeholder="e.g., Walmart, Amazon, Shell",
            key="vendor_search"
        )
        
        if vendor_search:
            vendor_results = search_vendor_transactions(filtered_df, vendor_search)
            
            if not vendor_results.empty:
                st.write(f"Found {len(vendor_results)} transactions for '{vendor_search}':")
                
                # Calculate summary statistics
                total_amount = vendor_results['Amount ($)'].sum()
                avg_amount = vendor_results['Amount ($)'].mean()
                
                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Transactions", len(vendor_results))
                with col2:
                    st.metric("Total Amount", f"${total_amount:,.2f}")
                with col3:
                    st.metric("Average Transaction", f"${avg_amount:,.2f}")
                
                # Display the transactions table
                st.dataframe(
                    vendor_results,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info(f"No transactions found for '{vendor_search}'. Try a different search term.")

        # Year-to-Date Summary
        st.subheader("Year-to-Date Summary")
        ytd_stats = calculate_ytd_summary(filtered_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "YTD Total Income", 
                f"${ytd_stats['ytd_income']:,.2f}",
                help="Total income for the current year"
            )
        with col2:
            st.metric(
                "YTD Total Expenses", 
                f"${ytd_stats['ytd_expenses']:,.2f}",
                help="Total expenses for the current year"
            )
        with col3:
            st.metric(
                "Average Monthly Expenses", 
                f"${ytd_stats['avg_monthly_expenses']:,.2f}",
                help="Average monthly expenses for the current year"
            )
        
        # Detailed Breakdown (moved to bottom)
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
        
        # Monthly Subscriptions Section
        st.subheader("Monthly Subscriptions & Recurring Expenses")
        st.write("Identified recurring expenses and subscriptions based on transaction patterns:")
        
        subscriptions_df = identify_monthly_subscriptions(filtered_df)
        
        if not subscriptions_df.empty:
            # Calculate summary statistics
            total_monthly_cost = subscriptions_df[subscriptions_df['Subscription Type'] == 'Monthly Subscription']['Average Amount'].sum()
            total_recurring_cost = subscriptions_df['Average Amount'].sum()
            subscription_count = len(subscriptions_df)
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Subscriptions", subscription_count)
            with col2:
                st.metric("Monthly Cost", f"${total_monthly_cost:,.2f}")
            with col3:
                st.metric("Total Recurring Cost", f"${total_recurring_cost:,.2f}")
            
            # Format the display dataframe
            display_df = subscriptions_df.copy()
            display_df['Average Amount'] = display_df['Average Amount'].apply(lambda x: f"${x:,.2f}")
            display_df['Frequency'] = display_df['Frequency'].apply(lambda x: f"{x:.1f}/month")
            display_df['Amount Consistency'] = display_df['Amount Consistency'].apply(lambda x: f"{x:.1%}")
            display_df['Months Active'] = display_df['Months Active'].apply(lambda x: f"{x}/{display_df['Total Months'].iloc[0]}")
            
            # Rename columns for display
            display_df = display_df.rename(columns={
                'Vendor': 'Vendor',
                'Category': 'Category',
                'Average Amount': 'Avg Amount',
                'Frequency': 'Frequency',
                'Months Active': 'Months',
                'Amount Consistency': 'Consistency',
                'Subscription Type': 'Type',
                'Last Transaction': 'Last Transaction',
                'Source': 'Source'
            })
            
            # Select columns for display
            display_columns = ['Vendor', 'Category', 'Avg Amount', 'Type', 'Frequency', 'Months', 'Last Transaction', 'Source']
            display_df = display_df[display_columns]
            
            # Display the subscriptions table
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Add insights
            st.subheader("💡 Insights")
            if total_monthly_cost > 0:
                st.write(f"**Monthly Subscriptions Total: ${total_monthly_cost:,.2f}** - This is your guaranteed monthly expense for subscriptions.")
            
            if subscription_count > 0:
                st.write(f"**Total Recurring Expenses: ${total_recurring_cost:,.2f}** - Including quarterly and annual subscriptions.")
                
            # Show potential savings opportunities
            high_cost_subscriptions = subscriptions_df[subscriptions_df['Average Amount'] > 50]
            if not high_cost_subscriptions.empty:
                st.write("**💸 High-Cost Subscriptions (>$50/month):**")
                for _, sub in high_cost_subscriptions.iterrows():
                    st.write(f"  • {sub['Vendor']}: ${sub['Average Amount']:,.2f}/month")
        
        else:
            st.info("No recurring expenses or subscriptions detected in your transaction data.")
        
        # Debug section for troubleshooting
        with st.expander("🔍 Debug: Check why a specific vendor wasn't detected"):
            debug_vendor = st.text_input(
                "Enter vendor name to analyze",
                placeholder="e.g., Audible, Netflix, Spotify",
                key="debug_vendor_input"
            )
            
            if debug_vendor:
                debug_result = debug_vendor_analysis(filtered_df, debug_vendor)
                
                if isinstance(debug_result, dict):
                    st.write("**Analysis Results:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Vendor:** {debug_result['Vendor']}")
                        st.write(f"**Total Transactions:** {debug_result['Total Transactions']}")
                        st.write(f"**Unique Months:** {debug_result['Unique Months']}")
                        st.write(f"**Total Months in Data:** {debug_result['Total Months in Data']}")
                        st.write(f"**Frequency:** {debug_result['Frequency']:.2f}/month")
                    
                    with col2:
                        st.write(f"**Average Amount:** ${debug_result['Average Amount']:.2f}")
                        st.write(f"**Amount Consistency:** {debug_result['Amount Consistency']:.1%}")
                        st.write(f"**Months Criteria (≥50%):** {'✅' if debug_result['Months Criteria (≥50%)'] else '❌'}")
                        st.write(f"**Frequency Criteria (≥0.8):** {'✅' if debug_result['Frequency Criteria (≥0.8)'] else '❌'}")
                        st.write(f"**Consistency Criteria (<30%):** {'✅' if debug_result['Consistency Criteria (<30%)'] else '❌'}")
                    
                    if debug_result['Would be detected as subscription']:
                        st.success("✅ This vendor WOULD be detected as a subscription with current criteria")
                    else:
                        st.warning("❌ This vendor would NOT be detected as a subscription")
                        
                        # Show what needs to change
                        issues = []
                        if not debug_result['Months Criteria (≥50%)']:
                            issues.append(f"Needs to appear in more months (currently {debug_result['Unique Months']}, needs at least {max(2, debug_result['Total Months in Data'] * 0.5)})")
                        if not debug_result['Frequency Criteria (≥0.8)']:
                            issues.append(f"Frequency too low (currently {debug_result['Frequency']:.2f}, needs at least 0.8)")
                        if not debug_result['Consistency Criteria (<30%)']:
                            issues.append(f"Amount too inconsistent (currently {debug_result['Amount Consistency']:.1%}, needs less than 30%)")
                        
                        if issues:
                            st.write("**Issues preventing detection:**")
                            for issue in issues:
                                st.write(f"• {issue}")
                else:
                    st.write(debug_result)
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
else:
    st.info("Please upload your CSV files to begin analysis.")
