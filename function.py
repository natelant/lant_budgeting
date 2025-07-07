import pandas as pd
import os
import plotly.express as px
from datetime import datetime
import numpy as np


# Functions --------------------------------------------------------------------------
# clean the checking account df
def clean_checking_account(df):
    # create category column based off of conditions
    df['Category'] = df.apply(categorize_transaction, axis=1)
    df['Sorting Type'] = df['Amount'].apply(lambda x: 'Income' if x > 0 else 'Expense')

    # remove and rename columns
    df = df.drop(columns=['Details', 'Check or Slip #', 'Balance'])
    df = df.rename(columns={'Posting Date': 'Transaction Date'})
    return df

def categorize_transaction(row):
    description = str(row['Description'])  # Convert to string
    amount = row['Amount']
    
    # Define rent amount (you can make this a parameter if it changes)
    rent = 1400
    
    # Categorization rules
    if pd.notna(description):  # Check if description is not NaN
        if 'JesusChrist' in description:
            return 'Tithing & Fast Offering'
        elif 'VENMO' in description and abs(amount) == rent:
            return 'Rent'
        elif 'VENMO' in description:
            return 'Venmo'
        elif 'External F' in description:
            return 'Automotive'
        elif 'AVENUE' in description:
            return 'PAY'
        elif 'DEPOSIT #' in description:
            return 'Check'
        
        elif any(term in description for term in ['Chase card', 'Transfer From ', 'Transfer to ', 'Transfer from', 'WELLS FARGO', 'Goldman']):
            return 'INTERNAL_TRANSFER'
    # Add more conditions as needed
    return 'Uncategorized'


def process_csv_files(folder_path):
    checking_account_dfs = []  # New list to store multiple checking account DFs
    credit_card_dfs = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.CSV'):
            file_path = os.path.join(folder_path, filename)
            
            if 'Chase5121' in filename:
                # Read checking account file
                df = pd.read_csv(file_path, index_col=False)
                
                # If the first column is unnamed, give it a name
                if df.columns[0] == 'Unnamed: 0':
                    df = df.rename(columns={'Unnamed: 0': 'Transaction_Type'})
                
                checking_account_dfs.append(df)  # Append to list instead of direct assignment
                
            else:
                # This is a credit card file
                credit_card_df = pd.read_csv(file_path)
                credit_card_df = credit_card_df.drop(columns=['Post Date', 'Memo'])
                # Filter out the credit card payments
                credit_card_df = credit_card_df[credit_card_df['Type'] != 'Payment']
                credit_card_df['Sorting Type'] = 'Expense'
                
                # Adjust categories for each credit card DataFrame
                credit_card_df['Category'] = credit_card_df.apply(lambda row: 'Groceries' if 'WALMART' in str(row['Description']) else row['Category'], axis=1)
                
                credit_card_dfs.append(credit_card_df)

    # Combine all checking account dataframes and clean the combined result
    if checking_account_dfs:
        checking_account_df = pd.concat(checking_account_dfs, ignore_index=True)
        checking_account_df = clean_checking_account(checking_account_df)
    else:
        checking_account_df = pd.DataFrame()  # Empty DataFrame if no checking files found

    # Combine all credit card dataframes
    combined_credit_card_df = pd.concat(credit_card_dfs, ignore_index=True)

    # Add source column to each dataframe before merging
    if not checking_account_df.empty:
        checking_account_df['Source'] = 'Checking Account'
    if not combined_credit_card_df.empty:
        combined_credit_card_df['Source'] = 'Credit Card'
    
    # Merge checking account and credit card data
    # Note: You may need to adjust the merge parameters based on your data structure
    merged_df = pd.concat([checking_account_df, combined_credit_card_df], ignore_index=True)

    # Rearrange columns
    merged_df = merged_df[['Transaction Date', 'Sorting Type', 'Category', 'Amount', 'Description', 'Source']]
    # Convert 'Transaction Date' to datetime and create 'Month' column
    merged_df['Transaction Date'] = pd.to_datetime(merged_df['Transaction Date'])
    merged_df['Month'] = merged_df['Transaction Date'].dt.to_period('M')
    merged_df['ID'] = merged_df.index

    return merged_df


def build_bottom_line(merged_df): 
    # Filter out the INTERNAL_TRANSFER category 
    filtered_df = merged_df[~merged_df['Category'].isin(['INTERNAL_TRANSFER'])]

    # Create separate DataFrames for Income and Expenses
    income_df = filtered_df[filtered_df['Sorting Type'] == 'Income']
    expense_df = filtered_df[filtered_df['Sorting Type'] == 'Expense']

    monthly_totals = filtered_df.groupby(['Month', 'Sorting Type']).sum('Amount').abs().reset_index()
    monthly_totals = monthly_totals.pivot(index='Month', columns='Sorting Type', values='Amount')
    monthly_totals['Balance'] = monthly_totals['Income'] - monthly_totals['Expense']
    # gether the data back into long format
    # Reset index to make Month a column and melt the dataframe
    long_format_df = monthly_totals.reset_index().melt(
        id_vars=['Month'],
        value_vars=['Expense', 'Income', 'Balance'],
        var_name='Sorting Type',
        value_name='Amount'
    )

    # Define the desired order of Sorting Type
    sorting_type_order = ['Income', 'Expense', 'Balance']
    
    # Create pivot and reindex rows to match desired order
    monthly_pivot = long_format_df.pivot(index='Sorting Type', columns='Month', values='Amount')\
                                .reindex(sorting_type_order)
    
    # Get the ordered list of months from monthly_pivot
    ordered_months = monthly_pivot.columns.tolist()
    
    # Create pivot tables for Income and Expenses and reindex columns to match monthly_pivot
    income_pivot = pd.pivot_table(income_df, values='Amount', index='Category', 
                                 columns='Month', aggfunc='sum').reindex(columns=ordered_months)
    expense_pivot = pd.pivot_table(expense_df, values='Amount', index='Category', 
                                  columns='Month', aggfunc='sum').reindex(columns=ordered_months)
    
    # concatenate the pivot tables
    pivot_table = pd.concat([
        pd.concat([income_pivot], keys=['Income']),
        pd.concat([expense_pivot], keys=['Expenses']),
        pd.concat([monthly_pivot], keys=['Monthly Totals'])
    ])

    return pivot_table


def build_bar_chart(merged_df):
    # Filter out the INTERNAL_TRANSFER category 
    filtered_df = merged_df[~merged_df['Category'].isin(['INTERNAL_TRANSFER'])]

    # get the monthly totals
    monthly_totals = filtered_df.groupby(['Month', 'Sorting Type']).sum('Amount').abs().reset_index()
    monthly_totals = monthly_totals.pivot(index='Month', columns='Sorting Type', values='Amount')
    monthly_totals['Balance'] = monthly_totals['Income'] - monthly_totals['Expense']
    
    # Convert Period index to strings
    monthly_totals.index = monthly_totals.index.astype(str)

    # build the bar chart
    fig = px.bar(monthly_totals, x=monthly_totals.index, y=['Income', 'Expense', 'Balance'], 
                 title='Monthly Totals', 
                 labels={'x':'Month', 'value':'Amount'}, 
                 barmode='group',
                 color_discrete_map={
                     'Income': '#2E86C1',  # Blue
                     'Expense': '#E74C3C',  # Red
                     'Balance': '#27AE60'   # Green
                 })
    
    # Update layout with correct tick configuration
    fig.update_layout(
        xaxis=dict(
            type='category',  # Treat x-axis as categorical
            tickangle=45,     # Rotate labels 45 degrees
            tickmode='array',
            tickvals=monthly_totals.index
        )
    )
    return fig

def build_timeseries(merged_df, income_or_expenses, month, category):
    # Filter data
    filtered_df = merged_df[merged_df['Sorting Type'] == income_or_expenses]
    filtered_df = filtered_df[filtered_df['Month'] == month]
    filtered_df = filtered_df[filtered_df['Category'] == category]
    
    # Sort by date and take absolute value of Amount
    filtered_df = filtered_df.sort_values('Transaction Date')
    filtered_df['Amount'] = filtered_df['Amount'].abs()
    filtered_df['Cumulative'] = filtered_df['Amount'].cumsum()

        # Get the first and last day of the month
    first_day = pd.to_datetime(month + '-01')
    last_day = (first_day + pd.offsets.MonthEnd(0))
    
    # Create custom week bins starting from the 1st Monday
    weekly_sums = filtered_df.groupby(pd.Grouper(
        key='Transaction Date',
        freq='W-MON',  # Weeks start on Monday
        origin='start_day'  # Align with calendar weeks
    ))['Amount'].sum()
    
    # Filter out weeks that aren't complete or aren't in our month
    weekly_sums = weekly_sums[
        (weekly_sums.index.strftime('%Y-%m') == month) & 
        (weekly_sums.index + pd.Timedelta(days=6) <= last_day)
    ]
    
    # Define color based on transaction type
    color = '#2E86C1' if income_or_expenses == 'Income' else '#E74C3C'
    
    # Create base scatter plot FIRST
    fig = px.scatter(
        filtered_df,
        x='Transaction Date',
        y='Amount',
        title=f'{category} Transactions for {month}',
        labels={'Transaction Date': 'Date', 'Amount': f'Amount ($)'},
        hover_data=['Description']
    )
    
    # Update scatter points color
    fig.update_traces(marker=dict(color=color, size=10))
    
    # Add cumulative line
    fig.add_scatter(
        x=filtered_df['Transaction Date'],
        y=filtered_df['Cumulative'],
        name='Running Total',
        line=dict(dash='dot', color=color),
        yaxis='y2',
        hovertemplate="Running Total: $%{y:.2f}<br>"
    )
    
    # Now add the horizontal lines for weekly sums
    for week_date, week_sum in weekly_sums.items():
        week_end = week_date + pd.Timedelta(days=6)
        
        # Add weekly sum line
        fig.add_shape(
            type="line",
            x0=week_date,
            x1=week_end,
            y0=week_sum,
            y1=week_sum,
            yref='y2',
            line=dict(
                color=color,
                width=2,
                dash="dot",
            ),
            opacity=0.7
        )
        # Add weekly sum annotation
        fig.add_annotation(
            x=week_end,
            y=week_sum,
            yref='y2',
            text=f"Week of {week_date.strftime('%m/%d')}: ${week_sum:.2f}",
            showarrow=False,
            xanchor="right",
            yanchor="bottom"
        )
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        xaxis_tickangle=45,
        yaxis2=dict(
            title='Cumulative Amount ($)',
            overlaying='y',
            side='right'
        ),
        yaxis_title='Transaction Amount ($)',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def build_scatterplot(merged_df, category):
    # Filter out the INTERNAL_TRANSFER category 
    filtered_df = merged_df[~merged_df['Category'].isin(['INTERNAL_TRANSFER'])].copy()

    expense_df = filtered_df[filtered_df['Sorting Type'] == category].copy()
    # Multiply by -1 to invert the signs while preserving reimbursements
    if category == 'Expense':
        expense_df['Amount'] = expense_df['Amount'] * -1
    elif category == 'Income':
        expense_df['Amount'] = expense_df['Amount']

    # build scatter plot of all transactions with color by category
    fig = px.scatter(
        expense_df, 
        x='Transaction Date', 
        y='Amount', 
        color='Category',
        title=f'All {category}s', 
        labels={'Transaction Date': 'Date', 'Amount': 'Amount ($)'}, 
        hover_data=['Description', 'ID'])

    return fig
    

def build_expense_tracker(merged_df, category, month, budget=None):
    # Filter data for the selected category
    filtered_df = merged_df[merged_df['Sorting Type'] == 'Expense']
    filtered_df = filtered_df[filtered_df['Category'] == category]
    
    # Get current month's data
    month_df = filtered_df[filtered_df['Month'] == month].copy()
    month_df['Day'] = month_df['Transaction Date'].dt.day
    month_df['Amount'] = month_df['Amount'] * -1  # Invert signs while preserving refunds
    month_df = month_df.sort_values('Day')
    month_df['Cumulative'] = month_df['Amount'].cumsum()
    
    # Calculate linear historical average trend
    historical_df = filtered_df[filtered_df['Month'] < month].copy()
    if not historical_df.empty:
        # Calculate monthly totals with inverted signs
        monthly_totals = (historical_df['Amount'] * -1).groupby(historical_df['Month']).sum()
        
        # Calculate average excluding outliers
        Q1 = monthly_totals.quantile(0.25)
        Q3 = monthly_totals.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_totals = monthly_totals[(monthly_totals >= lower_bound) & 
                                       (monthly_totals <= upper_bound)]
        avg_monthly_total = filtered_totals.mean()
        
        # Get number of days in the current month using the Period object
        days_in_month = month.days_in_month
        
        # Create linear trend line
        daily_avg = avg_monthly_total / days_in_month
        avg_by_day = pd.DataFrame({
            'Day': range(1, days_in_month + 1),
            'Cumulative': [daily_avg * day for day in range(1, days_in_month + 1)]
        })
    
    # Create the plot
    fig = px.scatter(
        month_df,
        x='Day',
        y='Cumulative',
        title=f'{category} Expense Tracker for {month}',
        labels={'Day': 'Day of Month', 'Cumulative': 'Cumulative Amount ($)'},
        hover_data=['Description', 'Amount']
    )
    
    # Update scatter points
    fig.update_traces(
        marker=dict(color='#E74C3C', size=10),
        name='Current Month'
    )
    
    # Add line connecting the points
    fig.add_scatter(
        x=month_df['Day'],
        y=month_df['Cumulative'],
        mode='lines',
        line=dict(color='#E74C3C', dash='dot'),
        showlegend=False
    )
    
    # Add historical average line if data exists
    if not historical_df.empty:
        fig.add_scatter(
            x=avg_by_day['Day'],
            y=avg_by_day['Cumulative'],
            mode='lines',
            name='Historical Average',
            line=dict(color='#2E86C1', dash='dash')
        )
    
    # Add budget line if budget is provided
    if budget is not None and budget > 0:
        days_in_month = month.days_in_month
        daily_budget = budget / days_in_month
        budget_by_day = pd.DataFrame({
            'Day': range(1, days_in_month + 1),
            'Cumulative': [daily_budget * day for day in range(1, days_in_month + 1)]
        })
        
        fig.add_scatter(
            x=budget_by_day['Day'],
            y=budget_by_day['Cumulative'],
            mode='lines',
            name='Budget',
            line=dict(color='#27AE60', width=3)  # Solid green line
        )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            tickangle=45
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def build_expense_trends(merged_df, category, exclude_month):
    # Filter data for the selected category and expenses only
    filtered_df = merged_df[
        (merged_df['Sorting Type'] == 'Expense') & 
        (merged_df['Category'] == category)
    ].copy()
    
    # Calculate monthly totals (multiply by -1 to make expenses positive)
    monthly_totals = filtered_df.groupby('Month')['Amount'].sum() * -1
    
    # Calculate average excluding specified month and outliers
    historical_data = monthly_totals[monthly_totals.index != exclude_month]
    
    # Calculate average excluding outliers
    Q1 = historical_data.quantile(0.25)
    Q3 = historical_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = historical_data[(historical_data >= lower_bound) & 
                                  (historical_data <= upper_bound)]
    historical_avg = filtered_data.mean()
    
    # Create the bar chart
    fig = px.bar(
        x=monthly_totals.index.astype(str),
        y=monthly_totals.values,
        title=f'{category} Monthly Expenses',
        labels={'x': 'Month', 'y': 'Total Amount ($)'}
    )
    
    # Update bar colors (excluded month in red, others in blue)
    fig.update_traces(
        marker_color=[
            '#E74C3C' if m == exclude_month else '#2E86C1' 
            for m in monthly_totals.index
        ]
    )
    
    # Add horizontal average line
    fig.add_hline(
        y=historical_avg,
        line_dash="dash",
        line_color="#27AE60",
        annotation_text=f"Historical Average: ${historical_avg:.2f}",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_tickangle=45,
        showlegend=False
    )
    
    return fig


def standardize_merchant_name(description):
    """Standardize merchant names to group similar entries."""
    description = str(description).upper()  # Convert to uppercase for consistency
    
    # Handle Audible transactions with unique IDs
    if 'AUDIBLE*' in description:
        return 'AUDIBLE'
    
    # Dictionary of merchant name mappings
    merchant_mappings = {
        'WALMART': ['WAL-MART', 'WALMART.COM', 'WAL MART', 'WALMART #'],
        'AMAZON': ['AMAZON.COM', 'AMZN', 'AMAZON PRIME', 'AMAZON DIGITAL'],
        'COSTCO': ['COSTCO WHSE', 'COSTCO GAS', 'COSTCO WHOLESALE'],
        'SMITHS': ['SMITHS FOOD', "SMITH'S", 'SMITHS #'],
        'JESUSCHRIST DONATION': ['JESUSCHRIST', 'DONATION'],
        'CHEVRON': ['CHEVRON'],
        'MAVERIK': ['MAVERIK'],
        '7-ELEVEN': ['7-ELEVEN'],
        'SPEEDWAY': ['SPEEDWAY'],
        'SHELL': ['SHELL'],
        'CVS': ['CVS/PHARMACY'],
        'TRADER JOES': ['TRADER JOE'],
        'CHICK-FIL-A': ['CHICK-FIL-A'],
        'WENDYS': ['WENDYS'],
        'COSTA VIDA': ['COSTA VIDA'],
        'CAFE RIO': ['CAFE RIO'],
        'JIMMY JOHNS': ['JIMMY JOHNS'],
        'PHILLIPS 66': ['PHILLIPS 66'],
        'ZUPAS': ['ZUPAS'],
        'AUBERGINE': ['AUBERGINE'],
        'DOORDASH': ['DOORDASH'],

        # Add more mappings as needed
    }
    
    # Check each merchant group
    for standard_name, variations in merchant_mappings.items():
        if any(variant in description for variant in variations + [standard_name]):
            return standard_name
            
    return description

def build_top_expenses_rank(merged_df, month, category):
    # Filter data for the selected month and expenses only
    filtered_df = merged_df[(merged_df['Month'] == month) & 
                           (merged_df['Sorting Type'] == 'Expense') &
                           (~merged_df['Category'].isin(['INTERNAL_TRANSFER', 'Venmo']))].copy()
    
    # Standardize merchant names
    filtered_df['Standardized_Description'] = filtered_df['Description'].apply(standardize_merchant_name)
    
    # Create a DataFrame with description totals and categories
    description_data = filtered_df.groupby('Standardized_Description').agg({
        'Amount': lambda x: x.sum() * -1,
        'Category': 'first',  # Get the category for each description
        'ID': 'count'  # Count number of transactions
    }).reset_index()
    
    # Sort by amount and get top 20
    description_data = description_data.sort_values('Amount', ascending=True).tail(30)
    
    # Create color array
    colors = ['#2E86C1' if cat == category else '#E74C3C' 
             for cat in description_data['Category']]
    
    # Create bar chart
    fig = px.bar(
        x=description_data['Amount'],
        y=description_data['Standardized_Description'],
        orientation='h',  # horizontal bars
        title=f'Top 20 Expenses by Merchant for {month}',
        labels={
            'x': 'Amount ($)',
            'y': 'Merchant',
            'Standardized_Description': 'Merchant'
        },
        custom_data=[description_data['Category'], description_data['ID']]  # Add category and transaction count data for hover
    )
    
    # Update traces with custom colors and hover template
    fig.update_traces(
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>" +
                      "Amount: $%{x:,.2f}<br>" +
                      "Category: %{customdata[0]}<br>" +
                      "Transactions: %{customdata[1]}" +
                      "<extra></extra>"  # Removes trace name from hover
    )
    
    # Update layout to ensure all labels are visible
    fig.update_layout(
        height=600,  # Fixed height that works well for 20 items
        margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins
        yaxis=dict(
            tickmode='linear',  # Show all ticks
            tickangle=0,  # Keep labels horizontal
        )
    )
    
    return fig

def build_bottom_line_per_day(merged_df): 
    # Filter out the INTERNAL_TRANSFER category 
    filtered_df = merged_df[~merged_df['Category'].isin(['INTERNAL_TRANSFER'])]

    # Convert Transaction Date column to datetime if it's not already
    filtered_df['Transaction Date'] = pd.to_datetime(filtered_df['Transaction Date'])

    # Calculate days in each month
    month_days = {}
    for month in filtered_df['Month'].unique():
        month_data = filtered_df[filtered_df['Month'] == month]
        if month == filtered_df['Month'].max():  # Most recent month
            # Calculate actual days in the data range
            days = (month_data['Transaction Date'].max() - month_data['Transaction Date'].min()).days + 1
        else:
            # Get the number of days in the full month
            days = pd.Period(month).days_in_month
        month_days[month] = days

    # Create separate DataFrames for Income and Expenses
    income_df = filtered_df[filtered_df['Sorting Type'] == 'Income']
    expense_df = filtered_df[filtered_df['Sorting Type'] == 'Expense']

    # Calculate monthly totals and divide by days in month
    monthly_totals = filtered_df.groupby(['Month', 'Sorting Type']).sum('Amount').abs().reset_index()
    monthly_totals['Amount'] = monthly_totals.apply(
        lambda row: row['Amount'] / month_days[row['Month']], 
        axis=1
    )
    
    monthly_totals = monthly_totals.pivot(index='Month', columns='Sorting Type', values='Amount')
    monthly_totals['Balance'] = monthly_totals['Income'] - monthly_totals['Expense']
    
    # Convert to long format
    long_format_df = monthly_totals.reset_index().melt(
        id_vars=['Month'],
        value_vars=['Expense', 'Income', 'Balance'],
        var_name='Sorting Type',
        value_name='Amount'
    )

    # Define the desired order of Sorting Type
    sorting_type_order = ['Income', 'Expense', 'Balance']
    
    # Create pivot and reindex rows to match desired order
    monthly_pivot = long_format_df.pivot(index='Sorting Type', columns='Month', values='Amount')\
                                .reindex(sorting_type_order)
    
    # Get the ordered list of months from monthly_pivot
    ordered_months = monthly_pivot.columns.tolist()
    
    # Create pivot tables for Income and Expenses, divide by days, and reindex columns
    income_pivot = pd.pivot_table(income_df, values='Amount', index='Category', 
                                 columns='Month', aggfunc='sum').reindex(columns=ordered_months)
    expense_pivot = pd.pivot_table(expense_df, values='Amount', index='Category', 
                                  columns='Month', aggfunc='sum').reindex(columns=ordered_months)
    
    # Divide values by days in month
    for month in ordered_months:
        if month in income_pivot.columns:
            income_pivot[month] = income_pivot[month] / month_days[month]
        if month in expense_pivot.columns:
            expense_pivot[month] = expense_pivot[month] / month_days[month]
    
    # concatenate the pivot tables
    pivot_table = pd.concat([
        pd.concat([income_pivot], keys=['Income']),
        pd.concat([expense_pivot], keys=['Expenses']),
        pd.concat([monthly_pivot], keys=['Monthly Totals'])
    ])

    return pivot_table


def get_expense_categories(merged_df):
    """Get unique expense categories for budget tracking."""
    expense_categories = merged_df[merged_df['Sorting Type'] == 'Expense']['Category'].unique()
    expense_categories = sorted(expense_categories[~np.isin(expense_categories, ['INTERNAL_TRANSFER'])])
    return expense_categories


def calculate_budget_comparison(merged_df, budget_month, budget_values):
    """Calculate budget vs actual spending comparison."""
    # Calculate actual spending for selected month
    month_data = merged_df[merged_df['Month'] == budget_month]
    actual_spending = month_data[month_data['Sorting Type'] == 'Expense'].groupby('Category')['Amount'].sum()
    
    # Get expense categories
    expense_categories = get_expense_categories(merged_df)
    
    # Create budget comparison dataframe
    budget_comparison = []
    for category in expense_categories:
        actual = abs(actual_spending.get(category, 0))  # Make actual spending positive
        budget = budget_values.get(category, 0)
        difference = actual - budget  # Now difference is actual (positive) - budget
        budget_comparison.append({
            'Category': category,
            'Budget': budget,
            'Actual': actual,
            'Difference': difference,
            'Over/Under': 'Over' if difference > 0 else 'Under' if difference < 0 else 'On Target'
        })
    
    return pd.DataFrame(budget_comparison)


def build_budget_vs_actual_chart(budget_df, budget_month):
    """Create budget vs actual spending bar chart."""
    if budget_df.empty:
        return None
    
    # Prepare data for plotting
    plot_data = []
    for _, row in budget_df.iterrows():
        plot_data.extend([
            {'Category': row['Category'], 'Amount': row['Budget'], 'Type': 'Budget'},
            {'Category': row['Category'], 'Amount': row['Actual'], 'Type': 'Actual'}
        ])
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the bar chart
    fig = px.bar(
        plot_df,
        x='Category',
        y='Amount',
        color='Type',
        barmode='group',
        title=f'Budget vs Actual Spending - {budget_month}',
        color_discrete_map={'Budget': '#1f77b4', 'Actual': '#ff7f0e'}
    )
    
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Amount ($)",
        showlegend=True
    )
    
    return fig


def style_budget_table(budget_df):
    """Style the budget dataframe to highlight over/under budget."""
    def color_difference(val):
        if pd.isna(val):
            return ''
        try:
            val = float(val)
            if val > 0:
                return 'background-color: rgba(255, 0, 0, 0.3)'  # Red for over budget
            elif val < 0:
                return 'background-color: rgba(0, 255, 0, 0.3)'  # Green for under budget
            else:
                return 'background-color: rgba(128, 128, 128, 0.3)'  # Gray for on target
        except:
            return ''
    
    return budget_df.style\
        .format({'Budget': '${:,.2f}', 'Actual': '${:,.2f}', 'Difference': '${:,.2f}'})\
        .applymap(color_difference, subset=['Difference'])


def get_budget_summary_stats(budget_df):
    """Get summary statistics for budget analysis."""
    total_budget = budget_df['Budget'].sum()
    total_actual = budget_df['Actual'].sum()
    total_difference = total_actual - total_budget
    
    return {
        'total_budget': total_budget,
        'total_actual': total_actual,
        'total_difference': total_difference
    }


def calculate_ytd_summary(merged_df):
    """Calculate year-to-date totals and average monthly expenses."""
    # Filter out internal transfers
    filtered_df = merged_df[~merged_df['Category'].isin(['INTERNAL_TRANSFER'])]
    
    # Get current year and month
    current_year = pd.Timestamp.now().year
    current_month = pd.Timestamp.now().to_period('M')
    
    # Filter for current year data
    ytd_df = filtered_df[filtered_df['Transaction Date'].dt.year == current_year]
    
    if ytd_df.empty:
        return {
            'ytd_income': 0,
            'ytd_expenses': 0,
            'avg_monthly_expenses': 0
        }
    
    # Calculate YTD totals
    ytd_income = ytd_df[ytd_df['Sorting Type'] == 'Income']['Amount'].sum()
    ytd_expenses = abs(ytd_df[ytd_df['Sorting Type'] == 'Expense']['Amount'].sum())
    
    # Calculate average monthly expenses excluding current month
    # Filter out current month for average calculation
    historical_df = ytd_df[ytd_df['Month'] != current_month]
    
    if not historical_df.empty:
        # Calculate expenses for completed months only
        historical_expenses = abs(historical_df[historical_df['Sorting Type'] == 'Expense']['Amount'].sum())
        completed_months = historical_df['Month'].nunique()
        avg_monthly_expenses = historical_expenses / completed_months if completed_months > 0 else 0
    else:
        avg_monthly_expenses = 0
    
    return {
        'ytd_income': ytd_income,
        'ytd_expenses': ytd_expenses,
        'avg_monthly_expenses': avg_monthly_expenses
    }


def search_vendor_transactions(merged_df, vendor_search_term):
    """Search for transactions by vendor name."""
    if not vendor_search_term or vendor_search_term.strip() == "":
        return pd.DataFrame()
    
    # Filter out internal transfers
    filtered_df = merged_df[~merged_df['Category'].isin(['INTERNAL_TRANSFER'])]
    
    # Search for vendor in description (case insensitive)
    search_results = filtered_df[
        filtered_df['Description'].str.contains(vendor_search_term, case=False, na=False)
    ].copy()
    
    if search_results.empty:
        return pd.DataFrame()
    
    # Format the results
    # Multiply by -1 so expenses show as positive and refunds as negative
    search_results['Amount'] = search_results['Amount'] * -1
    search_results['Transaction Date'] = pd.to_datetime(search_results['Transaction Date']).dt.strftime('%Y-%m-%d')
    
    # Select and rename columns for display
    display_df = search_results[['Transaction Date', 'Description', 'Category', 'Sorting Type', 'Source', 'Amount']].copy()
    display_df = display_df.rename(columns={
        'Transaction Date': 'Date',
        'Description': 'Vendor',
        'Sorting Type': 'Type',
        'Amount': 'Amount ($)'
    })
    
    # Sort by date (most recent first)
    display_df = display_df.sort_values('Date', ascending=False)
    
    return display_df


def search_category_transactions(merged_df, category, month):
    """Search for transactions by category and month."""
    if not category or category.strip() == "":
        return pd.DataFrame()
    
    # Filter out internal transfers
    filtered_df = merged_df[~merged_df['Category'].isin(['INTERNAL_TRANSFER'])]
    
    # Filter by category and month
    search_results = filtered_df[
        (filtered_df['Category'] == category) & 
        (filtered_df['Month'] == month)
    ].copy()
    
    if search_results.empty:
        return pd.DataFrame()
    
    # Format the results
    # Multiply by -1 so expenses show as positive and refunds as negative
    search_results['Amount'] = search_results['Amount'] * -1
    search_results['Transaction Date'] = pd.to_datetime(search_results['Transaction Date']).dt.strftime('%Y-%m-%d')
    
    # Select and rename columns for display
    display_df = search_results[['Transaction Date', 'Description', 'Category', 'Sorting Type', 'Source', 'Amount']].copy()
    display_df = display_df.rename(columns={
        'Transaction Date': 'Date',
        'Description': 'Vendor',
        'Sorting Type': 'Type',
        'Amount': 'Amount ($)'
    })
    
    # Sort by date (most recent first)
    display_df = display_df.sort_values('Date', ascending=False)
    
    return display_df


def get_default_budgets():
    """Get default budget values for common expense categories."""
    default_budgets = {
        'Automotive': 100.0,
        "Bills & Utilities": 250.0,
        "Food & Drink": 100.0,
        "Gas": 250.0,
        "Groceries": 750.0,
        "Health & Wellness": 50.0,
        "Rent": 1400.0,
        "Tithing & Fast Offering": 800.0
    }
    return default_budgets


def get_default_budget_for_category(category):
    """Get default budget value for a specific category."""
    default_budgets = get_default_budgets()
    return default_budgets.get(category, 0.0)


def identify_monthly_subscriptions(merged_df):
    """Identify monthly subscriptions and recurring expenses."""
    # Filter out internal transfers
    filtered_df = merged_df[~merged_df['Category'].isin(['INTERNAL_TRANSFER'])]
    
    # Focus on expenses only
    expense_df = filtered_df[filtered_df['Sorting Type'] == 'Expense'].copy()
    
    if expense_df.empty:
        return pd.DataFrame()
    
    # Standardize vendor names for better grouping
    expense_df['Standardized_Vendor'] = expense_df['Description'].apply(standardize_merchant_name)
    
    # Group by vendor and analyze patterns
    subscription_candidates = []
    
    for vendor in expense_df['Standardized_Vendor'].unique():
        vendor_transactions = expense_df[expense_df['Standardized_Vendor'] == vendor].copy()
        
        if len(vendor_transactions) < 2:  # Need at least 2 transactions to be a subscription
            continue
        
        # Get unique months where this vendor appears
        unique_months = vendor_transactions['Month'].nunique()
        total_months = expense_df['Month'].nunique()
        
        # Calculate frequency (how often this vendor appears)
        frequency = len(vendor_transactions) / unique_months
        
        # Calculate average amount
        avg_amount = abs(vendor_transactions['Amount'].mean())
        
        # Calculate amount consistency (standard deviation as percentage of mean)
        amounts = abs(vendor_transactions['Amount'])
        amount_consistency = (amounts.std() / avg_amount) if avg_amount > 0 else 1
        
        # Determine if this looks like a subscription
        is_subscription = False
        subscription_type = "Unknown"
        
        # Criteria for monthly subscription:
        # 1. Appears in at least 50% of available months
        # 2. Frequency is close to 1 (appears once per month)
        # 3. Amount is relatively consistent (low standard deviation)
        if (unique_months >= max(2, total_months * 0.5) and  # Appears in at least 50% of months
            frequency >= 0.8 and  # Appears at least 0.8 times per month on average
            amount_consistency < 0.3):  # Amount varies by less than 30%
            
            is_subscription = True
            subscription_type = "Monthly Subscription"
        
        if is_subscription:
            # Get most recent transaction
            most_recent = vendor_transactions.sort_values('Transaction Date').iloc[-1]
            
            subscription_candidates.append({
                'Vendor': vendor,
                'Category': most_recent['Category'],
                'Average Amount': avg_amount,
                'Frequency': frequency,
                'Months Active': unique_months,
                'Total Months': total_months,
                'Amount Consistency': amount_consistency,
                'Subscription Type': subscription_type,
                'Last Transaction': most_recent['Transaction Date'].strftime('%Y-%m-%d'),
                'Source': most_recent['Source']
            })
    
    if not subscription_candidates:
        return pd.DataFrame()
    
    # Create DataFrame and sort by average amount (highest first)
    subscriptions_df = pd.DataFrame(subscription_candidates)
    subscriptions_df = subscriptions_df.sort_values('Average Amount', ascending=False)
    
    return subscriptions_df


def debug_vendor_analysis(merged_df, vendor_name):
    """Debug function to analyze why a specific vendor wasn't detected as a subscription."""
    # Filter out internal transfers
    filtered_df = merged_df[~merged_df['Category'].isin(['INTERNAL_TRANSFER'])]
    
    # Focus on expenses only
    expense_df = filtered_df[filtered_df['Sorting Type'] == 'Expense'].copy()
    
    if expense_df.empty:
        return "No expense data found"
    
    # Standardize vendor names for better grouping
    expense_df['Standardized_Vendor'] = expense_df['Description'].apply(standardize_merchant_name)
    
    # Find the vendor (case insensitive)
    vendor_matches = expense_df[expense_df['Standardized_Vendor'].str.contains(vendor_name, case=False, na=False)]
    
    if vendor_matches.empty:
        return f"No transactions found for vendor containing '{vendor_name}'"
    
    # Get the standardized vendor name
    actual_vendor = vendor_matches['Standardized_Vendor'].iloc[0]
    vendor_transactions = expense_df[expense_df['Standardized_Vendor'] == actual_vendor].copy()
    
    # Calculate metrics
    unique_months = vendor_transactions['Month'].nunique()
    total_months = expense_df['Month'].nunique()
    frequency = len(vendor_transactions) / unique_months
    avg_amount = abs(vendor_transactions['Amount'].mean())
    amounts = abs(vendor_transactions['Amount'])
    amount_consistency = (amounts.std() / avg_amount) if avg_amount > 0 else 1
    
    # Check criteria
    months_criteria = unique_months >= max(2, total_months * 0.5)
    frequency_criteria = frequency >= 0.8
    consistency_criteria = amount_consistency < 0.3
    
    # Create debug report
    debug_info = {
        'Vendor': actual_vendor,
        'Total Transactions': len(vendor_transactions),
        'Unique Months': unique_months,
        'Total Months in Data': total_months,
        'Frequency': frequency,
        'Average Amount': avg_amount,
        'Amount Consistency': amount_consistency,
        'Months Criteria (≥50%)': months_criteria,
        'Frequency Criteria (≥0.8)': frequency_criteria,
        'Consistency Criteria (<30%)': consistency_criteria,
        'Would be detected as subscription': months_criteria and frequency_criteria and consistency_criteria
    }
    
    return debug_info


# # Example usage --------------------------------------------------------------------------

# folder_path = 'data/jan 19'

# merged_df = process_csv_files(folder_path)  # used to build the summary table, used in the viewer dropdown, and used in the time series graph



# expense_trends_fig = build_expense_trends(merged_df, 'Groceries', '2023-04')
# expense_trends_fig.show()

