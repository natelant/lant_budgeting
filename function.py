import pandas as pd
import os
import plotly.express as px
from datetime import datetime


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

    # Merge checking account and credit card data
    # Note: You may need to adjust the merge parameters based on your data structure
    merged_df = pd.concat([checking_account_df, combined_credit_card_df], ignore_index=True)

    # Rearrange columns
    merged_df = merged_df[['Transaction Date', 'Sorting Type', 'Category', 'Amount', 'Description']]
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
    

def build_expense_tracker(merged_df, category, month):
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
        'Category': 'first'  # Get the category for each description
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
        custom_data=[description_data['Category']]  # Add category data for hover
    )
    
    # Update traces with custom colors and hover template
    fig.update_traces(
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>" +
                      "Amount: $%{x:,.2f}<br>" +
                      "Category: %{customdata[0]}" +
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


# # Example usage --------------------------------------------------------------------------

# folder_path = 'data/jan 19'

# merged_df = process_csv_files(folder_path)  # used to build the summary table, used in the viewer dropdown, and used in the time series graph



# expense_trends_fig = build_expense_trends(merged_df, 'Groceries', '2023-04')
# expense_trends_fig.show()

