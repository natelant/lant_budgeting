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
        
        elif any(term in description for term in ['Chase card', 'Transfer from SAV', 'WELLS FARGO', 'Goldman']):
            return 'INTERNAL_TRANSFER'
    # Add more conditions as needed
    return 'Uncategorized'


def process_csv_files(folder_path):
    checking_account_df = None
    credit_card_dfs = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.CSV'):
            file_path = os.path.join(folder_path, filename)
            
            if 'Chase5121' in filename:
                # This is the checking account file
                checking_account_df = pd.read_csv(file_path, index_col=False)
                
                # If the first column is unnamed, give it a name
                if checking_account_df.columns[0] == 'Unnamed: 0':
                    checking_account_df = checking_account_df.rename(columns={'Unnamed: 0': 'Transaction_Type'})
                
                # Clean the checking account data
                checking_account_df = clean_checking_account(checking_account_df)
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
    
    # Calculate cumulative sum
    filtered_df['Cumulative'] = filtered_df['Amount'].cumsum()
    
    # Create scatter plot for individual transactions
    fig = px.scatter(
        filtered_df,
        x='Transaction Date',
        y='Amount',
        title=f'{category} Transactions for {month}',
        labels={
            'Transaction Date': 'Date',
            'Amount': f'Amount ($)'
        },
        hover_data=['Description']
    )
    
    # Add cumulative line on secondary y-axis
    fig.add_scatter(
        x=filtered_df['Transaction Date'],
        y=filtered_df['Cumulative'],
        name='Running Total',
        line=dict(dash='dot'),
        yaxis='y2',
        hovertemplate="Running Total: $%{y:.2f}<br>"
    )
    
    # Update layout with secondary y-axis
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
    
    # Update scatter plot traces
    fig.update_traces(
        marker=dict(size=10),
        hovertemplate="<br>".join([
            "Date: %{x}",
            "Amount: $%{y:.2f}",
            "Description: %{customdata[0]}"
        ]),
        selector=dict(mode='markers')  # Only apply to scatter points
    )
    
    return fig

# Example usage --------------------------------------------------------------------------

folder_path = 'data'
# maybe in the future we add rent as a parameter
merged_df = process_csv_files(folder_path)  # used to build the summary table, used in the viewer dropdown, and used in the time series graph
bottom_line_df = build_bottom_line(merged_df)  # displayed as summary table

print(bottom_line_df)

# bar_chart = build_bar_chart(merged_df)
# bar_chart.show()

timeseries_fig = build_timeseries(merged_df, 'Expense', '2024-09', 'Groceries')
timeseries_fig.show()

