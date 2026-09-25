import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from statsmodels.stats.weightstats import ztest

    import warnings
    warnings.filterwarnings("ignore")
    return datetime, np, pd, plt, sns, stats, ztest


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Overview

    This is a continuation of the Online Retail Data Processing and Cleaning notebook by building customer value and churn diagnostics from the cleaned dataset.

    The notebook is organized as follows:
    1. Load the cleaned retail workbook and calculate a `TotalPrice` field.
    2. Build RFM metrics for recency, frequency, and monetary value.
    3. Create customer cohorts and evaluate retention patterns across time.
    4. Estimate churn using a rolling inactivity threshold.
    5. Explore whether simple value-based features can help explain churn behavior.

    Notes:
    - Negative `Quantity` values are treated as returns or refunds and removed from the customer behavior analysis.
    - The analysis is interpretive rather than strictly prescriptive; several thresholds are chosen to reflect a practical business-use example.
    - Any final model or decision should be validated with a broader business context and a more complete feature set.
    """)
    return


@app.cell
def _(pd):
    # Read in cleaned data
    data = pd.read_excel('../Data/Online Retail_clean.xlsx',
                    header = 0)
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ---
    # Data Preparation

    The analysis begins with the cleaned Online Retail dataset produced by the earlier processing workflow. Loading a cleaned workbook makes this notebook focused on customer-level analysis while keeping data-quality operations reproducible in a separate stage.

    The initial preview is a quick validation step: it confirms that the expected columns are present, that the file loaded successfully, and that the records have the expected tabular structure before derived measures are created.
    """)
    return


@app.cell
def _(data):
    data.head(5)
    return


@app.cell
def _(data):
    # Calculate TotalPrice of an invoice
    data['TotalPrice'] = data['Quantity']*data['UnitPrice']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Assuming negative `Quantity` values represent returns or refunds, the notebook removes them for the purposes of customer behavior analysis.

    Filtering returns is an analytical choice rather than a claim that those records are invalid. Refunds can be important for a separate operational analysis, but including negative quantities in customer-value metrics would mix purchase demand with reversals and could make customer spend or activity look artificially low. The filtered dataset therefore provides a consistent basis for comparing purchasing behavior.
    """)
    return


@app.cell
def _(data):
    # Take out invoices with negative quantities
    data_filtered = data[data['Quantity'] >= 0]
    return (data_filtered,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ___
    ___
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RFM Analysis

    RFM stands for recency, frequency, and monetary value. This notebook uses that framework to understand customer value by measuring:
    - how recently a customer made a purchase (`Recency`)
    - how often they purchase (`Frequency`)
    - how much they spend (`Monetary`)

    This structure is commonly used for customer segmentation, retention targeting, and identifying high-value accounts.

    ## Recency
    Recency is the measurement of how recent a customer's last purchase or invoice was.
    """)
    return


@app.cell
def _(data_filtered):
    # Aggregate original data by CustomerID to
    # get number of unique Invoices
    # and the date of the last invoice
    days_since_last = data_filtered\
                        .groupby(['CustomerID','Country'])\
                        .agg({'InvoiceNo':'nunique',
                              'InvoiceDate':'max'})
    return (days_since_last,)


@app.cell
def _(datetime, days_since_last):
    # Calculate the number of days since last purchase
    ## Go from Jan 1, 2012 to simulate data analysis 
    ## on a narrower timescale
    days_since_last['day_since_last_purchase'] = (datetime.datetime.strptime('01012012','%m%d%Y') - days_since_last['InvoiceDate']).dt.days
    return


@app.cell
def _(days_since_last):
    last_invoice = days_since_last.sort_values(['InvoiceNo',
                                                 'InvoiceDate'],
                                                ascending = False)\
                                    .rename({'InvoiceDate':'LastInvoiceDate',
                                            'InvoiceNo':'NumInvoices'},
                                            axis = 1)
    return (last_invoice,)


@app.cell
def _(last_invoice):
    last_invoice.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frequency
    Frequency measures how many purchases a customer has made and how regularly they make them.

    The notebook computes this by looking at invoice dates, removing duplicates, and measuring the average gap between purchase events.
    """)
    return


@app.function
def days_between_purchases(group):
    # Custom groupby function to get the number of days between purchases
    # Get the InvoiceDate of each order and remove duplicates. Sort by date
    group_sorted = group[['InvoiceDate']]\
                    .drop_duplicates()\
                    .sort_values('InvoiceDate', ascending = True)

    # Calculate the number of days between purchase dates
    group_sorted['days_between_last_purchase'] = (group_sorted['InvoiceDate'] - group_sorted['InvoiceDate'].shift(1)).dt.days

    return group_sorted[['InvoiceDate', 'days_between_last_purchase']]


@app.cell
def _(data_filtered):
    days_between_purchases_1 = data_filtered.groupby(['CustomerID', 'Country']).apply(days_between_purchases).droplevel(2).reset_index()
    return (days_between_purchases_1,)


@app.cell
def _(days_between_purchases_1):
    # Aggregate by CustomerID to get the total number of InvoiceDate, the average time between them, and variance of time between them
    avg_days_between = days_between_purchases_1.groupby(['CustomerID', 'Country']).agg({'InvoiceDate': 'count', 'days_between_last_purchase': ['mean', 'var']})
    return (avg_days_between,)


@app.cell
def _(avg_days_between):
    avg_days_between.columns = ['NumInvoices', 'AvgDaysBetweenPurchases', 'VarDaysBetweenPurchases']
    return


@app.cell
def _(avg_days_between):
    avg_days_between.sort_values(['NumInvoices',
                                 'AvgDaysBetweenPurchases'],
                                 ascending = False,
                                inplace = True)
    return


@app.cell
def _(avg_days_between):
    avg_days_between.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Monetary
    Monetary captures how much each customer has spent in total and the average spend per invoice.

    This allows the notebook to distinguish not only between frequent customers, but also between customers whose value comes from larger basket sizes or higher transaction intensity.
    """)
    return


@app.cell
def _(data_filtered):
    # Calculate the total amount spent per customer
    customer_total_spent = data_filtered.groupby(['CustomerID','Country'])\
                            [['InvoiceNo','TotalPrice']]\
                            .agg({'InvoiceNo':'nunique',
                                  'TotalPrice':'sum'})\
                            .sort_values('TotalPrice',
                                         ascending = False)\
                            .rename({'InvoiceNo':'NumInvoices',
                                     'TotalPrice':'TotalSpend'},
                                    axis = 1)
    return (customer_total_spent,)


@app.cell
def _(customer_total_spent):
    # On average, how much does each customer spend
    customer_total_spent['AvgSpend'] = customer_total_spent['TotalSpend']/customer_total_spent['NumInvoices']
    return


@app.cell
def _(customer_total_spent):
    customer_total_spent.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Merging RFM
    """)
    return


@app.cell
def _(avg_days_between, customer_total_spent, last_invoice, pd):
    # Merge recency, frequency, and monetary data into one
    rfm_merge = pd.merge(pd.merge(last_invoice,
                                 avg_days_between.drop('NumInvoices',
                                                       axis = 1),
                                 how = 'outer',
                                 left_index = True,
                                 right_index = True),
                        customer_total_spent.drop('NumInvoices',
                                                  axis = 1),
                        how = 'outer',
                        left_index = True,
                        right_index = True)
    return (rfm_merge,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The three perspectives are joined at the `CustomerID` and `Country` level so each customer has one consolidated analytical record. Combining these measures is important because a customer may be recent but infrequent, frequent but low-spending, or historically valuable but inactive. RFM makes those distinctions visible for segmentation and retention planning.
    """)
    return


@app.cell
def _(rfm_merge):
    rfm_merge.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ___
    ___
    # Cohort Analysis

    Cohort analysis groups customers by their first purchase period so the notebook can compare retention and customer behavior across cohorts over time.

    This is useful for spotting whether newer customers behave differently from older ones and whether a specific acquisition period appears more stable or more at risk of churn.
    """)
    return


@app.cell
def _(data_filtered):
    # Get the date of a customers first invoice
    first_purchase_date = data_filtered\
                            .groupby(['CustomerID','Country'],
                                    dropna = False)\
                            [['InvoiceDate']]\
                            .min()\
                            .rename({'InvoiceDate':'FirstInvoiceDate'},
                                    axis = 1)
    return (first_purchase_date,)


@app.cell
def _(datetime, first_purchase_date):
    # Create cohorts based upon FirstInvoiceDate
    first_purchase_date['Cohort'] = first_purchase_date['FirstInvoiceDate'].apply(lambda x: datetime.datetime.strftime(x,'%b %Y'))
    return


@app.cell
def _(data_filtered, first_purchase_date, pd):
    # Merge the Cohort onto the original data
    data_cohorts = pd.merge(data_filtered,
                            first_purchase_date,
                            how = 'left',
                            left_on = ['CustomerID','Country'],
                            right_index = True)
    return (data_cohorts,)


@app.cell
def _(data_cohorts):
    # Get the cohort index, how many months from cohort month
    data_cohorts['CohortIndex'] = (data_cohorts['InvoiceDate'].dt.to_period('M') - data_cohorts['FirstInvoiceDate'].dt.to_period('M')).apply(lambda x: x.n)
    return


@app.cell
def _(data_cohorts):
    data_cohorts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Retention

    Retention is evaluated by tracking each cohort over monthly cohort indices and comparing how many customers remain active from one month to the next.

    The notebook normalizes each cohort against its initial month so the retention trend can be interpreted as a percentage rather than a raw customer count.
    """)
    return


@app.cell
def _(data_cohorts, datetime, pd):
    # Pivot cohort data to get number of unique customers per cohort
    # Take out December 2011 due to incomplete data
    cohort_retention_values = pd.pivot_table(data_cohorts[data_cohorts['InvoiceDate'] < datetime.datetime.strptime('12-2011','%m-%Y')],
                                             index = 'Cohort',
                                             columns = 'CohortIndex',
                                             values = 'CustomerID',
                                             aggfunc = 'nunique')
    return (cohort_retention_values,)


@app.cell
def _(cohort_retention_values, pd):
    # Convert index into datetime objects
    cohort_retention_values.index = pd.to_datetime(cohort_retention_values.index,
                                                   format = '%b %Y')
    return


@app.cell
def _(cohort_retention_values):
    cohort_retention_values.sort_index(inplace = True)
    return


@app.cell
def _(cohort_retention_values):
    # Divide values in each row by the first (0) CohortIndex
    ## Gets the percentage of a cohort retained each proceeding month
    cohort_retention_pct = cohort_retention_values.divide(cohort_retention_values.loc[:,0],
                                                          axis = 0)\
                            * 100
    return (cohort_retention_pct,)


@app.cell
def _(cohort_retention_pct, plt, sns):
    # Plot the cohort retention percentage
    _fig, _ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(cohort_retention_pct, annot=True, cmap='YlGnBu', fmt='.0f', ax=_ax)
    yticklabels = [cohort_retention_pct.index[int(tick)].strftime('%b-%Y') for tick in _ax.get_yticks()]
    _ax.set_yticklabels(yticklabels)
    _ax.set_title('Cohort Retention Pct by CohortIndex')
    plt.show()
    return


@app.cell
def _(cohort_retention_pct, pd):
    # Melt the pivoted cohorts to plot line graphs
    cohort_retention_melted = pd.melt(cohort_retention_pct.drop(0, axis = 1),
                                      ignore_index = False)\
                                .dropna()\
                                .rename({'value':'RetentionRate'},
                                        axis = 1)\
                                .reset_index()
    return (cohort_retention_melted,)


@app.cell
def _(cohort_retention_melted, plt, sns):
    # Plot line charts by Cohort
    _fig, _ax = plt.subplots(figsize=(15, 8))
    sns.lineplot(cohort_retention_melted, x='CohortIndex', y='RetentionRate', hue=cohort_retention_melted['Cohort'].dt.strftime('%b %Y'), ax=_ax)
    _ax.set_title('Cohort Retention Pct by CohortIndex')
    plt.show()
    return


@app.cell
def _(cohort_retention_melted, plt, sns):
    # Plot line charts by CohortIndex
    _fig, _ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(cohort_retention_melted, x='Cohort', y='RetentionRate', ax=_ax)
    _ax.set_title('Cohort Retention Rates')
    _ax.set_xticklabels([cohort.strftime('%b %Y') for cohort in sorted(cohort_retention_melted['Cohort'].unique())])
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By looking at the previous charts, it is clear that the first cohort, December 2010, shows the strongest repeat-customer behavior. It also appears to retain customers more steadily than later cohorts, although that observation alone does not determine which cohort is the most monetarily valuable.

    This motivates the follow-up monetary analysis, where value is examined in terms of spend per customer over time.
    """)
    return


@app.cell
def _(cohort_retention_melted, plt, sns):
    # Plot histograms of retention rates for the first cohort and all other cohorts
    _figure, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(cohort_retention_melted[cohort_retention_melted['Cohort'] == '2010-12-01']['RetentionRate'],
                kde=True,
                ax=ax)
    sns.histplot(cohort_retention_melted[cohort_retention_melted['Cohort'] != '2010-12-01']['RetentionRate'],
                kde=True,
                ax=ax)
    plt.legend(['Cohort: Dec 2010', 'All Other Cohorts'])
    plt.show()
    return


@app.cell
def _(cohort_retention_melted, stats):
    # Perform a t-test to compare the retention rates of the first cohort to all other cohorts
    t_stat, p_val = stats.ttest_ind(cohort_retention_melted[cohort_retention_melted['Cohort'] == '2010-12-01']['RetentionRate'],
                                                        cohort_retention_melted[cohort_retention_melted['Cohort'] != '2010-12-01']['RetentionRate'],
                                                        alternative='greater')
    print(t_stat)
    print(p_val)
    _alpha = 0.05
    if p_val <= _alpha:
        print('The null hypothesis is rejected. The first cohort retention rate is different from the other cohorts.')
    else:
    # Significance level
        print('The null hypothesis is not rejected. The first cohort retention rate is not different from the other cohorts.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The one-sided two-sample t-test evaluates whether the December 2010 retention observations are higher than those from the other cohorts. The p-value provides a statistical check on the visual pattern, while the 0.05 significance level defines the decision rule used here.

    This test is useful because it separates a potentially meaningful cohort difference from a pattern that may be explained by sampling variation. The result should still be treated cautiously because cohort observations may not be fully independent and the available sample is limited.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cohort Monetary Value

    This section compares cohorts by monetary output over time, using both total spend and spend-per-customer metrics.

    The goal is to determine whether the stronger retention signal observed in earlier cohorts also translates into higher revenue contribution.
    """)
    return


@app.cell
def _(data_cohorts, pd):
    # Calculate the total number of customers, invoices, and total price per cohort and cohort index
    cohort_value = data_cohorts.groupby([pd.to_datetime(data_cohorts['Cohort'],
                                                       format = '%b %Y'),
                                         'CohortIndex'])\
                        [['CustomerID','InvoiceNo','TotalPrice']]\
                        .agg({'CustomerID':'nunique',
                              'InvoiceNo':'nunique',
                              'TotalPrice':'sum'})\
                        .reset_index()
    return (cohort_value,)


@app.cell
def _(cohort_value):
    cohort_value['TotalPricePerCustomer'] = cohort_value['TotalPrice']/cohort_value['CustomerID']
    cohort_value['TotalPricePerInvoice'] = cohort_value['TotalPrice']/cohort_value['InvoiceNo']
    return


@app.cell
def _(cohort_value, plt, sns):
    # Plot line charts of TotalPrice and TotalPricePerCustomer by CohortIndex
    _figure, (_ax1, _ax2) = plt.subplots(nrows=2,
                                        ncols=1,
                                        figsize=(10, 10))
    sns.lineplot(cohort_value,
                    x='CohortIndex',
                    y='TotalPrice',
                    hue=cohort_value['Cohort'].dt.strftime('%b %Y'),
                    ax=_ax1)

    _ax1.set_title('TotalPrice Over CohortIndex by Cohort')

    sns.lineplot(cohort_value,
                    x='CohortIndex',
                    y='TotalPricePerCustomer',
                    hue=cohort_value['Cohort'].dt.strftime('%b %Y'),
                    ax=_ax2)

    # Month over month (CohortIndex) TotalPrice spent by cohort 
    _ax2.set_title('TotalPricePerCustomer Over CohortIndex by Cohort')

    # Month over month (CohortIndex) TotalPrice per CustomerID by cohort
    plt.show()
    return


@app.cell
def _(data_cohorts, pd):
    # Calculate the total amount spent per customer per cohort
    cohort_customer_spend = data_cohorts.groupby(['Cohort',\
                                                  'CustomerID',
                                                  'Country',
                                                  pd.Grouper(key = 'InvoiceDate',
                                                             freq = '1D')])\
                                [['TotalPrice']]\
                                .sum()\
                                .reset_index()
    return (cohort_customer_spend,)


@app.cell
def _(cohort_customer_spend, ztest):
    # Perform the two-sample z-test
    # value=0 means we test the null hypothesis that the difference between means is zero (mean_A = mean_B)
    z_stat, p_value = ztest(x1=cohort_customer_spend[cohort_customer_spend['Cohort'] == 'Dec 2010']['TotalPrice'],
                                              x2=cohort_customer_spend[cohort_customer_spend['Cohort'] != 'Dec 2010']['TotalPrice'],
                                              value=0,
                                              alternative='larger')
    print(z_stat)
    print(p_value)

    _alpha = 0.05
    if p_value <= _alpha:
        print('The null hypothesis is rejected. The first cohort spends more than the other cohorts.')
    else:
    # Significance level
        print("The null hypothesis is not rejected. The first cohort doesn't spend more than the other cohorts.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two-sample z-test compares daily customer spend for the December 2010 cohort with all other cohorts. Testing monetary behavior matters because retention alone does not show business impact: a cohort can return often while generating relatively little revenue, or produce high value despite lower retention.

    When used together, the retention and spend comparisons support a trend that the first cohort, Dec 2020, are retained and spend more than all other cohorts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Churn Rate

    Churn rate is the percentage of customers who become inactive over a selected time period. In this notebook, churn is approached as a customer inactivity problem: after a long gap in purchases, the customer is considered to have churned.

    This gives the analysis a simple operational definition that can be used for a rolling time-series view of customer retention.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Consolidate Daily Transactions

    To simplify the churn analysis, transactions are consolidated by customer and by day. This removes intra-day invoice noise and produces a cleaner sequence of purchase activity for each customer.

    The resulting daily series is then used to estimate customer inactivity and define a churn window.
    """)
    return


@app.cell
def _(data_filtered, pd):
    # Group and resample data by CustomerID, Country, and InvoiceDate by day
    ## Remove negative values
    data_grouped = data_filtered\
                        .groupby(['CustomerID',
                                 'Country',
                                 pd.Grouper(key = 'InvoiceDate',
                                            freq = '1D')])\
                        [['Quantity','TotalPrice']]\
                        .sum()\
                        .reset_index()
    return (data_grouped,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mean Days Between Transactions

    This section estimates the average time between repeat-purchase events so a practical churn threshold can be chosen.

    The threshold is intentionally a business-use example rather than a universal rule. A smaller threshold would make churn appear more frequent, while a larger threshold would delay the churn signal.
    """)
    return


@app.function
# Calculate the mean time between purchases (TBP) for each customer
def mean_tbp(group):
    group['Previous Invoice Date'] = group['InvoiceDate'].shift(1)

    group['TBP'] = (group['InvoiceDate'] - group['Previous Invoice Date']).dt.days

    mean_tbp = group[['TBP']].mean()

    return mean_tbp


@app.cell
def _(data_grouped):
    customer_mean_tbp = data_grouped.groupby(['CustomerID','Country']).apply(mean_tbp)
    return (customer_mean_tbp,)


@app.cell
def _(customer_mean_tbp):
    # Calculate the average time between purchases for all customers
    # This will be used to help determine a baseline number of days before a customer is considered to have churned
    print('Average time between purchases: %.2f' %customer_mean_tbp['TBP'].mean())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since the average number of days between purchases for repeat customers is roughly 79 days, a churn threshold in that vicinity—plus a small buffer—is a reasonable starting point.

    In this notebook, the threshold is set to 90 days for a simple illustrative rule. That choice is meant to create a stable operational definition, not to claim that 90 days is the only valid churn cutoff.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rolling Churn Rate

    Once a churn threshold has been set, the notebook evaluates how the churn rate evolves over time.

    This gives a rolling perspective on customer inactivity and helps determine whether the business is retaining customers at a stable rate or losing activity over time.
    """)
    return


@app.cell
def _(datetime, pd):
    def calculate_rolling_churn(group):
        # Custom groupby function to get determine if customer churned since last purchase
        # Get the InvoiceDate of each order and remove duplicates. Sort by date
        group_dropped = group[['InvoiceDate','Quantity','TotalPrice']]\
                        .drop_duplicates()

        group_consolidated = group_dropped\
                                .groupby('InvoiceDate')\
                                .sum()

        # Set rolling period
        rolling_period = 90

        # Calculate rolling sum of Quantity and TotalPrice over the rolling period
        group_rolling = group_consolidated.rolling(rolling_period, 
                                                   min_periods = 1)\
                        .sum()\
                        .reset_index()

        # Shift the InvoiceDate to get the last invoice date for the customer
        # This will help determine the number of days between purchases
        group_rolling['LastInvoiceDate'] = group_rolling['InvoiceDate'].shift(1)

        # Calculate the number of days since last purchase
        group_rolling['DaysSinceLastPurchase'] = (group_rolling['InvoiceDate'] - group_rolling['LastInvoiceDate']).dt.days

        # Get a rolling count of invoices per customer over the rolling period
        group_rolling['NumInvoices'] = group_rolling['InvoiceDate']\
                                        .rolling(rolling_period,
                                                 min_periods = 1)\
                                        .count()

        # Get a rolling average of days between purchases over the rolling period
        group_rolling['RollingAvgDaysBetweenPurchases'] = group_rolling['DaysSinceLastPurchase']\
                                                            .rolling(rolling_period,
                                                                    min_periods = 1)\
                                                            .mean()

        # Determine if the customer has churned based upon the number of days before the next purchase, if one exists
        # If not, fill the next purchase date with a date far in the future to ensure that the customer is considered churned
        group_rolling['Churn'] = (group_rolling['InvoiceDate'].shift(-1,
                                                                    fill_value = datetime.datetime.strptime('01012012', '%m%d%Y'))\
                                 - group_rolling['InvoiceDate']).dt.days > 90

        # Calculate the date in which the customer will have churned based upong the InvoiceDate
        group_rolling['Churn Date'] = group_rolling['InvoiceDate'] + pd.DateOffset(days = rolling_period)

        return group_rolling

    return (calculate_rolling_churn,)


@app.cell
def _(calculate_rolling_churn, data_grouped):
    # Determine which customers churned. Can use days_between_purchases data
    data_agg = data_grouped\
                .groupby(['CustomerID', 'Country'])\
                [['InvoiceDate','Quantity','TotalPrice']]\
                .apply(calculate_rolling_churn)\
                .droplevel(2)\
                .reset_index()
    return (data_agg,)


@app.cell
def _(data_agg):
    # Rename columns to indicate that they are rolling values
    data_agg.rename(columns = {'Quantity':'RollingQuantity',
                               'TotalPrice':'RollingTotalPrice',
                               'NumInvoices':'RollingNumInvoices'},
                    inplace = True)
    return


@app.cell
def _(data_agg, datetime):
    data_agg[(data_agg['InvoiceDate'] <= datetime.datetime.strptime('03012011','%m%d%Y'))
                # & (data_agg['RollingAvgDaysBetweenPurchases'].isna())
    ]['CustomerID'].nunique()
    return


@app.cell
def _(data_agg, pd):
    # List of customers by day
    data_daily_customers = data_agg.groupby(pd.Grouper(key = 'InvoiceDate',
                                                       freq = 'D'))\
                                    .apply(lambda x: x['CustomerID'].unique())
    return (data_daily_customers,)


@app.cell
def _(data_daily_customers):
    # Resample list of customers by month
    data_resample_monthly = data_daily_customers.resample('ME')\
                                                .apply(lambda x: [item for i in x
                                                                  for item in i])
    return (data_resample_monthly,)


@app.cell
def _(data_resample_monthly, pd):
    # Get 3 month rolling list of unique customers
    rolling_customers = pd.DataFrame()

    for i in range(2, len(data_resample_monthly.index)):
        rolling_customers.loc[data_resample_monthly.index[i],
                                '3 Month Rolling Unique Customers'] = len(set([*data_resample_monthly.iloc[i-2],
                                                                               *data_resample_monthly.iloc[i-1],
                                                                               *data_resample_monthly.iloc[i]]))
    return (rolling_customers,)


@app.cell
def _(data_agg, pd):
    # Number of customers churned in given month
    monthly_churn_customers = data_agg.groupby(pd.Grouper(key =  'Churn Date',
                                                          freq = 'ME'))\
                                        .apply(lambda x: x[x['Churn'] == True][['CustomerID']].nunique())\
                                        .rename(columns = {'CustomerID':'Churned Customers'})
    return (monthly_churn_customers,)


@app.cell
def _(monthly_churn_customers, pd, rolling_customers):
    # Join number of monthly customers and monthly churned
    monthly_churn_stats = pd.merge(rolling_customers,
                                   monthly_churn_customers,
                                   how = 'inner',
                                   left_index = True,
                                   right_index = True)
    return (monthly_churn_stats,)


@app.cell
def _(monthly_churn_stats):
    monthly_churn_stats['Churn Rate'] = monthly_churn_stats['Churned Customers']/monthly_churn_stats['3 Month Rolling Unique Customers']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The monthly churn rate is calculated as churned customers divided by the three-month rolling population of active customers. Using a rolling denominator smooths short-term fluctuations in customer activity and makes monthly comparisons more stable than relying on a single day's customer count.

    The final chart combines customer population, churned-customer counts, and the churn-rate line. This view is important for decision-making because the same number of churned customers has a different business meaning when the active customer base is growing versus shrinking.
    """)
    return


@app.cell
def _(monthly_churn_stats):
    monthly_churn_stats
    return


@app.cell
def _(monthly_churn_stats):
    # Get average 
    customer_churn_pct = monthly_churn_stats['Churned Customers'].sum()/monthly_churn_stats['3 Month Rolling Unique Customers'].sum()

    print('Average monthly churn rate: %f' % (customer_churn_pct * 100))
    return


@app.cell
def _(monthly_churn_stats, np, pd, plt, sns):
    # Plot 3 month rolling unique customers and churned customers
    _fig, _ax1 = plt.subplots(figsize=(15, 8))
    churn_stats_melted = pd.melt(monthly_churn_stats,
                                            value_vars=['3 Month Rolling Unique Customers',
                                                        'Churned Customers',
                                                        'Churn Rate'],
                                            ignore_index=False)\
                                    .reset_index()\
                                    .rename(columns={'index': 'Month'})
    sns.barplot(churn_stats_melted[churn_stats_melted['variable'] != 'Churn Rate'],
                x='Month',
                y='value',
                hue='variable',
                ax=_ax1)

    _ax2 = _ax1.twinx()
    sns.lineplot(monthly_churn_stats.reset_index(),
                x=np.arange(len(monthly_churn_stats)),
                y='Churn Rate',
                color='red',
                ax=_ax2)

    xticklabels = [churn_stats_melted['Month'].unique()[int(tick)].strftime('%b %Y')
                                    for tick in _ax1.get_xticks()]
    _ax1.set_xticklabels(xticklabels)
    plt.show()
    return (churn_stats_melted,)


@app.cell
def _(churn_stats_melted):
    churn_stats_melted
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Executive Conclusions and Next Steps

    The analysis provides a structured view of customer value, retention, and inactivity using the cleaned Online Retail transaction data. The RFM results establish a practical customer-level foundation by combining recency, purchasing frequency, and total spend. This helps distinguish recently active customers from customers who have historically generated value but may now require re-engagement.

    The cohort analysis indicates that the December 2010 cohort demonstrated stronger repeat-customer behavior than the later cohorts in the observed data. The retention charts and statistical comparison provide evidence for investigating what differentiated this group, although the relatively small sample and potential dependence between observations mean that the result should be treated as directional rather than definitive. The monetary analysis adds an important business perspective by testing whether stronger retention was also associated with higher customer spend.

    The churn analysis translates inactivity into an operational measure. Customers with a gap of more than 90 days between purchases are classified as churned, based on an observed average purchase interval of approximately 79 days plus a practical buffer. The monthly churn view makes changes in customer activity easier to monitor, but the threshold is an analytical assumption and should be calibrated against business cycles, product purchasing patterns, and known customer outcomes.

    From a business perspective, the results support prioritizing retention efforts around high-value customers with increasing recency, declining purchase frequency, or inactivity beyond the selected threshold. Cohort differences should also prompt a review of acquisition source, onboarding experience, product mix, promotions, geography, and customer service history to identify the factors associated with stronger retention.

    Recommended next steps:
    - Extend the feature set with product categories, order size, channel, country, seasonality, returns, and discount or promotion information.
    - Use survival analysis or a supervised churn model to estimate each customer’s probability of future activity and evaluate model performance on unseen data.
    - Test targeted retention campaigns with a control group so that changes in repeat purchases and customer value can be attributed to the intervention.
    - Establish recurring monitoring of cohort retention, churn rate, customer lifetime value, and campaign outcomes in a business dashboard.

    These steps would move the work from exploratory analysis toward a validated customer-retention process that can support measurable commercial decisions.

    ## Outputs and Business Use

    The workbook export preserves the main analytical tables for downstream reporting: customer-level RFM measures, cohort retention percentages, cohort transaction records, and customer-level churn observations. Separating these outputs into named sheets makes the results reusable in dashboards or stakeholder review without rerunning every chart.

    The overall workflow demonstrates a progression from transaction-level preparation to customer-level metrics, cohort diagnostics, statistical comparison, and an operational churn indicator. In a production setting, the next validation steps would include testing alternative churn windows, checking statistical assumptions, monitoring the definition of returns, and adding product, channel, geography, and customer-segment features before using the results for targeting decisions.
    """)
    return


@app.cell
def _(cohort_retention_pct, data_agg, data_cohorts, pd, rfm_merge):
    # Export cohort data to excel
    # Data from this notebook will be used downstream for further analysis and modeling
    with pd.ExcelWriter('./Data/Online Retail_Analysis.xlsx', mode='a', if_sheet_exists='replace') as _writer:
        rfm_merge.to_excel(_writer,
                            sheet_name='RFM',
                            index=True)
        cohort_retention_pct.to_excel(_writer,
                                        sheet_name='Cohorts Pct',
                                        index=False)
        data_cohorts.to_excel(_writer,
                                sheet_name='Cohorts',
                                index=False)
        data_agg.to_excel(_writer,
                            sheet_name='Customer Churn',
                            index=False )
    return


if __name__ == "__main__":
    app.run()
