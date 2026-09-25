import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    import pandas as pd
    import numpy as np
    import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from os.path import dirname, realpath
    import sys

    import warnings
    warnings.filterwarnings("ignore")
    return dirname, mo, pd, plt, realpath, sns, stats, sys


@app.cell
def _():
    import xgboost as xgb

    from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split, GridSearchCV, KFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay, auc, fbeta_score, make_scorer

    return (
        ColumnTransformer,
        ConfusionMatrixDisplay,
        FunctionTransformer,
        GridSearchCV,
        KFold,
        KNeighborsClassifier,
        LogisticRegression,
        OneHotEncoder,
        Pipeline,
        RandomForestClassifier,
        SimpleImputer,
        StandardScaler,
        auc,
        classification_report,
        confusion_matrix,
        fbeta_score,
        make_scorer,
        roc_curve,
        train_test_split,
        xgb,
    )


@app.cell
def _():
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    return (variance_inflation_factor,)


@app.cell
def _(dirname, realpath, sys):
    current_dir = dirname(realpath(__file__))
    parent_dir = dirname(current_dir)
    sys.path.append(parent_dir)

    from src.column_transformers import (
        split_date_monthday,
        split_date_month,
        split_date_day,
        monthday_feature_names,
        month_feature_names,
        day_feature_names,
    )

    return (
        day_feature_names,
        month_feature_names,
        monthday_feature_names,
        split_date_day,
        split_date_month,
        split_date_monthday,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ---
    # Overview
    The following notebook takes the Online Retail data and features engineered in the **RFM and Churn Analysis** notebook and attempts to develop a machine learning model to predict whether or not a customer will churn.

    # Data Import
    """)
    return


@app.cell
def _(pd):
    # Import the Customer Churn and Cohort data from the Online Retail_Analysis.xlsx file
    # This file is produced by the RFM and Cohort Analysis notebook
    data_customers = pd.read_excel('../Data/Online Retail_Analysis.xlsx',
                                    sheet_name='Customer Churn',
                                    header = 0)

    data_cohorts = pd.read_excel('../Data/Online Retail_Analysis.xlsx',
                                    sheet_name='Cohorts',
                                    header = 0)
    return data_cohorts, data_customers


@app.cell
def _(data_customers):
    data_customers.head(5)
    return


@app.cell
def _(data_cohorts):
    data_cohorts.head(5)
    return


@app.cell
def _(data_cohorts):
    # Remove duplicate customers from the cohort data to ensure a 1:1 relationship between the customer and their cohort
    customer_cohorts = data_cohorts[['CustomerID','Country','Cohort', 'FirstInvoiceDate']]\
                                .drop_duplicates(['CustomerID','Country'],
                                                keep='first')\
                                .reset_index(drop=True)
    return (customer_cohorts,)


@app.cell
def _(customer_cohorts, data_customers, pd):
    # Merge the customer churn data with the cohort data to create a single dataset for modeling
    customer_features = pd.merge(data_customers,
                                        customer_cohorts,
                                        how='left',
                                        on=['CustomerID','Country'])
    return (customer_features,)


@app.cell
def _(customer_features):
    customer_features
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploratory Data Analysis
    Before modeling, a deeper dive into the different features is necessary to determine any outliers or problems within the data that may skew it.

    First, I will look at the distribution of the target variable, `Churn`. This will help give a baseline number for evaluating different models. If a model can't perform better than this baseline number, then the model will not be considerably better than random guessing.

    ## `Churn`
    """)
    return


@app.cell
def _(customer_features):
    # Determine what percentage of invoices lead to churn
    churned_invoices_pct = customer_features['Churn'].sum()/len(customer_features)

    # Determine what percentage of customers churned at least once
    customers_churned_grouped = customer_features.groupby(['CustomerID','Country'])\
                                                            [['Churn']]\
                                                            .max()\
                                                            .reset_index()

    churned_customers_pct = customers_churned_grouped['Churn'].sum()/len(customers_churned_grouped)

    print(f'Total number of invoices: {len(customer_features)}')
    print(f'Percentage of invoices that lead to churn: {churned_invoices_pct:.2%}')
    print(f'Total number of customers: {len(customers_churned_grouped)}')
    print(f'Percentage of customers that churned at least once: {churned_customers_pct:.2%}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are roughly 16,800 invoices and 4,300 customers in the dataset, of which, approximately 21% and 67% of invoices and customers, respectively, have churned. These will be the baselines for models when evaluating their performances compared to random guessing.

    ## `InvoiceDate`

    Next, I want to see the distribution of invoices and when they were made. This will give me an idea if when the invoices where made will have any impact on churn. As we saw in the **RFM and Churn Analysis** notebook, the first cohort, December 2010, is the most loyal and highest spenders meaning if there were significantly more invoices during this time, it may deflate the overall churn prediction. However, if there is more or less an even distribution of invoices month-over-month, then the model may perform better when predicting whether or not a customer will churn based upon the invoice data.
    """)
    return


@app.cell
def _(customer_features, plt, sns):
    # Plot the distribution of invoices by invoice date
    fig, ax = plt.subplots(figsize=(10,10))
    fig = sns.histplot(data = customer_features,
                                x = 'InvoiceDate',
                                hue = 'Churn',
                                bins = 25)
    plt.title('Number of Invoices by Invoice Date Distribution')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By looking at the distribution of invoices by invoice date, and the number of churned invoices, we can see that it is weighted more towards the December months and end of the year and although that was something I mentioned to keep an eye on, it may not be a major issue in this analysis. Given the limited size of the dataset and the timeline provided, it's hard to say if this is a isolated event or if the number of invoices is seasonal with more occurring towards the end of the year. It's certainly plausible that the data has a seasonal trend to it, and given the one year sample size, I'm only going to keep the month aspect when engineering this feature.

    Next, I want to break up the `InvoiceDate` column into separate features for the year, month, and day of the invoice. This will allow us to analyze the data more granularly and potentially identify patterns related to specific time periods.
    """)
    return


@app.cell
def _(customer_features):
    # Create columns for the month, and day of the InvoiceDate
    # Don't need to create a column for the year since all invoices are from 2011
    customer_features['InvoiceMonth'] = customer_features['InvoiceDate'].dt.month
    customer_features['InvoiceDay'] = customer_features['InvoiceDate'].dt.day
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `Country`

     If they could be used to predict `Churn`, I also want to turn the `Country` feature into dummy variables, which will create a column for each of the values within `Country` and for each row, hold the value **0** for all the columns except the rows true country label where it will be **1**. The issue is that it'll create a lot of columns for each of the countries. I want to simplify it to reduce the number of columns by keeping the common countries and labeling other values as **Other**.
    """)
    return


@app.cell
def _(customer_features):
    # Determine how many of each country is represented
    country_counts = customer_features['Country'].value_counts(normalize = True)
    return (country_counts,)


@app.cell
def _(country_counts, customer_features):
    # Gather a list of countries where they represent less than 1% of the data
    # This list will be used to change their country to 'Other' to reduce the number of dimensionals when creating dummy variables
    other_countries_list = country_counts[country_counts < 0.01].index
    print(f'Ratio of countries that represent less than 1% of the data: {len(other_countries_list)}:{len(country_counts)}')

    customer_features['Country'] = customer_features['Country'].replace(other_countries_list, 'Other')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that I've narrowed down the `Country` feature to a few number of categories, I want to see if there's any correlation between it and `Churn`. To do this, I will perform a Chi-Square statistical test with a alpha of 0.05. If I manage to get a p-value less that 0.05, it will be safe to assume that `Country` and `Churn` are correlated. If not, then there may not be any correlation between the two variables.
    """)
    return


@app.cell
def _(customer_features, pd):
    # Create a contingency table between country and churn to get the frequency distribution between the two
    country_contingency_table = pd.crosstab(customer_features['Country'],
                                                        customer_features['Churn'])
    return (country_contingency_table,)


@app.cell
def _(country_contingency_table):
    country_contingency_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before performing a Chi-Square test on the country data, one thing to note is the small sample size in churned EIRE. Since the sample size is less than 5, the Chi-Square test becomes inaccurate. One way to compensate for this is to upsample the data using the Monte Carlo method. This simulate thousands of random data point while preserving the overall distribution ratio.
    """)
    return


@app.cell
def _(country_contingency_table, stats):
    # Perform a Chi-Square test between Country and Churn
    country_chi_sq, country_p_val, country_dof, country_expected = stats.chi2_contingency(country_contingency_table,
                                                                                                                                correction = False,
                                                                                                                                method = stats.MonteCarloMethod())

    print(f'Chi-Square Statistic: {country_chi_sq:.4f}')
    print(f'P-value: {country_p_val:.4f}')

    country_alpha = 0.05

    if country_p_val < country_alpha:
        print('Reject the null hypothesis; Country and Churn are correlated')
    else:
        print('Cannot reject the null hypothesis; Country and Churn are not significantly correlated')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since I was able to reject the null hypothesis and deduce that `Country` and `Churn` are related, I can proceed on in creating the dummy variables for `Country`, which will be conducted later when making a preprocessing pipeline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `FirstInvoiceMonth`

    The next set of features I want to confirm if they have any statisical correlation with `Churn` are the `FirstInvoiceMonth`, `InvoiceMonth`, and `InvoiceDay`. Technically speaking, each of these features take on categorical values and can be analyzed similarly to `Country`. Again, the year of the first invoice for a customer and invoice year can be ignored due to the fact that the data is roughly a year long. First, I would need to get the month of a customers first invoice and turn it into a new column.
    """)
    return


@app.cell
def _(customer_features):
    # Create a column extracting the only the month of the first date of invoice
    customer_features['FirstInvoiceMonth'] = customer_features['FirstInvoiceDate'].dt.month
    return


@app.cell
def _(customer_features, pd):
    # Create a contingency table between FirstInvoiceMonth and Churn to get the frequency distribution between the two
    fim_contingency_table = pd.crosstab(customer_features['FirstInvoiceMonth'],
                                                        customer_features['Churn'])
    return (fim_contingency_table,)


@app.cell
def _(fim_contingency_table):
    fim_contingency_table
    return


@app.cell
def _(country_contingency_table, stats):
    # Perform a Chi-Square test between Country and Churn
    fim_chi_sq, fim_p_val, fim_dof, fim_expected = stats.chi2_contingency(country_contingency_table,
                                                                                                                correction = False,
                                                                                                                method = stats.MonteCarloMethod())

    print(f'Chi-Square Statistic: {fim_chi_sq:.4f}')
    print(f'P-value: {fim_p_val:.4f}')

    fim_alpha = 0.05

    if fim_p_val < fim_alpha:
        print('Reject the null hypothesis; FirstInvoiceMonth and Churn are correlated')
    else:
        print('Cannot reject the null hypothesis; FirstInvoiceMonth and Churn are not significantly correlated')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `InvoiceMonth`
    """)
    return


@app.cell
def _(customer_features, pd):
    # Create a contingency table between InvoiceMonth and Churn to get the frequency distribution between the two
    invmonth_contingency_table = pd.crosstab(customer_features['InvoiceMonth'],
                                                        customer_features['Churn'])
    return (invmonth_contingency_table,)


@app.cell
def _(invmonth_contingency_table):
    invmonth_contingency_table
    return


@app.cell
def _(invmonth_contingency_table, stats):
    # Perform a Chi-Square test between InvoiceMonth and Churn
    invmonth_chi_sq, invmonth_p_val, invmonth_dof, invmonth_expected = stats.chi2_contingency(invmonth_contingency_table,
                                                                                                                                    correction = False,
                                                                                                                                    method = stats.MonteCarloMethod())

    print(f'Chi-Square Statistic: {invmonth_chi_sq:.4f}')
    print(f'P-value: {invmonth_p_val:.4f}')

    invmonth_alpha = 0.05

    if invmonth_p_val < invmonth_alpha:
        print('Reject the null hypothesis; InvoiceMonth and Churn are correlated')
    else:
        print('Cannot reject the null hypothesis; InvoiceMonth and Churn are not significantly correlated')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `InvoiceDay`
    """)
    return


@app.cell
def _(customer_features, pd):
    # Create a contingency table between InvoiceDay and Churn to get the frequency distribution between the two
    invday_contingency_table = pd.crosstab(customer_features['InvoiceDay'],
                                                        customer_features['Churn'])
    return (invday_contingency_table,)


@app.cell
def _(invday_contingency_table):
    invday_contingency_table
    return


@app.cell
def _(invday_contingency_table, stats):
    # Perform a Chi-Square test between InvoiceDay and Churn
    invday_chi_sq, invday_p_val, invday_dof, invday_expected = stats.chi2_contingency(invday_contingency_table)

    print(f'Chi-Square Statistic: {invday_chi_sq:.4f}')
    print(f'P-value: {invday_p_val:.4f}')
    print(f'Degrees of Freedom: {invday_dof}')

    invday_alpha = 0.05

    if invday_p_val < invday_alpha:
        print('Reject the null hypothesis; InvoiceDay and Churn are correlated')
    else:
        print('Cannot reject the null hypothesis; InvoiceDay and Churn are not significantly correlated')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Collinearity Analysis

    Before getting to modeling, I want to see if there are any patterns or correlations between the features themselves and the target variable, `Churn`.
    """)
    return


@app.cell
def _(customer_features, pd):
    customer_dummies = pd.get_dummies(data = customer_features,
                                                columns = ['Country'])\
                                    .drop(['CustomerID','InvoiceDate','LastInvoiceDate',
                                                    'Churn Date','Cohort','FirstInvoiceDate'],
                                            axis = 1)
    return (customer_dummies,)


@app.cell
def _(customer_dummies):
    # Create a correlation matrix to find any correlations between features and churn
    customer_corr_matrix = customer_dummies.corr()
    return (customer_corr_matrix,)


@app.cell
def _(customer_corr_matrix, plt, sns):
    # Plot the correlation matrix
    plt.subplots(figsize = (10,8))

    sns.heatmap(customer_corr_matrix,
                        annot = True,
                        fmt = '.2f',
                        cmap = 'coolwarm',
                        center = 0,
                        linewidths = 1,
                        linecolor = 'black')

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Given the correlation matrix, we can see that some of the features have a strong correlation with `Churn`. Features such as the `InvoiceMonth`, `FirstInvoiceMonth`, `RollingNumInvoices`, and `DaysSinceLastPurchase` have the strongest correlations with `Churn` while other features like the country and `InvoiceDate` do not. This is solely for the Pearson correlation but as we saw earlier, the country seems to be statisically correlated to churn, it just may not be significant in certain models.

    There also seems to be some correlation between features like `RollingQuantity`, `RollingTotalPrice`, `RollingAvgDaysBetweenPurchases`, `DaysSinceLastPurchase`, and the different countries that may negatively impact the performance of the models if all of them are included. It's something to account for when developing models or handling beforehand. Handling it beforehand includes either removing features or reducing the dimensionality via principal component analysis.

    Another method for analyzing collinearity is with with variance inflation factoring. The variance inflation factor (VIF) a measure of how much a regression coefficient is inflated by other variables, on a scale from 1 to $\infty$. However, VIF only works with non-null numeric values which isn't true for `DaysSinceLastPurchase` and `RollingAvgDaysBetweenPurchases`. For that purpose, I will fill those null values with 0 solely for VIF analysis.

    Also, through the correlation matrix, I can start by removing features that are already highly correlated. For starters, I will remove `RollingQuantity`, `RollingAvgDaysBetweenPurchases`, and `Country_Other`.
    """)
    return


@app.cell
def _(customer_dummies, pd, variance_inflation_factor):
    # Instantiate a empty dataframe for VIF data
    vif_data = pd.DataFrame()

    # Create a copy of the customer_dummies dataframe
    customer_dummies_copy = customer_dummies.copy()

    # Fill in blanks values
    customer_dummies_copy.fillna(0,
                                inplace = True)

    # Drop unnecessary columns
    customer_dummies_copy.drop(['Churn','Country_Other',
                                        'RollingQuantity',
                                        'RollingAvgDaysBetweenPurchases'],
                                axis = 1,
                                inplace = True)

    # Convert all values to floats
    customer_dummies_numeric = customer_dummies_copy.astype(float)

    # Add an constant intercept column 
    customer_dummies_numeric['intercept'] = 1

    # Add a column of the different features
    vif_data['feature'] = customer_dummies_numeric.columns

    # Calculate the VIF for the features
    vif_data['VIF'] = [variance_inflation_factor(customer_dummies_numeric.values, i)
                        for i in range(len(customer_dummies_numeric.columns))]

    vif_data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After removing the features that had some degree of collinearity found in the correlation matrix, the remaining features don't appear to be significantly correlated with each other. If they had been, the VIF value for the given feature would be above 5, which isn't the case.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prediction Modeling
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Preprocessing Pipeline

    Now that I know which columns to keep and how the values should be formatted, I can create a data preprocessing pipeline to streamline data processing during modeling. This helps standardize the data being fed into the different models and reduce any data leakages.

    Also in terms of standardizing the data, I want to normalize all the features so they are all on the same scale. This helps improve the performance of machine learning models by ensuring that no single feature dominates the others due to its scale. This can be done with the Min-Max Scaler or StandardScaler. In this project, I'll go with StandardScaler.
    """)
    return


@app.cell
def _(country_counts):
    # Create a list of countries to keep
    keep_countries_list = country_counts[country_counts > 0.01].index
    return (keep_countries_list,)


@app.cell
def _(
    FunctionTransformer,
    Pipeline,
    StandardScaler,
    monthday_feature_names,
    split_date_monthday,
):
    date_monthday_transformer = FunctionTransformer(
                                                split_date_monthday,
                                                validate = False,
                                                feature_names_out = monthday_feature_names
                                                )

    date_monthday_scaled_pipeline = Pipeline([
                                        ('date_day',date_monthday_transformer),
                                        ('scaler',StandardScaler())
                                        ])
    return (date_monthday_scaled_pipeline,)


@app.cell
def _(
    FunctionTransformer,
    Pipeline,
    StandardScaler,
    month_feature_names,
    split_date_month,
):
    date_month_transformer = FunctionTransformer(
                                                split_date_month,
                                                validate = False,
                                                feature_names_out = month_feature_names
                                                )

    date_month_scaled_pipeline = Pipeline([
                                        ('date_month',date_month_transformer),
                                        ('scaler',StandardScaler())
                                        ])
    return (date_month_scaled_pipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The last steps towards preprocessing the data is to remove features that won't be necessary or incompatible with machine learning models and fill blank values in with **0**. That will fill in blank values under `DaysSinceLastPurchase` for new customers.
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    StandardScaler,
    date_month_scaled_pipeline,
    date_monthday_scaled_pipeline,
    keep_countries_list,
):
    drop_cols = ['InvoiceDate','LastInvoiceDate', 'Cohort',
                            'FirstInvoiceDate','Churn Date','CustomerID',
                            'RollingQuantity','RollingAvgDaysBetweenPurchases']

    num_cols = ['RollingTotalPrice','DaysSinceLastPurchase',
                            'RollingNumInvoices']

    cate_cols = ['Country']

    preprocessor = ColumnTransformer(
                        [
                        ('scale',StandardScaler(),num_cols),
                        ('onehotencode',OneHotEncoder(categories=[keep_countries_list],handle_unknown='ignore',sparse_output=False),cate_cols),
                        ('date_monthday',date_monthday_scaled_pipeline,['InvoiceDate']),
                        ('date_month',date_month_scaled_pipeline,['FirstInvoiceDate']),
                        ('remove_cols','drop',drop_cols)
                        ],
                        remainder='drop'
                    )
    return (preprocessor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Models

    There are a number of different methods to model this data to predict churn. Since the main objective is to classify where or not a customer will churn based upon varying data points, the main methods are a logistic regression, random forest, and XGBoost, each with their own benefits.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Before modeling, I want to split the data into a training set and a testing set. This will allow me to train a model on a portion of the data while testing a model on data it hasn't seen.
    """)
    return


@app.cell
def _(customer_features, train_test_split):
    # Split the data into a train and test set evenly with respect the churn
    # Ensure there's an even amount of churned invoices in both sets
    X = customer_features.drop('Churn',
                                        axis = 1)
    Y = customer_features['Churn']                                    

    x_train, x_test, y_train, y_test = train_test_split(X,
                                                                                            Y,
                                                                                            random_state = 333,
                                                                                            test_size = 0.3,
                                                                                            stratify = Y)
    return x_test, x_train, y_test, y_train


@app.cell
def _(KFold):
    # Set random state for cross validation for reproducability
    cv_splitter = KFold(n_splits = 5,
                        shuffle = True,
                        random_state= 133)
    return (cv_splitter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When evaluating the different models and cross validation, I'm going to focus on the recall score to determine which models perform best. The reason for choosing this method is because I believe it is important for the model to identify when churn will occur and do so accurately. The F2 score is another metric that takes into account the precision of a model but by weighing recall more heavily.
    """)
    return


@app.cell
def _(fbeta_score, make_scorer):
    # Make a scorer to use when cross validating models
    f2_scorer = make_scorer(fbeta_score, beta = 1)
    return (f2_scorer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Logistic Regression

    This first model I'm going to use is logistic regression. In order to hone in on the proper parameters, I'm going to perform grid search cross validation where I list out different parameter settings and will choose which parameters score highest based on a specified scoring method.
    """)
    return


@app.cell
def _(
    GridSearchCV,
    LogisticRegression,
    Pipeline,
    SimpleImputer,
    cv_splitter,
    f2_scorer,
    preprocessor,
):
    # Instantiate a logistic regression model
    logreg = LogisticRegression(random_state = 133)

    # Define pipeline with logistic regression model
    # Need to add an imputer to replace null values as logistic regression can't handle null values
    logreg_pipeline = Pipeline([('preprocess',preprocessor),
                                        ('zero_imputer',SimpleImputer(strategy = 'constant',fill_value=0)),
                                    ('logreg',logreg)]
                                )

    # m Define GridSearchCV parameters for logreg
    logreg_param_grid = [
                                                                    {'logreg__solver':['lbfgs'],'logreg__penalty':['l2'], 'logreg__C':[0.01,0.1,1]},
                                                                    {'logreg__solver':['saga','liblinear'],'logreg__penalty':['l1'], 'logreg__C':[0.01,0.1,1]},
                                                                    {'logreg__solver':['saga'],'logreg__penalty':['elasticnet'],'logreg__C':[0.01,0.1,1]}
                                                                    ]

    # Instantiate GridSearch LogReg model
    logreg_gs = GridSearchCV(estimator=logreg_pipeline,
                            param_grid=logreg_param_grid,
                            cv = cv_splitter,
                            scoring=f2_scorer)
    return (logreg_gs,)


@app.cell
def _(logreg_gs, x_test, x_train, y_train):
    # Fit the logistic regression model to the scaled training data
    logreg_gs.fit(x_train, y_train)

    # Print the best grid search parameters
    print(f'Best Parameters: {logreg_gs.best_params_}')

    # Print the best F2 score
    print(f'Best F2 Score: {logreg_gs.best_score_:.3f}')

    # Store the best grid search estimator
    logreg_best_estim = logreg_gs.best_estimator_

    # Used the fitted logistic regression model to predict using the test data
    logreg_gs_y_pred = logreg_best_estim.predict(x_test)

    # Calculate the probabilities for the predictions
    logreg_gs_y_pred_prob = logreg_best_estim.predict_proba(x_test)[:,1]
    return logreg_gs_y_pred, logreg_gs_y_pred_prob


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Random Forest
    This first model I'm going to use is logistic regression.
    """)
    return


@app.cell
def _(
    GridSearchCV,
    Pipeline,
    RandomForestClassifier,
    cv_splitter,
    f2_scorer,
    preprocessor,
):
    # Instantiate a logistic regression model
    rf = RandomForestClassifier(random_state = 133)

    # Define pipeline with random forest classifier
    rf_pipeline = Pipeline([
                            ('preprocess',preprocessor),
                            ('rf',rf)
                                ]
                            )

    # m Define GridSearchCV parameters for logreg
    rf_param_grid = {'rf__n_estimators':[100,250,500],
                                            'rf__max_depth':[10,25,50],
                                            'rf__min_samples_split':[2,5,10],
                                            }

    # Instantiate GridSearch LogReg model
    rf_gs = GridSearchCV(estimator=rf_pipeline,
                            param_grid=rf_param_grid,
                            cv = cv_splitter,
                            scoring=f2_scorer)
    return (rf_gs,)


@app.cell
def _(rf_gs, x_test, x_train, y_train):
    # Fit the logistic regression model to the scaled training data
    rf_gs.fit(x_train, y_train)

    # Print the best grid search parameters
    print(f'Best Parameters: {rf_gs.best_params_}')

    # Print the best F2 score
    print(f'Best F2 Score: {rf_gs.best_score_:.3f}')

    # Store the best grid search estimator
    rf_best_estim = rf_gs.best_estimator_

    # Used the fitted logistic regression model to predict using the test data
    rf_gs_y_pred = rf_best_estim.predict(x_test)

    # Calculate the probabilities for the predictions
    rf_gs_y_pred_prob = rf_best_estim.predict_proba(x_test)[:,1]
    return rf_gs_y_pred, rf_gs_y_pred_prob


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### XGBoost
    The next model is a XGBoost ensemble model which similar to random forest.
    """)
    return


@app.cell
def _(GridSearchCV, Pipeline, cv_splitter, f2_scorer, preprocessor, xgb):
    # Instantiate a logistic regression model
    xgb_class = xgb.XGBClassifier(objective='binary:logistic',
                                    random_state = 133)

    # Define pipeline with random forest classifier
    xgb_pipeline = Pipeline([
                            ('preprocess',preprocessor),
                            ('xgb',xgb_class)
                                ]
                            )

    # m Define GridSearchCV parameters for logreg
    xgb_param_grid = {'xgb__n_estimators':[100,250,500],
                                            'xgb__max_depth':[3,5,10,15],
                                            'xgb__learning_rate':[0.01,0.1,0.3],
                                            'xgb__min_child_weight':[1,5,10],
                                            'xgb__colsample_bytree':[0.5,0.75,1]
                                            }

    # Instantiate GridSearch LogReg model
    xgb_gs = GridSearchCV(estimator=xgb_pipeline,
                            param_grid=xgb_param_grid,
                            cv = cv_splitter,
                            scoring=f2_scorer)
    return xgb_gs, xgb_param_grid


@app.cell
def _(x_test, x_train, xgb_gs, y_train):
    # Fit the logistic regression model to the scaled training data
    xgb_gs.fit(x_train, y_train)

    # Print the best grid search parameters
    print(f'Best Parameters: {xgb_gs.best_params_}')

    # Print the best F2 score
    print(f'Best F2 Score: {xgb_gs.best_score_:.3f}')

    # Store the best grid search estimator
    xgb_best_estim = xgb_gs.best_estimator_

    # Used the fitted logistic regression model to predict using the test data
    xgb_gs_y_pred = xgb_best_estim.predict(x_test)

    # Calculate the probabilities for the predictions
    xgb_gs_y_pred_prob = xgb_best_estim.predict_proba(x_test)[:,1]
    return xgb_best_estim, xgb_gs_y_pred, xgb_gs_y_pred_prob


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### k-Nearest Neighbors
    """)
    return


@app.cell
def _(
    GridSearchCV,
    KNeighborsClassifier,
    Pipeline,
    SimpleImputer,
    cv_splitter,
    f2_scorer,
    preprocessor,
):
    # Instantiate a logistic regression model
    knn = KNeighborsClassifier()

    # Define kNN pipeline
    # Need to add an imputer to replace null values as knn can't handle null values
    knn_pipeline = Pipeline([
                                ('preprocess',preprocessor),
                                ('zero_imputer',SimpleImputer(strategy = 'constant',fill_value=0)),
                                ('knn',knn)
                                ]
                            )

    # m Define GridSearchCV parameters for logreg
    knn_param_grid = {'knn__n_neighbors':[5, 10, 20],
                                            'knn__weights':['uniform','distance']
                                            }

    # Instantiate GridSearch LogReg model
    knn_gs = GridSearchCV(estimator=knn_pipeline,
                            param_grid=knn_param_grid,
                            cv = cv_splitter,
                            scoring=f2_scorer)
    return (knn_gs,)


@app.cell
def _(knn_gs, x_test, x_train, y_train):
    # Fit the logistic regression model to the scaled training data
    knn_gs.fit(x_train, y_train)

    # Print the best grid search parameters
    print(f'Best Parameters: {knn_gs.best_params_}')

    # Print the best F2 score
    print(f'Best F2 Score: {knn_gs.best_score_:.3f}')

    # Store the best grid search estimator
    knn_best_estim = knn_gs.best_estimator_

    # Used the fitted logistic regression model to predict using the test data
    knn_gs_y_pred = knn_best_estim.predict(x_test)

    # Calculate the probabilities for the predictions
    knn_gs_y_pred_prob = knn_best_estim.predict_proba(x_test)[:,1]
    return knn_gs_y_pred, knn_gs_y_pred_prob


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Evaluations

    Here I want to compare the best models for each type and determine which one is able to predict churn the best. There are many ways to evaluate a models performance, such as accuracy, precision, ROC AUC, etc. However, in this scenario, the models ability to correctly predict a true churn customer event is most desirable, or in technical terms, the models recall performance.
    """)
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    logreg_gs_y_pred,
    logreg_gs_y_pred_prob,
    plt,
    roc_curve,
    sns,
    y_test,
):
    # Print out the model performance metrics
    logreg_cm = confusion_matrix(y_test, logreg_gs_y_pred)

    # Display the confusion matrix
    logreg_disp = ConfusionMatrixDisplay(logreg_cm)

    logreg_fig, (logreg_ax1, logreg_ax2) = plt.subplots(nrows = 1,
                                                    ncols = 2,
                                                    figsize = (10,5))

    sns.heatmap(logreg_cm,
                annot = True,
                fmt = 'd',
                cmap = 'Blues',
                center = 0,
                cbar = False,
                ax = logreg_ax1)

    logreg_ax1.set_title('Logistic Regression Confusion Matrix')
    logreg_ax1.set_xlabel('Predicted Label')
    logreg_ax1.set_ylabel('True Label')

    # Plot the ROC curve
    logreg_fpr, logreg_tpr, _ = roc_curve(y_test, logreg_gs_y_pred_prob)
    logreg_roc_auc = auc(logreg_fpr, logreg_tpr)
    logreg_ax2.plot(logreg_fpr,
                logreg_tpr)
    logreg_ax2.set_title(f'Logistic Regression ROC (AUC = {logreg_roc_auc:.2f})')

    plt.show()
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    plt,
    rf_gs_y_pred,
    rf_gs_y_pred_prob,
    roc_curve,
    sns,
    y_test,
):
    # Print out the model performance metrics
    rf_cm = confusion_matrix(y_test, rf_gs_y_pred)

    # Display the confusion matrix
    rf_disp = ConfusionMatrixDisplay(rf_cm)

    rf_fig, (rf_ax1, rf_ax2) = plt.subplots(nrows = 1,
                                                    ncols = 2,
                                                    figsize = (10,5))

    sns.heatmap(rf_cm,
                annot = True,
                fmt = 'd',
                cmap = 'Blues',
                center = 0,
                cbar = False,
                ax = rf_ax1)

    rf_ax1.set_title('Random Forest Confusion Matrix')
    rf_ax1.set_xlabel('Predicted Label')
    rf_ax1.set_ylabel('True Label')

    # Plot the ROC curve
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_gs_y_pred_prob)
    rf_roc_auc = auc(rf_fpr, rf_tpr)
    rf_ax2.plot(rf_fpr,
                rf_tpr)
    rf_ax2.set_title(f'Random Forest ROC (AUC = {rf_roc_auc:.2f})')

    plt.show()
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    plt,
    roc_curve,
    sns,
    xgb_gs_y_pred,
    xgb_gs_y_pred_prob,
    y_test,
):
    # Print out the model performance metrics
    xgb_cm = confusion_matrix(y_test, xgb_gs_y_pred)

    # Display the confusion matrix
    xgb_disp = ConfusionMatrixDisplay(xgb_cm)

    xgb_fig, (xgb_ax1, xgb_ax2) = plt.subplots(nrows = 1,
                                                    ncols = 2,
                                                    figsize = (10,5))

    sns.heatmap(xgb_cm,
                annot = True,
                fmt = 'd',
                cmap = 'Blues',
                center = 0,
                cbar = False,
                ax = xgb_ax1)

    xgb_ax1.set_title('XGBoost Confusion Matrix')
    xgb_ax1.set_xlabel('Predicted Label')
    xgb_ax1.set_ylabel('True Label')

    # Plot the ROC curve
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_gs_y_pred_prob)
    xgb_roc_auc = auc(xgb_fpr, xgb_tpr)
    xgb_ax2.plot(xgb_fpr,
                xgb_tpr)
    xgb_ax2.set_title(f'XGBoost ROC (AUC = {xgb_roc_auc:.2f})')

    plt.show()
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    knn_gs_y_pred,
    knn_gs_y_pred_prob,
    plt,
    roc_curve,
    sns,
    y_test,
):
    # Print out the model performance metrics
    knn_cm = confusion_matrix(y_test, knn_gs_y_pred)

    # Display the confusion matrix
    knn_disp = ConfusionMatrixDisplay(knn_cm)

    knn_fig, (knn_ax1, knn_ax2) = plt.subplots(nrows = 1,
                                                    ncols = 2,
                                                    figsize = (10,5))

    sns.heatmap(knn_cm,
                annot = True,
                fmt = 'd',
                cmap = 'Blues',
                center = 0,
                cbar = False,
                ax = knn_ax1)

    knn_ax1.set_title('kNN Confusion Matrix')
    knn_ax1.set_xlabel('Predicted Label')
    knn_ax1.set_ylabel('True Label')

    # Plot the ROC curve
    knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_gs_y_pred_prob)
    knn_roc_auc = auc(knn_fpr, knn_tpr)
    knn_ax2.plot(knn_fpr,
                knn_tpr)
    knn_ax2.set_title(f'kNN ROC (AUC = {knn_roc_auc:.2f})')

    plt.show()
    return


@app.cell
def _(classification_report, fbeta_score, logreg_gs_y_pred, y_test):
    print('Logistic Regression Classification Report:')
    print(classification_report(y_test, logreg_gs_y_pred))
    print(f'F2 Score: {fbeta_score(y_test, logreg_gs_y_pred, beta=2):.3f}')
    return


@app.cell
def _(classification_report, fbeta_score, rf_gs_y_pred, y_test):
    print('Random Forst Classification Report:')
    print(classification_report(y_test, rf_gs_y_pred))
    print(f'F2 Score: {fbeta_score(y_test, rf_gs_y_pred, beta=2):.3f}')
    return


@app.cell
def _(classification_report, fbeta_score, xgb_gs_y_pred, y_test):
    print('XGBoost Classification Report:')
    print(classification_report(y_test, xgb_gs_y_pred))
    print(f'F2 Score: {fbeta_score(y_test, xgb_gs_y_pred, beta=2):.3f}')
    return


@app.cell
def _(classification_report, fbeta_score, knn_gs_y_pred, y_test):
    print('k-Nearest Neighbors Classification Report:')
    print(classification_report(y_test, knn_gs_y_pred))
    print(f'F2 Score: {fbeta_score(y_test, knn_gs_y_pred, beta=2):.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As analyzed earlier in the notebook, nearly 21% of events resulted in churn. The logistic regression model produced here is marginally better than random guessing with a recall rate of 27%. The k-nearest neighbor model performed considerably better at 45% recall rate. Lastly, the best random forest model and XGBoost model were nearly identical with both having 48% recall rates. However, the XGBoost model has better F1 and F2 scores, which place higher emphasis on correct recall predictions, making it the model of choice moving foward.

    # XGBoost Model

    Now that I have a model that is able to perform well, I want to further tune and analyze the model to see what it does well to hopefully improve on its performance.

    ## Prediction Analysis
    """)
    return


@app.cell
def _(pd, xgb_best_estim):
    xgb_feature_importance = pd.DataFrame({'feature_name':xgb_best_estim[:-1].get_feature_names_out(),
                                            'importance':xgb_best_estim[-1].feature_importances_})\
                                .sort_values('importance',
                                            ascending = False)
    return (xgb_feature_importance,)


@app.cell
def _(xgb_feature_importance):
    xgb_feature_importance
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adjusted Model

    Based upon the XGBoost models feature importances, it appears that a customers rolling number of invoices is a key feature when the model is predicting churn. On the other end of the spectrum, it looks like the customers country and the invoice month aren't as important in this model. In order to tune the model, I want to see how removing some of these features might affect the performance.
    """)
    return


@app.cell
def _(
    FunctionTransformer,
    Pipeline,
    StandardScaler,
    day_feature_names,
    split_date_day,
):
    date_day_transformer = FunctionTransformer(
                                                split_date_day,
                                                validate = False,
                                                feature_names_out = day_feature_names
                                                )

    date_day_scaled_pipeline = Pipeline([
                                        ('date_day',date_day_transformer),
                                        ('scaler',StandardScaler())
                                        ])
    return (date_day_scaled_pipeline,)


@app.cell
def _(
    ColumnTransformer,
    StandardScaler,
    date_day_scaled_pipeline,
    date_month_scaled_pipeline,
):
    # Create a new preprocessor for a new set of features to be used in a XGBoost model
    adj_drop_cols = ['InvoiceDate','LastInvoiceDate', 'Cohort',
                            'FirstInvoiceDate','Churn Date','CustomerID',
                            'RollingQuantity','RollingAvgDaysBetweenPurchases',
                            'Country']

    adj_num_cols = ['RollingTotalPrice','DaysSinceLastPurchase',
                            'RollingNumInvoices']

    adj_preprocessor = ColumnTransformer(
                        [
                        ('scale',StandardScaler(),adj_num_cols),
                        ('date_day',date_day_scaled_pipeline,['InvoiceDate']),
                        ('date_month',date_month_scaled_pipeline,['FirstInvoiceDate']),
                        ('remove_cols','drop',adj_drop_cols)
                        ],
                        remainder='drop'
                    )
    return (adj_preprocessor,)


@app.cell
def _(
    GridSearchCV,
    Pipeline,
    adj_preprocessor,
    cv_splitter,
    f2_scorer,
    xgb,
    xgb_param_grid,
):
    # Instantiate a logistic regression model
    adj_xgb_class = xgb.XGBClassifier(objective='binary:logistic',
                                    random_state = 133)

    # Define pipeline with random forest classifier
    adj_xgb_pipeline = Pipeline([
                            ('preprocess', adj_preprocessor),
                            ('xgb',adj_xgb_class)
                                ]
                            )

    # Instantiate GridSearch LogReg model
    adj_xgb_gs = GridSearchCV(estimator=adj_xgb_pipeline,
                            param_grid=xgb_param_grid,
                            cv = cv_splitter,
                            scoring=f2_scorer)
    return (adj_xgb_gs,)


@app.cell
def _(adj_xgb_gs, x_test, x_train, y_train):
    # Fit the XGBoost model to the scaled training data
    adj_xgb_gs.fit(x_train, y_train)

    # Print the best grid search parameters
    print(f'Best Parameters: {adj_xgb_gs.best_params_}')

    # Print the best F2 score
    print(f'Best F2 Score: {adj_xgb_gs.best_score_:.3f}')

    # Store the best grid search estimator
    adj_xgb_best_estim = adj_xgb_gs.best_estimator_

    # Used the fitted XGBoost model to predict using the test data
    adj_xgb_gs_y_pred = adj_xgb_best_estim.predict(x_test)

    # Calculate the probabilities for the predictions
    adj_xgb_gs_y_pred_prob = adj_xgb_best_estim.predict_proba(x_test)[:,1]
    return adj_xgb_gs_y_pred, adj_xgb_gs_y_pred_prob


@app.cell
def _(
    ConfusionMatrixDisplay,
    adj_xgb_gs_y_pred,
    adj_xgb_gs_y_pred_prob,
    auc,
    confusion_matrix,
    plt,
    roc_curve,
    sns,
    y_test,
):
    # Print out the model performance metrics
    adj_xgb_cm = confusion_matrix(y_test, adj_xgb_gs_y_pred)

    # Display the confusion matrix
    adj_xgb_disp = ConfusionMatrixDisplay(adj_xgb_cm)

    adj_xgb_fig, (adj_xgb_ax1, adj_xgb_ax2) = plt.subplots(nrows = 1,
                                                    ncols = 2,
                                                    figsize = (10,5))

    sns.heatmap(adj_xgb_cm,
                annot = True,
                fmt = 'd',
                cmap = 'Blues',
                center = 0,
                cbar = False,
                ax = adj_xgb_ax1)

    adj_xgb_ax1.set_title('Adjusted XXGBoost Confusion Matrix')
    adj_xgb_ax1.set_xlabel('Predicted Label')
    adj_xgb_ax1.set_ylabel('True Label')

    # Plot the ROC curve
    adj_xgb_fpr, adj_xgb_tpr, _ = roc_curve(y_test, adj_xgb_gs_y_pred_prob)
    adj_xgb_roc_auc = auc(adj_xgb_fpr, adj_xgb_tpr)
    adj_xgb_ax2.plot(adj_xgb_fpr,
                adj_xgb_tpr)
    adj_xgb_ax2.set_title(f'Adjusted XGBoost ROC (AUC = {adj_xgb_roc_auc:.2f})')

    plt.show()
    return


@app.cell
def _(adj_xgb_gs_y_pred, classification_report, fbeta_score, y_test):
    print('Adjusted XGBoost Classification Report:')
    print(classification_report(y_test, adj_xgb_gs_y_pred))
    print(f'F2 Score: {fbeta_score(y_test, adj_xgb_gs_y_pred, beta=2):.3f}')
    return


@app.cell
def _(mo):
    mo.md(r"""
    When creating another XGBoost model without less important features, the performance didn't improve or perform better than with the features or the random forest model, but now I want to further tune the original XGBoost model to see if I can gain any further performance improvements.

    ## Tuned Model
    """)
    return


@app.cell
def _(GridSearchCV, Pipeline, cv_splitter, preprocessor, xgb):
    # Instantiate a logistic regression model
    tuned_xgb_class = xgb.XGBClassifier(objective='binary:logistic',
                                    random_state = 133)

    # Define pipeline with random forest classifier
    tuned_xgb_pipeline = Pipeline([
                            ('preprocess',preprocessor),
                            ('xgb',tuned_xgb_class)
                                ]
                            )

    # m Define GridSearchCV parameters for logreg
    tuned_xgb_param_grid = {'xgb__n_estimators':[75,100,125,150],
                                            'xgb__max_depth':[3,4,5],
                                            'xgb__learning_rate':[0.05,0.1,0.2],
                                            'xgb__min_child_weight':[8,10,12],
                                            'xgb__colsample_bytree':[0.8,0.9,1]
                                            }

    # Instantiate GridSearch LogReg model
    tuned_xgb_gs = GridSearchCV(estimator=tuned_xgb_pipeline,
                            param_grid=tuned_xgb_param_grid,
                            cv = cv_splitter,
                            scoring='recall')
    return (tuned_xgb_gs,)


@app.cell
def _(tuned_xgb_gs, x_test, x_train, y_train):
    # Fit the logistic regression model to the scaled training data
    tuned_xgb_gs.fit(x_train, y_train)

    # Print the best grid search parameters
    print(f'Best Parameters: {tuned_xgb_gs.best_params_}')

    # Print the best Recall score
    print(f'Best Recall Score: {tuned_xgb_gs.best_score_:.3f}')

    # Store the best grid search estimator
    tuned_xgb_best_estim = tuned_xgb_gs.best_estimator_

    # Used the fitted logistic regression model to predict using the test data
    tuned_xgb_gs_y_pred = tuned_xgb_best_estim.predict(x_test)

    # Calculate the probabilities for the predictions
    tuned_xgb_gs_y_pred_prob = tuned_xgb_best_estim.predict_proba(x_test)[:,1]
    return tuned_xgb_gs_y_pred, tuned_xgb_gs_y_pred_prob


@app.cell
def _(
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    plt,
    roc_curve,
    sns,
    tuned_xgb_gs_y_pred,
    tuned_xgb_gs_y_pred_prob,
    y_test,
):
    # Print out the model performance metrics
    tuned_xgb_cm = confusion_matrix(y_test, tuned_xgb_gs_y_pred)

    # Display the confusion matrix
    tuned_xgb_disp = ConfusionMatrixDisplay(tuned_xgb_cm)

    tuned_xgb_fig, (tuned_xgb_ax1, tuned_xgb_ax2) = plt.subplots(nrows = 1,
                                                    ncols = 2,
                                                    figsize = (10,5))

    sns.heatmap(tuned_xgb_cm,
                annot = True,
                fmt = 'd',
                cmap = 'Blues',
                center = 0,
                cbar = False,
                ax = tuned_xgb_ax1)

    tuned_xgb_ax1.set_title('Tuned XGBoost Confusion Matrix')
    tuned_xgb_ax1.set_xlabel('Predicted Label')
    tuned_xgb_ax1.set_ylabel('True Label')

    # Plot the ROC curve
    tuned_xgb_fpr, tuned_xgb_tpr, _ = roc_curve(y_test, tuned_xgb_gs_y_pred_prob)
    tuned_xgb_roc_auc = auc(tuned_xgb_fpr, tuned_xgb_tpr)
    tuned_xgb_ax2.plot(tuned_xgb_fpr,
                tuned_xgb_tpr)
    tuned_xgb_ax2.set_title(f'Tuned XGBoost ROC (AUC = {tuned_xgb_roc_auc:.2f})')

    plt.show()
    return


@app.cell
def _(classification_report, fbeta_score, tuned_xgb_gs_y_pred, y_test):
    print('Tuned XGBoost Classification Report:')
    print(classification_report(y_test, tuned_xgb_gs_y_pred))
    print(f'F2 Score: {fbeta_score(y_test, tuned_xgb_gs_y_pred, beta=2):.3f}')
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Model Deployment

    After looking at different ways to model this data to predict churn, it looks like the XGBoost model with the following parameters performed the best on the test set:
    - 'colsample_bytree' = 1
    - 'xgb__learning_rate' = 0.1
    - 'xgb__max_depth' = 5
    - 'xgb__min_child_weight' = 5
    - 'xgb__n_estimators' = 100

    Now I want to be able to deploy this model for use on additional data
    """)
    return


@app.cell
def _(xgb_best_estim):
    import joblib

    joblib.dump(xgb_best_estim,
                '../Models/XGBoost/xgb_best_estim.joblib')
    return


if __name__ == "__main__":
    app.run()
