import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Overview

    This notebook is a data-quality and preprocessing walkthrough for the Online Retail dataset.

    The workflow follows four main steps:
    1. Load the raw Excel file and inspect the schema.
    2. Review the general data profile using `info()` and `describe()`.
    3. Investigate inconsistent or missing text fields such as `Description` and `CustomerID`.
    4. Export a cleaned workbook that can be used in downstream exploratory analysis.

    Important notebook note:
    - The notebook is intentionally diagnostic and exploratory.
    - Several cleaning decisions are based on business interpretation rather than a strict one-size-fits-all rule.
    - Where a field cannot be confidently reconstructed, the notebook preserves the original data and documents the limitation.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy
    import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter

    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data

    The data was donated on November 5, 2015 to University of California Irvine Machine Learning repository and can be found at the URL link `https://archive.ics.uci.edu/dataset/352/online+retail`.

    It's stated in the documentation that this data is compiled of transactions occurring between January 12, 2010 to September 12, 2011 for UK-based and registered non-store online retail with 541,909 instances with no missing values.
    """)
    return


@app.cell
def _(pd):
    # Read Excel data
    data = pd.read_excel('./Data/online+retail/Online Retail.xlsx',
                        header = 0)
    return (data,)


@app.cell
def _(data):
    data.info()
    return


@app.cell
def _(data):
    data.describe()
    return


@app.cell
def _(data):
    data.head(5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Dates appear to span from December 1, 2010 to December 9, 2011.
    - The initial inspection also reveals missing values in the `Description` and `CustomerID` columns.
    - This may reflect either genuinely incomplete data or non-standard transaction records that should be treated carefully during analysis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Cleaning
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Although the dataset documentation explicitly states there are no missing values, the notebook shows missing records in the `Description` and `CustomerID` features.

    This apparent contradiction suggests one of two possibilities:
    - the documentation is simplified or incomplete; or
    - the dataset contains records that are not strictly missing but are represented by unusual values or inconsistent business context.

    The goal here is not to force-clean every row blindly, but to document the pattern and keep any assumptions visible in the notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Missing/Multiple `Description`

    The notebook tests a simple repair strategy: try to recover missing descriptions from the `StockCode` field using other rows that share the same item code.

    This is a practical approach, but it is limited by a key observation from the analysis:
    - many `StockCode` values are associated with more than one distinct `Description`
    - those descriptions are often inconsistent because the source data is user-entered text, not a controlled catalog

    So the notebook treats `StockCode` as the more reliable identifier when the descriptive text is noisy.
    """)
    return


@app.cell
def _(data):
    # Isolate null Descriptions
    null_des = data[data['Description'].isnull()]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before moving forward, we need to figure out if there is a 1-to-1 relationship between the `StockCode` and `Description` features.
    """)
    return


@app.cell
def _(data):
    # Use the groupby function to get the number of unique Descriptions by StockCode
    stock_code_desc_nunique = data.groupby('StockCode')[['Description']].nunique()
    return (stock_code_desc_nunique,)


@app.cell
def _(stock_code_desc_nunique):
    stock_code_desc_nunique[stock_code_desc_nunique['Description'] > 1].sort_values('Description', ascending = False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It does appear there are 650 `StockCode` values with more than one unique `Description`. The analysis then checks whether these descriptions are similar enough to standardize.

    The result is important for the overall methodology:
    - uppercasing the descriptions does not resolve the inconsistency
    - the text itself is not standardized and appears to be free-form user input

    This means the notebook avoids overfitting a fragile text-cleaning strategy and instead keeps the methodology transparent.
    """)
    return


@app.cell
def _(data):
    # Are there any similarities in the "extraneous" descriptions?
    data[data['StockCode'].isin([20713,23084,85175,21830,21181])]['Description'].unique()
    return


@app.cell
def _(data):
    # How do the numbers change when we capitalize all the descriptions?
    data['Description'] = data['Description'].str.upper()
    return


@app.cell
def _(data):
    # Use the groupby function to get the number of unique Descriptions by StockCode
    stock_code_desc_nunique_upper = data.groupby('StockCode')[['Description']].nunique()
    return (stock_code_desc_nunique_upper,)


@app.cell
def _(stock_code_desc_nunique_upper):
    stock_code_desc_nunique_upper[stock_code_desc_nunique_upper['Description'] > 1].sort_values('Description', ascending = False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By capitalizing all the letters in the descriptions, it didn't change the total number of `StockCode` that have more than 1 unique `Description`. The description themselves are not standardized to follow a distinguished convention and are simply user input, therefore cleaning up the descriptions may not be time efficient.

    I do want to investigate what is happening to the descriptions that contain "wrong" and "found".
    """)
    return


@app.cell
def _(data):
    wrong_desc = data[(data['Description'].str.contains('WRONG'))
                        & (data['Description'].notnull())]

    wrong_desc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Descriptions that contain the word "WRONG" seem to be clerical/operational errors. Each instance seems to have corresponding return/refund and re-purchase/correction quantities which should balance each other out when doing further analysis.
    """)
    return


@app.cell
def _(data):
    found_desc = data[(data['Description'].str.contains('FOUND'))
                        & (data['Description'].notnull())]

    found_desc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    "Found" instances are more interesting, especially the record where a quantity of `-9` was marked as found. This suggests the description field can contain operational or correction notes rather than a clean item name.

    For downstream analysis, the notebook therefore treats these cases as exception records and keeps the analysis context visible.

    Before making any final assumptions, the notebook checks how many `StockCode` values still have multiple descriptions after excluding obvious `FOUND`, `WRONG`, and blank cases.
    """)
    return


@app.cell
def _(data):
    # Remove items that have found, wrong, and blank descriptions
    maybe_clean_desc = data[(~data['Description'].str.contains('FOUND', na = False))
                            & ~(data['Description'].str.contains('WRONG', na = False))
                            & (data['Description'].notnull())]\
                        .groupby('StockCode')\
                        [['Description']]\
                        .nunique()
    return (maybe_clean_desc,)


@app.cell
def _(maybe_clean_desc):
    # How many unique items have more than one description now?
    maybe_clean_desc[maybe_clean_desc['Description'] > 1]\
        .sort_values('Description',
                     ascending = False)
    return


@app.cell
def _(data):
    # How do the top 5 offenders descriptions look?
    data[data['StockCode'].isin([23084,21830,21181,23131,'72807A'])]['Description'].unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When removing items that do not contain `WRONG`, `FOUND`, or null descriptions, 634 items still have more than one unique description. That indicates the remaining inconsistency is likely due to inconsistent user input, spelling variations, and non-standard catalog text.

    Practical takeaway:
    - `Description` is not a stable key for item identity.
    - `StockCode` is the more reliable field for distinguishing products.
    - Any analysis that depends on description text should be treated as exploratory unless a stronger product identifier is introduced.
    ___
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Null `CustomerID`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, the notebook looks for patterns in the records where `CustomerID` is missing.

    This is important because missing customer identifiers often signal a different record type, such as:
    - internal corrections
    - stock adjustments
    - supplier or warehouse movement
    - return/refund-related administrative records

    The notebook does not assume the missing `CustomerID` values can be safely imputed without external context.
    """)
    return


@app.cell
def _(data):
    missing_custom = data[data['CustomerID'].isna()]
    return (missing_custom,)


@app.cell
def _(missing_custom):
    missing_custom
    return


@app.cell
def _(data, missing_custom):
    print('Percent of Invoice Numbers with missing Customer ID: {:.2f}'.format(missing_custom['InvoiceNo'].nunique()/data['InvoiceNo'].nunique()))
    return


@app.cell
def _(data, missing_custom):
    print('Percent of Invoice Dates with missing Customer ID: {:.2f}'.format(missing_custom['InvoiceDate'].nunique()/data['InvoiceDate'].nunique()))
    return


@app.cell
def _(data, missing_custom):
    print('Percent of StockCodes with missing Customer ID: {:.2f}'.format(missing_custom['StockCode'].nunique()/data['StockCode'].nunique()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Instances where the `CustomerID` is missing do not appear to be specific to any one `StockCode`, which suggests the issue affects many products and may correspond to operational or clerical correction activity rather than a single product category.

    The lower percentages for `InvoiceNo` and `InvoiceDate` may indicate a subset of unusual or correction-style transactions. Because the notebook does not have enough supporting context to assign a missing customer reliably, it leaves those records as-is and documents the uncertainty.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Export Data for Analysis

    The final code cell exports the working dataset to a cleaned workbook for use in downstream analysis.

    This export is intentionally simple and transparent:
    - it preserves the original data structure
    - it writes out the current notebook state for later analysis
    - it leaves the cleaning rationale in the notebook narrative rather than silently mutating the data in code
    """)
    return


@app.cell
def _(data):
    data.to_excel('./Data/Online Retail_clean.xlsx',
                  index = False)
    return


if __name__ == "__main__":
    app.run()
