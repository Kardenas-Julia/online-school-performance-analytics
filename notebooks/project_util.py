# -*- coding: utf-8 -*-
"""
Helper functions for the Online School Performance Analytics project.

This module contains reusable functions for:
- data cleaning;
- dataset structure checks;
- descriptive statistics;
- basic business metric calculations;
- summary table preparation for analysis.

The functions are used across the analytical notebooks to keep the project
logic consistent and avoid duplicating utility code.
"""

import re

import numpy as np
import pandas as pd


# General helper functions

def safe_divide(numerator, denominator):
    """
    Perform safe division.

    If the denominator is zero or missing, return NaN.
    This is used for conversion rates, CPL, CAC, Revenue / Spend
    and other metrics where division by zero may occur.
    """
    return np.where(
        (denominator != 0) & pd.notna(denominator),
        numerator / denominator,
        np.nan
    )


def to_percent(series, decimals=2):
    """
    Convert a ratio to a percentage and round the result.

    Example:
    0.1534 -> 15.34
    """
    return (series * 100).round(decimals)


def dict_to_check_df(data, key_col='Check', value_col='Result'):
    """
    Convert a dictionary into a two-column DataFrame.

    This is used for readable data quality checks and summary outputs.
    """
    return pd.DataFrame(
        data.items(),
        columns=[key_col, value_col]
    )


# Text and CRM identifier cleaning functions

def clean_text_column(series):
    """
    Clean a text column.

    Steps:
    - convert values to pandas string type;
    - remove leading and trailing spaces;
    - replace empty strings with pd.NA.

    Used for categorical and text fields such as:
    Source, Campaign, Product, City, Stage and others.
    """
    cleaned = (
        series
        .astype('string')
        .str.strip()
    )

    cleaned = cleaned.replace('', pd.NA)

    return cleaned


def clean_id_column(series):
    """
    Clean CRM identifiers.

    CRM IDs are not numeric metrics, so they are processed as strings.

    Steps:
    - convert values to pandas string type;
    - remove extra spaces;
    - remove the trailing .0 if an ID was read as a float;
    - replace technical empty values with pd.NA.

    Used for:
    - Id;
    - CONTACTID;
    - Contact Name.
    """
    cleaned = (
        series
        .astype('string')
        .str.strip()
        .str.replace(r'\.0$', '', regex=True)
    )

    cleaned = cleaned.replace(
        ['', 'nan', 'NaN', 'None', '<NA>'],
        pd.NA
    )

    return cleaned


# Financial value cleaning functions

def clean_money_value(value):
    """
    Clean a single monetary value and convert it to float.

    Handles:
    - currency symbols: €, $;
    - spaces and non-breaking spaces;
    - European number format: 3.500,00;
    - US number format: 3,500.00;
    - comma as decimal separator: 3500,00.

    If the value cannot be converted to a number, return NaN.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == '':
        return np.nan

    value = (
        value
        .replace('€', '')
        .replace('$', '')
        .replace(' ', '')
        .replace('\xa0', '')
    )

    # If both comma and dot are present, detect the format by the last separator.
    if ',' in value and '.' in value:
        # European format: 3.500,00 -> 3500.00
        if value.rfind(',') > value.rfind('.'):
            value = value.replace('.', '').replace(',', '.')
        # US format: 3,500.00 -> 3500.00
        else:
            value = value.replace(',', '')

    # Format: 3500,00 -> 3500.00
    elif ',' in value:
        value = value.replace(',', '.')

    try:
        return float(value)
    except ValueError:
        return np.nan


def clean_money_column(series):
    """
    Clean a monetary column and convert values to numeric format.

    Used for:
    - Initial Amount Paid;
    - Offer Total Amount;
    - other monetary fields when needed.
    """
    return series.apply(clean_money_value)


# German language level standardization

def standardize_deutsch_level(value):
    """
    Standardize the German language level field.

    Returns one of the following values:
    - A1
    - A2
    - B1
    - B2
    - C1
    - C2
    - Unknown

    If the field contains free text, the function searches for a language
    level inside the text. If the level cannot be identified, it returns Unknown.
    """
    if pd.isna(value):
        return 'Unknown'

    value_str = str(value).strip().upper()

    if value_str == '':
        return 'Unknown'

    levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

    for level in levels:
        if re.search(rf'\b{level}\b', value_str):
            return level

    return 'Unknown'


# Descriptive statistics functions

def numeric_summary(df, columns):
    """
    Calculate descriptive statistics for selected numeric columns.

    For each column, the function calculates:
    - number of non-missing values;
    - mean;
    - median;
    - mode;
    - minimum;
    - maximum;
    - range.

    Technical identifiers should not be passed to this function.
    """
    summary = []

    for col in columns:
        data = df[col].dropna()

        mode_series = data.mode()
        mode_value = mode_series.iloc[0] if not mode_series.empty else np.nan

        summary.append({
            'Field': col,
            'Non-Missing Values': data.count(),
            'Mean': data.mean(),
            'Median': data.median(),
            'Mode': mode_value,
            'Minimum': data.min(),
            'Maximum': data.max(),
            'Range': data.max() - data.min()
        })

    return pd.DataFrame(summary)


def categorical_summary(df, column, top_n=None):
    """
    Calculate a frequency distribution for a categorical column.

    For each value, the function calculates:
    - number of rows;
    - share of total rows.

    If top_n is provided, only the top_n values are returned.
    """
    summary = (
        df[column]
        .value_counts(dropna=False)
        .reset_index()
    )

    summary.columns = [column, 'Count']

    summary['Share, %'] = (
        summary['Count'] / summary['Count'].sum() * 100
    ).round(2)

    if top_n is not None:
        summary = summary.head(top_n)

    return summary


# Deal analysis functions

def deals_group_summary(df, group_col):
    """
    Calculate basic deal metrics grouped by the selected field.

    Metrics:
    - number of deals;
    - number of paid deals;
    - payment conversion;
    - revenue;
    - average revenue.

    Used for analysis by:
    - Source;
    - Campaign;
    - Product;
    - Deal Owner Name;
    - City;
    - Level of Deutsch.
    """
    summary = (
        df
        .groupby(group_col, dropna=False)
        .agg(
            deals_count=('Id', 'count'),
            paid_deals=('Is Paid', 'sum'),
            revenue=('Revenue', 'sum'),
            avg_revenue=('Revenue', 'mean')
        )
        .reset_index()
    )

    summary['Payment Conversion, %'] = (
        summary['paid_deals'] / summary['deals_count'] * 100
    ).round(2)

    summary = summary.rename(columns={
        'deals_count': 'Deals Count',
        'paid_deals': 'Paid Deals',
        'revenue': 'Revenue',
        'avg_revenue': 'Avg Revenue'
    })

    summary = summary.sort_values(
        ['Paid Deals', 'Revenue'],
        ascending=False
    )

    return summary


# Call analysis functions

def calls_group_summary(df, group_col):
    """
    Calculate basic call metrics grouped by the selected field.

    Metrics:
    - number of calls;
    - average call duration;
    - median call duration.

    Used to analyze sales team call activity.
    """
    summary = (
        df
        .groupby(group_col, dropna=False)
        .agg(
            calls_count=('Id', 'count'),
            avg_call_duration=('Call Duration (in seconds)', 'mean'),
            median_call_duration=('Call Duration (in seconds)', 'median')
        )
        .reset_index()
    )

    summary = summary.rename(columns={
        'calls_count': 'Calls Count',
        'avg_call_duration': 'Avg Call Duration',
        'median_call_duration': 'Median Call Duration'
    })

    summary = summary.sort_values(
        'Calls Count',
        ascending=False
    )

    return summary


# Marketing metric functions

def spend_group_summary(df, group_col):
    """
    Calculate basic marketing metrics grouped by the selected field.

    Metrics:
    - spend;
    - impressions;
    - clicks;
    - CTR;
    - CPC.

    Used for analysis by:
    - Source;
    - Campaign.
    """
    summary = (
        df
        .groupby(group_col, dropna=False)
        .agg(
            spend=('Spend', 'sum'),
            impressions=('Impressions', 'sum'),
            clicks=('Clicks', 'sum')
        )
        .reset_index()
    )

    summary['CTR, %'] = np.where(
        summary['impressions'] > 0,
        summary['clicks'] / summary['impressions'] * 100,
        np.nan
    )

    summary['CPC'] = np.where(
        summary['clicks'] > 0,
        summary['spend'] / summary['clicks'],
        np.nan
    )

    summary = summary.rename(columns={
        'spend': 'Spend',
        'impressions': 'Impressions',
        'clicks': 'Clicks'
    })

    summary['CTR, %'] = summary['CTR, %'].round(2)
    summary['CPC'] = summary['CPC'].round(2)

    summary = summary.sort_values(
        'Spend',
        ascending=False
    )

    return summary


# Dataset structure checks

def check_required_columns(dataframes, required_columns):
    """
    Check whether each dataset contains all required columns.

    Parameters
    ----------
    dataframes : dict
        Dictionary with DataFrames.
        Example:
        {
            'deals': deals,
            'calls': calls
        }

    required_columns : dict
        Dictionary with required columns for each dataset.
        Example:
        {
            'deals': ['Id', 'Stage', 'Revenue'],
            'calls': ['Id', 'Call Start Time']
        }

    Returns
    -------
    DataFrame
        Table with missing columns by dataset.
    """
    check_result = []

    for df_name, columns in required_columns.items():
        missing_columns = [
            col for col in columns
            if col not in dataframes[df_name].columns
        ]

        check_result.append({
            'Table': df_name,
            'Missing Columns': missing_columns,
            'Missing Columns Count': len(missing_columns)
        })

    return pd.DataFrame(check_result)


# Cleaning result checks

def cleaning_summary(raw_df, clean_df, dataset_name):
    """
    Show a short comparison of a dataset before and after cleaning.

    Used after cleaning each table:
    - Spend;
    - Contacts;
    - Calls;
    - Deals.
    """
    summary = pd.DataFrame({
        'Metric': [
            'Rows before cleaning',
            'Rows after cleaning',
            'Columns before cleaning',
            'Columns after cleaning'
        ],
        'Value': [
            raw_df.shape[0],
            clean_df.shape[0],
            raw_df.shape[1],
            clean_df.shape[1]
        ]
    })

    print(f'Dataset summary: {dataset_name}')
    return summary
