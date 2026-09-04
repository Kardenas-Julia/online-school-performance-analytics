# Online School Performance Analytics

End-to-end business analytics project for an online school: data preparation, exploratory data analysis, marketing and sales performance analysis, product analytics, unit economics, growth hypothesis and interactive dashboard.

## Project Overview

This project analyzes the performance of an online school using CRM, sales, calls and marketing spend data.

The goal of the project is to identify key factors affecting business performance and find manageable growth opportunities across the sales funnel, marketing channels, products, payments and unit economics.

The project was originally completed as a final Data Analytics project and is being adapted for an international portfolio.

## Business Questions

The analysis focuses on the following questions:

* How many deals are created and how many convert to payment?
* Which marketing sources and campaigns generate deals, paid deals and revenue?
* How effective is the sales team across deal volume, conversion and revenue?
* Which products and payment types drive the main financial result?
* What data quality limitations affect geography and language analysis?
* What does unit economics show by product?
* Which metric can be used as a manageable growth point?
* What hypothesis can be tested within 14 days?

## Project Workflow

The project follows an end-to-end analytics workflow:

1. Data audit and data quality assessment
2. Data cleaning and preparation
3. Business rules definition
4. Descriptive statistics and EDA
5. Sales funnel and time series analysis
6. Marketing source and campaign analysis
7. Sales team performance analysis
8. Product, payment and education type analysis
9. Geography and language data quality analysis
10. Financial analysis
11. Unit economics calculation
12. Growth point selection and HADI hypothesis
13. Interactive dashboard development

## Tools and Technologies

* Python
* pandas
* numpy
* plotly
* Dash
* dash-bootstrap-components
* Google Colab
* PyCharm
* GitHub

## Repository Structure

```text
online-school-performance-analytics/
├── README.md
├── requirements.txt
├── notebooks/
├── dashboard/
├── data/
│   └── processed/
├── docs/
└── reports/
```

## Data

The project uses cleaned and processed datasets prepared during the data cleaning stage.

Main processed files include:

* `deals_clean.csv`
* `contacts_clean.csv`
* `calls_clean.csv`
* `spend_clean.csv`

Additional output tables are used for unit economics, growth scenarios and dashboard visualizations.

## Documentation

The `docs/` folder contains supporting project documentation:

- `data_dictionary_en.xlsx` — editable data dictionary with field descriptions, data types, valid values, cleaning rules and analysis usage notes.
- `data_dictionary_en.pdf` — PDF version for quick review.

## Key Business Rules

Several business rules were defined before analysis:

* The main analysis table is `deals_clean.csv`.
* A paid deal is defined as `Stage = Payment Done`.
* Actual revenue is calculated using `Initial Amount Paid` only for paid deals.
* `Offer Total Amount` is not treated as actual payment revenue.
* `Source` is used as the main level for marketing analysis.
* `Campaign` is used as an additional level of marketing detail.
* Product, payment type and education type analysis is mainly based on paid deals.

## Key Findings

Main findings will be summarized in the final portfolio version of this README.

Current key insights include:

* The online school has a large deal flow, but payment conversion is low.
* A high share of deals is lost before payment.
* Facebook Ads, Google Ads, Organic and SMM are key sources of business results.
* Digital Marketing is the strongest product by paid deals and revenue.
* Payment type analysis is limited by a high share of unknown values.
* Geography and German language level analysis is limited by missing data.
* Unit economics is positive at the aggregate level.
* C1, conversion from lead to customer, was selected as the main growth metric.
* The proposed HADI hypothesis can be tested within 14 days.

## Dashboard

An interactive dashboard was built with Plotly Dash.

The dashboard includes five main sections:

* Overview
* Marketing
* Sales
* Products
* Unit Economics

To run the dashboard locally:

```bash
pip install -r requirements.txt
python dashboard/app.py
```

## Reports

The repository includes final project materials in the `reports/` folder:

* full analytical report;
* project presentation.

## Project Status

This repository is currently being adapted from a final school project into an English-language portfolio project for the international market.

Next improvements:

* translate dashboard interface into English;
* clean and adapt notebooks for portfolio use;
* add dashboard screenshots;
* improve README with final visuals and concise business conclusions;
* prepare a short LinkedIn project summary.
