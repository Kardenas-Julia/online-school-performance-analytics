# Dashboard for the final Data Analytics project
# Online school performance analysis: sales, marketing, products and unit economics
#
# The data is cleaned in 02_data_cleaning.ipynb.
# The main analytical table is deals_clean.csv.
# A paid deal is defined by the business rule Stage = Payment Done.

from pathlib import Path

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, dash_table
import dash_bootstrap_components as dbc


# Data loading

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed"

deals = pd.read_csv(DATA_PATH / "deals_clean.csv")
contacts = pd.read_csv(DATA_PATH / "contacts_clean.csv")
calls = pd.read_csv(DATA_PATH / "calls_clean.csv")
spend = pd.read_csv(DATA_PATH / "spend_clean.csv")

unit_economics_total = pd.read_csv(DATA_PATH / "unit_economics_total.csv")
unit_economics_display = pd.read_csv(DATA_PATH / "unit_economics_display.csv")
products_status = pd.read_csv(DATA_PATH / "products_status.csv")
revenue_comparison = pd.read_csv(DATA_PATH / "revenue_comparison.csv")
unit_economics_check = pd.read_csv(DATA_PATH / "unit_economics_check.csv")
growth_scenarios_display = pd.read_csv(DATA_PATH / "growth_scenarios_display.csv")
best_growth_points = pd.read_csv(DATA_PATH / "best_growth_points.csv")
experiment_hypothesis = pd.read_csv(DATA_PATH / "experiment_hypothesis.csv")
daily_ua_summary = pd.read_csv(DATA_PATH / "daily_ua_summary.csv")
experiment_summary = pd.read_csv(DATA_PATH / "experiment_summary.csv")


# Compatibility rename for supporting files that may still contain Russian display columns

products_status = products_status.rename(columns={
    "Product": "Product",
    "Есть клиенты в расчётной базе": "Has Clients in Calculation Base",
    "Статус": "Status"
})

revenue_comparison = revenue_comparison.rename(columns={
    "Метрика": "Metric",
    "Metric": "Metric",
    "Значение": "Value"
})

unit_economics_check = unit_economics_check.rename(columns={
    "Metric": "Metric",
    "Метрика": "Metric",
    "Description": "Description",
    "Значение": "Value"
})

experiment_hypothesis = experiment_hypothesis.rename(columns={
    "Metric": "Metric",
    "Метрика": "Metric",
    "Значение": "Value"
})

experiment_summary = experiment_summary.rename(columns={
    "Базовая C1 (p), %": "Baseline C1 (p), %",
    "Целевая C1 (x), %": "Target C1 (x), %",
    "Sample Size per Group (n)": "Sample Size per Group (n)",
    "Общая выборка (2*n)": "Total Sample (2*n)",
    "Средний UA в день": "Average Daily UA",
    "Длительность эксперимента, дней": "Experiment Duration, Days",
    "Can be run within 14 days": "Can Run in 14 Days"
})


# Data type preparation

date_columns_deals = [
    "Created Time",
    "Create Date",
    "Closing Date",
    "Closing Date Only"
]

date_columns_contacts = [
    "Created Time",
    "Modified Time"
]

date_columns_calls = [
    "Call Start Time"
]

date_columns_spend = [
    "Date"
]

for col in date_columns_deals:
    if col in deals.columns:
        deals[col] = pd.to_datetime(deals[col], errors="coerce")

for col in date_columns_contacts:
    if col in contacts.columns:
        contacts[col] = pd.to_datetime(contacts[col], errors="coerce")

for col in date_columns_calls:
    if col in calls.columns:
        calls[col] = pd.to_datetime(calls[col], errors="coerce")

for col in date_columns_spend:
    if col in spend.columns:
        spend[col] = pd.to_datetime(spend[col], errors="coerce")

if "SLA" in deals.columns:
    deals["SLA"] = pd.to_timedelta(deals["SLA"], errors="coerce")

# General style settings and helper functions

COLORS = {
    "background": "#F7F8FA",
    "card": "#FFFFFF",
    "text": "#1F2937",
    "muted_text": "#6B7280",
    "primary": "#2F80ED",
    "secondary": "#56CCF2",
    "accent": "#F2994A",
    "success": "#27AE60",
    "danger": "#EB5757",
    "grid": "#E5E7EB"
}

HEATMAP_COLORSCALE = [
    [0.00, "#F7F8FA"],
    [0.25, "#DDEEFF"],
    [0.50, "#A7D8F5"],
    [0.75, "#56CCF2"],
    [1.00, "#2F80ED"]
]

PAYMENT_TYPE_COLORS = {
    "Unknown": "#A7D8F5",
    "Recurring Payments": "#2F80ED",
    "One Payment": "#F2994A",
    "Reservation": "#BDBDBD"
}

# Keep charts compact so each subtab fits better into a 16:9 screen
DEFAULT_CHART_HEIGHT = 360

def safe_divide(numerator, denominator):
    return np.where(
        (denominator != 0) & pd.notna(denominator),
        numerator / denominator,
        np.nan
    )


def format_int(value):
    if pd.isna(value):
        return "-"
    return f"{int(round(value)):,}".replace(",", " ")


def format_money(value):
    if pd.isna(value):
        return "-"
    return f"{float(value):,.2f}".replace(",", " ")


def format_percent(value):
    if pd.isna(value):
        return "-"
    return f"{float(value):.2f}%"


def format_percent_auto(value):
    if pd.isna(value):
        return "-"
    value = float(value)
    if abs(value) <= 1:
        value = value * 100
    return f"{value:.2f}%"


def clean_numeric_value(value):
    if pd.isna(value):
        return np.nan

    value_str = str(value)
    value_str = (
        value_str
        .replace("%", "")
        .replace("€", "")
        .replace("$", "")
        .replace("\xa0", "")
        .replace(" ", "")
        .replace(",", ".")
    )

    try:
        return float(value_str)
    except ValueError:
        return np.nan


def convert_possible_numeric_columns(df):
    """
    Converts object columns to numeric when most values can be parsed.
    This is needed for unit economics CSV files where numbers may be stored as text.
    """
    result = df.copy()

    text_columns = [
        "Product",
        "Scenario",
        "Status",
        "Metric",
        "Metric",
        "Description",
        "Can Run in 14 Days"
    ]

    for col in result.columns:
        if col in text_columns:
            continue

        if result[col].dtype == "object":
            converted = result[col].apply(clean_numeric_value)
            share_converted = converted.notna().mean()

            if share_converted >= 0.5:
                result[col] = converted

    return result


unit_economics_total = convert_possible_numeric_columns(unit_economics_total)
unit_economics_display = convert_possible_numeric_columns(unit_economics_display)
growth_scenarios_display = convert_possible_numeric_columns(growth_scenarios_display)
best_growth_points = convert_possible_numeric_columns(best_growth_points)
experiment_hypothesis = convert_possible_numeric_columns(experiment_hypothesis)
daily_ua_summary = convert_possible_numeric_columns(daily_ua_summary)
experiment_summary = convert_possible_numeric_columns(experiment_summary)


def find_col(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def create_empty_figure(title, message="No data to display"):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color=COLORS["muted_text"])
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        margin=dict(l=40, r=40, t=75, b=40),
        height=DEFAULT_CHART_HEIGHT
    )
    return fig


def apply_chart_layout(fig, title=None):
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=title,
            x=0.02,
            xanchor="left",
            font=dict(size=16, color=COLORS["text"])
        ),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(
            family="Arial",
            size=12,
            color=COLORS["text"]
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#D9E2F2",
            font=dict(
                color=COLORS["text"],
                size=12
            )
        ),
        margin=dict(l=50, r=40, t=75, b=45),
        height=DEFAULT_CHART_HEIGHT
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLORS["grid"]
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["grid"]
    )

    return fig


def apply_legend_style(fig, mode="top"):
    if mode == "top":
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.14,
                xanchor="center",
                x=0.5,
                title_text="",
                font=dict(size=10)
            ),
            margin=dict(l=50, r=40, t=105, b=45)
        )

    elif mode == "right":
        fig.update_layout(
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                title_text="",
                font=dict(size=10)
            ),
            margin=dict(l=50, r=170, t=80, b=50)
        )

    elif mode == "off":
        fig.update_layout(
            showlegend=False,
            margin=dict(l=50, r=40, t=75, b=45)
        )

    return fig


def create_kpi_card(title, value, subtitle=None):
    return dbc.Card(
        dbc.CardBody([
            html.Div(
                title,
                style={
                    "fontSize": "14px",
                    "color": COLORS["muted_text"],
                    "marginBottom": "6px"
                }
            ),
            html.Div(
                value,
                style={
                    "fontSize": "26px",
                    "fontWeight": "700",
                    "color": COLORS["text"]
                }
            ),
            html.Div(
                subtitle if subtitle else "",
                style={
                    "fontSize": "12px",
                    "color": COLORS["muted_text"],
                    "marginTop": "6px"
                }
            )
        ]),
        style={
            "backgroundColor": COLORS["card"],
            "border": "1px solid #E5E7EB",
            "borderRadius": "14px",
            "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.04)",
            "height": "100%"
        }
    )


def create_chart_card(figure, graph_height=DEFAULT_CHART_HEIGHT):
    return dbc.Card(
        dbc.CardBody([
            dcc.Graph(
                figure=figure,
                config={"displayModeBar": True},
                style={"height": f"{graph_height}px"}
            )
        ]),
        style={
            "borderRadius": "14px",
            "border": "1px solid #E5E7EB",
            "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.04)",
            "backgroundColor": COLORS["card"]
        }
    )


def create_table_card(df, page_size=10):
    return dbc.Card(
        dbc.CardBody([
            dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[
                    {"name": col, "id": col}
                    for col in df.columns
                ],
                page_size=page_size,
                sort_action="native",
                filter_action="native",
                style_table={
                    "overflowX": "auto"
                },
                style_header={
                    "backgroundColor": "#F3F4F6",
                    "fontWeight": "700",
                    "color": COLORS["text"],
                    "border": "1px solid #E5E7EB"
                },
                style_cell={
                    "fontFamily": "Arial",
                    "fontSize": "12px",
                    "padding": "8px",
                    "textAlign": "left",
                    "border": "1px solid #E5E7EB",
                    "maxWidth": "220px",
                    "whiteSpace": "normal"
                },
                style_data={
                    "backgroundColor": COLORS["card"],
                    "color": COLORS["text"]
                }
            )
        ]),
        style={
            "borderRadius": "14px",
            "border": "1px solid #E5E7EB",
            "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.04)",
            "backgroundColor": COLORS["card"]
        }
    )


def create_section_header(title, description):
    return html.Div(
        [
            html.H3(
                title,
                style={
                    "color": COLORS["text"],
                    "fontWeight": "700",
                    "marginBottom": "8px"
                }
            ),
            html.P(
                description,
                style={
                    "color": COLORS["muted_text"],
                    "marginBottom": "24px"
                }
            )
        ]
    )


# Core business rules for calculations

# Payment is defined only by Stage = Payment Done, which is the main business rule of the project
deals["Is Paid"] = deals["Stage"] == "Payment Done"

deals["Revenue Paid"] = np.where(
    deals["Is Paid"],
    deals["Initial Amount Paid"].fillna(0),
    0
)

paid_deals = deals[deals["Is Paid"]].copy()


# Overview tab

total_deals = deals.shape[0]
paid_deals_count = paid_deals.shape[0]
payment_conversion = paid_deals_count / total_deals * 100
revenue = paid_deals["Initial Amount Paid"].sum()
avg_revenue = paid_deals["Initial Amount Paid"].mean()
lost_deals_count = deals[deals["Stage"] == "Lost"].shape[0]
revenue_per_deal = revenue / total_deals if total_deals > 0 else np.nan


overview_kpi_cards = dbc.Row(
    [
        dbc.Col(create_kpi_card("Total Deals", format_int(total_deals), "All records in Deals"), md=2),
        dbc.Col(create_kpi_card("Paid Deals", format_int(paid_deals_count), "Stage = Payment Done"), md=2),
        dbc.Col(create_kpi_card("Payment Conversion", format_percent(payment_conversion), "Paid Deals / Total Deals"), md=2),
        dbc.Col(create_kpi_card("Actual Revenue", format_money(revenue), "Initial Amount Paid"), md=2),
        dbc.Col(create_kpi_card("Average Check", format_money(avg_revenue), "Average across paid deals"), md=2),
        dbc.Col(create_kpi_card("Revenue per Deal", format_money(revenue_per_deal), "Revenue / Total Deals"), md=2)
    ],
    className="g-3",
    style={"marginBottom": "24px"}
)


stage_summary = (
    deals
    .groupby("Stage", dropna=False)
    .agg(Deals_Count=("Id", "count"))
    .reset_index()
)

stage_summary["Share, %"] = (
    stage_summary["Deals_Count"] / stage_summary["Deals_Count"].sum() * 100
).round(2)

stage_summary = stage_summary.sort_values("Deals_Count", ascending=False)

stage_fig = px.bar(
    stage_summary,
    x="Deals_Count",
    y="Stage",
    orientation="h",
    text="Share, %",
    labels={
        "Deals_Count": "Deals Count",
        "Stage": "Stage",
        "Share, %": "Share, %"
    },
    color_discrete_sequence=[COLORS["primary"]]
)

stage_fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside",
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Deals Count: %{x}<br>"
        "Share: %{text:.2f}%<extra></extra>"
    )
)

stage_fig.update_layout(yaxis={"categoryorder": "total ascending"})
stage_fig = apply_chart_layout(stage_fig, "Sales Funnel by Stage")


deals["Created Month"] = deals["Created Time"].dt.to_period("M").dt.to_timestamp()

monthly_deals = (
    deals
    .groupby("Created Month")
    .agg(
        Deals_Count=("Id", "count"),
        Paid_Deals=("Is Paid", "sum")
    )
    .reset_index()
)

monthly_deals["Payment Conversion, %"] = (
    monthly_deals["Paid_Deals"] / monthly_deals["Deals_Count"] * 100
).round(2)

monthly_deals_fig = go.Figure()

monthly_deals_fig.add_trace(
    go.Bar(
        x=monthly_deals["Created Month"],
        y=monthly_deals["Deals_Count"],
        name="Created Deals",
        marker_color=COLORS["primary"],
        opacity=0.85,
        hovertemplate="Month: %{x|%Y-%m}<br>Created Deals: %{y}<extra></extra>"
    )
)

monthly_deals_fig.add_trace(
    go.Scatter(
        x=monthly_deals["Created Month"],
        y=monthly_deals["Paid_Deals"],
        name="Paid Deals",
        mode="lines+markers",
        line=dict(color=COLORS["accent"], width=3),
        marker=dict(size=8),
        yaxis="y2",
        hovertemplate="Month: %{x|%Y-%m}<br>Paid Deals: %{y}<extra></extra>"
    )
)

monthly_deals_fig.update_layout(
    xaxis=dict(title="Month"),
    yaxis=dict(title="Created Deals"),
    yaxis2=dict(
        title="Paid Deals",
        overlaying="y",
        side="right"
    )
)

monthly_deals_fig = apply_chart_layout(
    monthly_deals_fig,
    "Monthly Dynamics of Created and Paid Deals"
)

monthly_conversion_fig = px.line(
    monthly_deals,
    x="Created Month",
    y="Payment Conversion, %",
    markers=True,
    labels={
        "Created Month": "Month",
        "Payment Conversion, %": "Payment Conversion, %"
    }
)

monthly_conversion_fig.update_traces(
    line=dict(color=COLORS["accent"], width=3),
    marker=dict(size=8),
    hovertemplate="Month: %{x|%Y-%m}<br>Payment Conversion: %{y:.2f}%<extra></extra>"
)

monthly_conversion_fig = apply_chart_layout(
    monthly_conversion_fig,
    "Monthly Payment Conversion Dynamics"
)


top_sources_revenue = (
    deals
    .groupby("Source", dropna=False)
    .agg(Revenue=("Revenue Paid", "sum"))
    .reset_index()
    .fillna("Unknown")
    .sort_values("Revenue", ascending=False)
    .head(10)
)

top_sources_revenue_fig = px.bar(
    top_sources_revenue,
    x="Revenue",
    y="Source",
    orientation="h",
    labels={"Revenue": "Revenue", "Source": "Source"},
    color_discrete_sequence=[COLORS["primary"]]
)

top_sources_revenue_fig.update_layout(yaxis={"categoryorder": "total ascending"})
top_sources_revenue_fig = apply_chart_layout(top_sources_revenue_fig, "Top Sources by Revenue")


top_products_revenue = (
    paid_deals
    .groupby("Product", dropna=False)
    .agg(
        Paid_Deals=("Id", "count"),
        Revenue=("Initial Amount Paid", "sum")
    )
    .reset_index()
    .fillna("Unknown")
    .sort_values("Revenue", ascending=False)
    .head(10)
)

top_products_revenue_fig = px.bar(
    top_products_revenue,
    x="Revenue",
    y="Product",
    orientation="h",
    labels={"Revenue": "Revenue", "Product": "Product"},
    color_discrete_sequence=[COLORS["accent"]]
)

top_products_revenue_fig.update_layout(yaxis={"categoryorder": "total ascending"})
top_products_revenue_fig = apply_chart_layout(top_products_revenue_fig, "Top Products by Revenue")


# Marketing tab

source_summary = (
    deals
    .groupby("Source", dropna=False)
    .agg(
        Deals_Count=("Id", "count"),
        Paid_Deals=("Is Paid", "sum"),
        Revenue=("Revenue Paid", "sum")
    )
    .reset_index()
)

source_summary["Source"] = source_summary["Source"].fillna("Unknown")

source_summary["Payment Conversion, %"] = (
    source_summary["Paid_Deals"] / source_summary["Deals_Count"] * 100
).round(2)

spend_agg_dict = {
    "Spend": ("Spend", "sum")
}

if "Impressions" in spend.columns:
    spend_agg_dict["Impressions"] = ("Impressions", "sum")

if "Clicks" in spend.columns:
    spend_agg_dict["Clicks"] = ("Clicks", "sum")

spend_source_summary = (
    spend
    .groupby("Source", dropna=False)
    .agg(**spend_agg_dict)
    .reset_index()
)

spend_source_summary["Source"] = spend_source_summary["Source"].fillna("Unknown")

source_marketing_summary = source_summary.merge(
    spend_source_summary,
    on="Source",
    how="left"
)

source_marketing_summary["Spend"] = source_marketing_summary["Spend"].fillna(0)

source_marketing_summary["CPL"] = safe_divide(
    source_marketing_summary["Spend"],
    source_marketing_summary["Deals_Count"]
)

source_marketing_summary["CAC"] = safe_divide(
    source_marketing_summary["Spend"],
    source_marketing_summary["Paid_Deals"]
)

source_marketing_summary["Revenue / Spend"] = safe_divide(
    source_marketing_summary["Revenue"],
    source_marketing_summary["Spend"]
)

total_spend = source_marketing_summary["Spend"].sum()
marketing_revenue = source_marketing_summary["Revenue"].sum()
marketing_paid_deals = source_marketing_summary["Paid_Deals"].sum()
marketing_deals = source_marketing_summary["Deals_Count"].sum()
marketing_conversion = marketing_paid_deals / marketing_deals * 100
marketing_cpl = total_spend / marketing_deals
marketing_cac = total_spend / marketing_paid_deals if marketing_paid_deals > 0 else np.nan

marketing_kpi_cards = dbc.Row(
    [
        dbc.Col(create_kpi_card("Spend", format_money(total_spend), "Total Marketing Spend"), md=2),
        dbc.Col(create_kpi_card("Revenue", format_money(marketing_revenue), "Revenue by Source"), md=2),
        dbc.Col(create_kpi_card("Paid Deals", format_int(marketing_paid_deals), "Paid Deals Count"), md=2),
        dbc.Col(create_kpi_card("Conversion", format_percent(marketing_conversion), "Paid Deals / Deals"), md=2),
        dbc.Col(create_kpi_card("CPL", format_money(marketing_cpl), "Cost per Lead"), md=2),
        dbc.Col(create_kpi_card("CAC", format_money(marketing_cac), "Customer Acquisition Cost"), md=2)
    ],
    className="g-3",
    style={"marginBottom": "24px"}
)

source_bubble_plot = (
    source_marketing_summary
    .sort_values("Deals_Count", ascending=False)
    .head(12)
    .copy()
)

source_bubble_fig = px.scatter(
    source_bubble_plot,
    x="Deals_Count",
    y="Payment Conversion, %",
    size="Revenue",
    color="Source",
    text="Source",
    hover_name="Source",
    custom_data=["Paid_Deals", "Revenue", "CPL", "CAC"],
    size_max=55,
    labels={
        "Deals_Count": "Deals Count",
        "Payment Conversion, %": "Payment Conversion, %",
        "Revenue": "Revenue",
        "Source": "Source"
    }
)

source_bubble_fig.update_traces(
    marker=dict(opacity=0.75, line=dict(width=0.5, color="#1F2937")),
    textposition="top center",
    textfont=dict(size=10),
    cliponaxis=False,
    hovertemplate=(
        "<b>%{hovertext}</b><br>"
        "Deals Count: %{x}<br>"
        "Paid Deals: %{customdata[0]}<br>"
        "Conversion: %{y:.2f}%<br>"
        "Revenue: %{customdata[1]:,.2f}<br>"
        "CPL: %{customdata[2]:,.2f}<br>"
        "CAC: %{customdata[3]:,.2f}<extra></extra>"
    )
)

source_bubble_fig = apply_chart_layout(source_bubble_fig, "Sources: Deal Volume, Conversion and Revenue")
source_bubble_fig.update_layout(showlegend=False, margin=dict(l=50, r=40, t=90, b=50))
spend_revenue_plot = source_marketing_summary[source_marketing_summary["Spend"] > 0].copy()

spend_revenue_fig = px.scatter(
    spend_revenue_plot,
    x="Spend",
    y="Revenue",
    size="Deals_Count",
    color="Source",
    text="Source",
    hover_name="Source",
    custom_data=["Deals_Count", "Paid_Deals", "Spend", "Revenue", "CPL", "CAC", "Revenue / Spend"],
    size_max=50,
    labels={
        "Spend": "Spend",
        "Revenue": "Revenue",
        "Deals_Count": "Deals Count",
        "Source": "Source"
    }
)

spend_revenue_fig.update_traces(
    marker=dict(opacity=0.75, line=dict(width=0.5, color="#1F2937")),
    textposition="top center",
    textfont=dict(size=10),
    cliponaxis=False,
    hovertemplate=(
        "<b>%{hovertext}</b><br>"
        "Spend: %{customdata[2]:,.2f}<br>"
        "Revenue: %{customdata[3]:,.2f}<br>"
        "Deals: %{customdata[0]}<br>"
        "Paid Deals: %{customdata[1]}<br>"
        "CPL: %{customdata[4]:,.2f}<br>"
        "CAC: %{customdata[5]:,.2f}<br>"
        "Revenue / Spend: %{customdata[6]:.2f}<extra></extra>"
    )
)

if not spend_revenue_plot.empty:
    max_value = max(spend_revenue_plot["Spend"].max(), spend_revenue_plot["Revenue"].max())

    spend_revenue_fig.add_trace(
        go.Scatter(
            x=[0, max_value],
            y=[0, max_value],
            mode="lines",
            name="Revenue = Spend",
            line=dict(color=COLORS["muted_text"], dash="dash"),
            showlegend=False,
            hovertemplate="Revenue = Spend<extra></extra>"
        )
    )

    spend_revenue_fig.add_annotation(
        x=max_value * 0.72,
        y=max_value * 0.72,
        text="Revenue = Spend",
        showarrow=False,
        font=dict(size=11, color=COLORS["muted_text"]),
        bgcolor="rgba(255,255,255,0.75)"
    )

spend_revenue_fig = apply_chart_layout(spend_revenue_fig, "Source Performance: Spend, Revenue and Deals")
spend_revenue_fig.update_layout(showlegend=False, margin=dict(l=50, r=40, t=90, b=50))


if "Quality" in deals.columns:
    quality_data = deals.copy()
    quality_data["Is Quality Lead"] = ~quality_data["Quality"].astype(str).str.contains(
        "Non Qualified|Non Target|Unknown",
        case=False,
        na=False
    )

    quality_by_source = (
        quality_data
        .groupby("Source", dropna=False)
        .agg(
            Deals_Count=("Id", "count"),
            Quality_Leads=("Is Quality Lead", "sum")
        )
        .reset_index()
    )

    quality_by_source["Source"] = quality_by_source["Source"].fillna("Unknown")
    quality_by_source["Quality Lead Share, %"] = (
        quality_by_source["Quality_Leads"] / quality_by_source["Deals_Count"] * 100
    ).round(2)

    quality_by_source = quality_by_source.sort_values(
        "Quality Lead Share, %",
        ascending=False
    ).head(12)

    quality_fig = px.bar(
        quality_by_source,
        x="Quality Lead Share, %",
        y="Source",
        orientation="h",
        labels={
            "Quality Lead Share, %": "Quality Lead Share, %",
            "Source": "Source"
        },
        color_discrete_sequence=[COLORS["success"]]
    )

    quality_fig.update_layout(yaxis={"categoryorder": "total ascending"})
    quality_fig = apply_chart_layout(quality_fig, "Lead Quality by Source")

else:
    quality_fig = create_empty_figure("Lead Quality by Source", "Quality field not found")


if "Campaign" in deals.columns:
    campaign_summary = (
        deals
        .groupby("Campaign", dropna=False)
        .agg(
            Deals_Count=("Id", "count"),
            Paid_Deals=("Is Paid", "sum"),
            Revenue=("Revenue Paid", "sum")
        )
        .reset_index()
    )

    campaign_summary["Campaign"] = campaign_summary["Campaign"].fillna("Unknown")
    campaign_summary["Payment Conversion, %"] = (
        campaign_summary["Paid_Deals"] / campaign_summary["Deals_Count"] * 100
    ).round(2)

    campaign_summary = campaign_summary.sort_values(
        ["Paid_Deals", "Revenue"],
        ascending=False
    ).head(30)

else:
    campaign_summary = pd.DataFrame({
        "Campaign": [],
        "Deals_Count": [],
        "Paid_Deals": [],
        "Payment Conversion, %": [],
        "Revenue": []
    })

# Campaigns are added as an additional detail; Source remains the main marketing analysis level

if not campaign_summary.empty:
    campaign_plot = campaign_summary.copy()

    campaign_plot = campaign_plot[
        campaign_plot["Campaign"].astype(str) != "Unknown"
    ].copy()

    campaign_plot = campaign_plot.sort_values(
        "Revenue",
        ascending=False
    ).head(15)

    campaign_fig = go.Figure()

    campaign_fig.add_trace(
        go.Bar(
            x=campaign_plot["Campaign"],
            y=campaign_plot["Revenue"],
            name="Revenue",
            marker_color=COLORS["primary"],
            opacity=0.85,
            customdata=np.stack(
                [
                    campaign_plot["Deals_Count"],
                    campaign_plot["Paid_Deals"],
                    campaign_plot["Payment Conversion, %"]
                ],
                axis=-1
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Revenue: %{y:,.2f}<br>"
                "Deals: %{customdata[0]}<br>"
                "Paid Deals: %{customdata[1]}<br>"
                "Conversion: %{customdata[2]:.2f}%<extra></extra>"
            )
        )
    )

    campaign_fig.add_trace(
        go.Scatter(
            x=campaign_plot["Campaign"],
            y=campaign_plot["Payment Conversion, %"],
            name="Payment Conversion, %",
            mode="lines+markers",
            line=dict(color=COLORS["accent"], width=3),
            marker=dict(size=8),
            yaxis="y2",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Conversion: %{y:.2f}%<extra></extra>"
            )
        )
    )

    campaign_fig.update_layout(
        xaxis=dict(
            title="Campaign",
            tickangle=-35
        ),
        yaxis=dict(title="Revenue"),
        yaxis2=dict(
            title="Payment Conversion, %",
            overlaying="y",
            side="right"
        )
    )

    campaign_fig = apply_chart_layout(
        campaign_fig,
        "Top Campaigns by Revenue and Conversion"
    )

    campaign_fig = apply_legend_style(campaign_fig, "top")

else:
    campaign_fig = create_empty_figure(
        "Top Campaigns by Revenue and Conversion",
        "No campaign data available"
    )

campaign_table = campaign_summary.copy()

campaign_table = campaign_table.rename(columns={
    "Campaign": "Campaign",
    "Deals_Count": "Deals",
    "Paid_Deals": "Paid Deals",
    "Payment Conversion, %": "Conversion, %",
    "Revenue": "Revenue"
})

campaign_table = campaign_table[
    ["Campaign", "Deals", "Paid Deals", "Conversion, %", "Revenue"]
]

campaign_table["Revenue"] = campaign_table["Revenue"].round(2)
campaign_table["Conversion, %"] = campaign_table["Conversion, %"].round(2)

# Sales tab

manager_performance_summary = (
    deals
    .groupby("Deal Owner Name", dropna=False)
    .agg(
        Deals_Count=("Id", "count"),
        Paid_Deals=("Is Paid", "sum"),
        Revenue=("Revenue Paid", "sum")
    )
    .reset_index()
)

manager_performance_summary["Deal Owner Name"] = (
    manager_performance_summary["Deal Owner Name"].fillna("Unknown Manager")
)

manager_performance_summary["Payment Conversion, %"] = (
    manager_performance_summary["Paid_Deals"] / manager_performance_summary["Deals_Count"] * 100
).round(2)

manager_performance_summary["Paid Deals Share, %"] = (
    manager_performance_summary["Paid_Deals"] / manager_performance_summary["Paid_Deals"].sum() * 100
).round(2)

manager_performance_summary["Revenue Share, %"] = (
    manager_performance_summary["Revenue"] / manager_performance_summary["Revenue"].sum() * 100
).round(2)

# Short sales team summary

sales_managers_count = manager_performance_summary[
    manager_performance_summary["Deal Owner Name"] != "Unknown Manager"
]["Deal Owner Name"].nunique()

sales_total_deals = manager_performance_summary["Deals_Count"].sum()
sales_paid_deals = manager_performance_summary["Paid_Deals"].sum()
sales_revenue = manager_performance_summary["Revenue"].sum()

sales_avg_manager_conversion = manager_performance_summary[
    manager_performance_summary["Deals_Count"] > 0
]["Payment Conversion, %"].mean()

best_manager_row = manager_performance_summary.sort_values(
    "Revenue",
    ascending=False
).head(1)

if not best_manager_row.empty:
    best_manager_name = best_manager_row["Deal Owner Name"].iloc[0]
    best_manager_revenue = best_manager_row["Revenue"].iloc[0]
else:
    best_manager_name = "-"
    best_manager_revenue = np.nan


sales_kpi_cards = dbc.Row(
    [
        dbc.Col(
            create_kpi_card(
                "Managers",
                format_int(sales_managers_count),
                "Managers in Deals"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "Total Deals",
                format_int(sales_total_deals),
                "Deals by Manager"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "Paid Deals",
                format_int(sales_paid_deals),
                "Stage = Payment Done"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "Revenue",
                format_money(sales_revenue),
                "Revenue by Manager"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "Average Conversion",
                format_percent(sales_avg_manager_conversion),
                "Average across managers"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "Revenue Leader",
                best_manager_name,
                f"Revenue: {format_money(best_manager_revenue)}"
            ),
            md=2
        )
    ],
    className="g-3",
    style={"marginBottom": "24px"}
)

manager_bubble_plot = manager_performance_summary.sort_values("Deals_Count", ascending=False).head(20)

manager_bubble_fig = px.scatter(
    manager_bubble_plot,
    x="Deals_Count",
    y="Payment Conversion, %",
    size="Revenue",
    color="Deal Owner Name",
    text="Deal Owner Name",
    hover_name="Deal Owner Name",
    custom_data=["Paid_Deals", "Revenue"],
    size_max=55,
    labels={
        "Deals_Count": "Deals Count",
        "Payment Conversion, %": "Payment Conversion, %",
        "Revenue": "Revenue",
        "Deal Owner Name": "Manager"
    }
)

manager_bubble_fig.update_traces(
    marker=dict(opacity=0.75, line=dict(width=0.5, color="#1F2937")),
    textposition="top center",
    textfont=dict(size=10),
    cliponaxis=False,
    hovertemplate=(
        "<b>%{hovertext}</b><br>"
        "Deals: %{x}<br>"
        "Paid Deals: %{customdata[0]}<br>"
        "Conversion: %{y:.2f}%<br>"
        "Revenue: %{customdata[1]:,.2f}<extra></extra>"
    )
)

manager_bubble_fig = apply_chart_layout(manager_bubble_fig, "Managers: Workload, Conversion and Revenue")
manager_bubble_fig = apply_legend_style(manager_bubble_fig, "off")


manager_share_plot = manager_performance_summary.sort_values("Revenue Share, %", ascending=False).head(12)

manager_share_fig = go.Figure()

manager_share_fig.add_trace(
    go.Bar(
        x=manager_share_plot["Deal Owner Name"],
        y=manager_share_plot["Paid Deals Share, %"],
        name="Paid Deals Share, %",
        marker_color=COLORS["primary"]
    )
)

manager_share_fig.add_trace(
    go.Bar(
        x=manager_share_plot["Deal Owner Name"],
        y=manager_share_plot["Revenue Share, %"],
        name="Revenue Share, %",
        marker_color=COLORS["accent"]
    )
)

manager_share_fig.update_layout(
    barmode="group",
    xaxis=dict(title="Manager"),
    yaxis=dict(title="Share, %")
)

manager_share_fig = apply_chart_layout(manager_share_fig, "Manager Contribution to Paid Deals and Revenue")
manager_share_fig = apply_legend_style(manager_share_fig, "top")


if "SLA" in deals.columns:
    deals["SLA Minutes"] = deals["SLA"].dt.total_seconds() / 60

    deals["SLA Group"] = pd.cut(
        deals["SLA Minutes"],
        bins=[-np.inf, 5, 15, 60, 1440, np.inf],
        labels=["0–5 min", "5–15 min", "15–60 min", "1–24 hours", "24+ hours"]
    )

    sla_conversion_summary = (
        deals
        .groupby("SLA Group", observed=False)
        .agg(
            Deals_Count=("Id", "count"),
            Paid_Deals=("Is Paid", "sum")
        )
        .reset_index()
    )

    sla_conversion_summary["Payment Conversion, %"] = (
        sla_conversion_summary["Paid_Deals"] / sla_conversion_summary["Deals_Count"] * 100
    ).round(2)

else:
    sla_conversion_summary = pd.DataFrame({
        "SLA Group": [],
        "Deals_Count": [],
        "Paid_Deals": [],
        "Payment Conversion, %": []
    })

sla_fig = go.Figure()

sla_fig.add_trace(
    go.Bar(
        x=sla_conversion_summary["SLA Group"],
        y=sla_conversion_summary["Deals_Count"],
        name="Deals Count",
        marker_color=COLORS["primary"],
        opacity=0.85
    )
)

sla_fig.add_trace(
    go.Scatter(
        x=sla_conversion_summary["SLA Group"],
        y=sla_conversion_summary["Payment Conversion, %"],
        name="Payment Conversion, %",
        mode="lines+markers",
        line=dict(color=COLORS["accent"], width=3),
        marker=dict(size=8),
        yaxis="y2"
    )
)

sla_fig.update_layout(
    xaxis=dict(title="SLA Group"),
    yaxis=dict(title="Deals Count"),
    yaxis2=dict(
        title="Payment Conversion, %",
        overlaying="y",
        side="right"
    )
)

sla_fig = apply_chart_layout(sla_fig, "SLA: Deal Volume and Conversion")
sla_fig = apply_legend_style(sla_fig, "top")


if "Call Owner Name" in calls.columns:
    calls_manager_summary = (
        calls
        .groupby("Call Owner Name", dropna=False)
        .agg(Calls_Count=("Call Owner Name", "count"))
        .reset_index()
        .rename(columns={"Call Owner Name": "Deal Owner Name"})
    )

    calls_manager_summary["Deal Owner Name"] = (
        calls_manager_summary["Deal Owner Name"].fillna("Unknown Manager")
    )

    manager_sales_calls_summary = manager_performance_summary.merge(
        calls_manager_summary,
        on="Deal Owner Name",
        how="left"
    )

    manager_sales_calls_summary["Calls_Count"] = (
        manager_sales_calls_summary["Calls_Count"].fillna(0)
    )

else:
    manager_sales_calls_summary = manager_performance_summary.copy()
    manager_sales_calls_summary["Calls_Count"] = 0

manager_calls_plot = manager_sales_calls_summary.sort_values("Deals_Count", ascending=False).head(20)

manager_calls_fig = px.scatter(
    manager_calls_plot,
    x="Calls_Count",
    y="Paid_Deals",
    size="Deals_Count",
    color="Deal Owner Name",
    text="Deal Owner Name",
    hover_name="Deal Owner Name",
    custom_data=["Deals_Count", "Calls_Count", "Paid_Deals", "Revenue"],
    size_max=55,
    labels={
        "Calls_Count": "Calls Count",
        "Paid_Deals": "Paid Deals Count",
        "Deals_Count": "Deals Count",
        "Deal Owner Name": "Manager"
    }
)

manager_calls_fig.update_traces(
    marker=dict(opacity=0.75, line=dict(width=0.5, color="#1F2937")),
    textposition="top center",
    textfont=dict(size=10),
    cliponaxis=False,
    hovertemplate=(
        "<b>%{hovertext}</b><br>"
        "Calls: %{customdata[1]}<br>"
        "Deals: %{customdata[0]}<br>"
        "Paid Deals: %{customdata[2]}<br>"
        "Revenue: %{customdata[3]:,.2f}<extra></extra>"
    )
)

manager_calls_fig = apply_chart_layout(manager_calls_fig, "Managers: Calls, Deals and Paid Deals")
manager_calls_fig = apply_legend_style(manager_calls_fig, "off")

# Analyze not only paid deal volume but also the financial contribution of each manager

manager_paid_revenue_plot = manager_performance_summary.copy()

manager_paid_revenue_plot = manager_paid_revenue_plot[
    (manager_paid_revenue_plot["Deal Owner Name"] != "Unknown Manager") &
    (manager_paid_revenue_plot["Paid_Deals"] > 0)
].copy()

manager_paid_revenue_plot["Average Check"] = safe_divide(
    manager_paid_revenue_plot["Revenue"],
    manager_paid_revenue_plot["Paid_Deals"]
)

manager_paid_revenue_plot = manager_paid_revenue_plot.sort_values(
    "Revenue",
    ascending=False
).head(20)

if not manager_paid_revenue_plot.empty:
    manager_paid_revenue_fig = px.scatter(
        manager_paid_revenue_plot,
        x="Paid_Deals",
        y="Revenue",
        size="Average Check",
        color="Deal Owner Name",
        text="Deal Owner Name",
        hover_name="Deal Owner Name",
        custom_data=["Paid_Deals", "Revenue", "Average Check", "Payment Conversion, %"],
        size_max=45,
        labels={
            "Paid_Deals": "Paid Deals",
            "Revenue": "Revenue",
            "Average Check": "Average Check",
            "Deal Owner Name": "Manager"
        }
    )

    manager_paid_revenue_fig.update_traces(
        marker=dict(opacity=0.75, line=dict(width=0.5, color="#1F2937")),
        textposition="top center",
        textfont=dict(size=10),
        cliponaxis=False,
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Paid Deals: %{customdata[0]}<br>"
            "Revenue: %{customdata[1]:,.2f}<br>"
            "Average Check: %{customdata[2]:,.2f}<br>"
            "Conversion: %{customdata[3]:.2f}%<extra></extra>"
        )
    )

    manager_paid_revenue_fig = apply_chart_layout(
        manager_paid_revenue_fig,
        "Managers: Paid Deals, Revenue and Average Check"
    )

    manager_paid_revenue_fig.update_layout(
        showlegend=False,
        margin=dict(l=50, r=40, t=90, b=50)
    )

else:
    manager_paid_revenue_fig = create_empty_figure(
        "Managers: Paid Deals, Revenue and Average Check",
        "No paid deal data available"
    )

# Manager-source link: the structure of incoming traffic should be taken into account

if "Source" in deals.columns and "Deal Owner Name" in deals.columns:
    manager_source_data = deals.copy()

    manager_source_data["Deal Owner Name"] = manager_source_data["Deal Owner Name"].fillna("Unknown Manager")
    manager_source_data["Source"] = manager_source_data["Source"].fillna("Unknown")

    top_managers_for_source = (
        manager_source_data
        .groupby("Deal Owner Name")
        .agg(Paid_Deals=("Is Paid", "sum"))
        .reset_index()
        .sort_values("Paid_Deals", ascending=False)
        .head(10)["Deal Owner Name"]
    )

    top_sources_for_manager = (
        manager_source_data
        .groupby("Source")
        .agg(Paid_Deals=("Is Paid", "sum"))
        .reset_index()
        .sort_values("Paid_Deals", ascending=False)
        .head(10)["Source"]
    )

    manager_source_heatmap_data = manager_source_data[
        manager_source_data["Deal Owner Name"].isin(top_managers_for_source) &
        manager_source_data["Source"].isin(top_sources_for_manager)
    ].copy()

    manager_source_pivot = manager_source_heatmap_data.pivot_table(
        index="Deal Owner Name",
        columns="Source",
        values="Is Paid",
        aggfunc="sum",
        fill_value=0
    )

    if not manager_source_pivot.empty:
        manager_source_fig = px.imshow(
            manager_source_pivot,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=HEATMAP_COLORSCALE,
            labels={
                "x": "Source",
                "y": "Manager",
                "color": "Paid Deals"
            }
        )

        manager_source_fig = apply_chart_layout(
            manager_source_fig,
            "Managers × Sources: Paid Deals"
        )

        manager_source_fig.update_layout(
            xaxis=dict(tickangle=-35),
            margin=dict(l=80, r=40, t=90, b=90),
            coloraxis_colorbar=dict(
                title="Paid Deals",
                thickness=14,
                len=0.75
            )
        )

        manager_source_fig.update_traces(
            textfont=dict(size=11, color=COLORS["text"]),
            hovertemplate=(
                "Manager: %{y}<br>"
                "Source: %{x}<br>"
                "Paid Deals: %{z}<extra></extra>"
            )
        )

    else:
        manager_source_fig = create_empty_figure(
            "Managers × Sources: Paid Deals",
            "No data available for manager-source analysis"
        )

else:
    manager_source_fig = create_empty_figure(
        "Managers × Sources: Paid Deals",
        "Source or Deal Owner Name fields not found"
    )


# Manager-campaign link as an additional sales analysis dimension

if "Campaign" in deals.columns and "Deal Owner Name" in deals.columns:
    manager_campaign_data = deals.copy()

    manager_campaign_data["Deal Owner Name"] = manager_campaign_data["Deal Owner Name"].fillna("Unknown Manager")
    manager_campaign_data["Campaign"] = manager_campaign_data["Campaign"].fillna("Unknown")

    manager_campaign_data = manager_campaign_data[
        manager_campaign_data["Campaign"].astype(str) != "Unknown"
    ].copy()

    top_managers_for_campaign = (
        manager_campaign_data
        .groupby("Deal Owner Name")
        .agg(Revenue=("Revenue Paid", "sum"))
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .head(10)["Deal Owner Name"]
    )

    top_campaigns_for_manager = (
        manager_campaign_data
        .groupby("Campaign")
        .agg(Revenue=("Revenue Paid", "sum"))
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .head(10)["Campaign"]
    )

    manager_campaign_heatmap_data = manager_campaign_data[
        manager_campaign_data["Deal Owner Name"].isin(top_managers_for_campaign) &
        manager_campaign_data["Campaign"].isin(top_campaigns_for_manager)
    ].copy()

    manager_campaign_pivot = manager_campaign_heatmap_data.pivot_table(
        index="Deal Owner Name",
        columns="Campaign",
        values="Revenue Paid",
        aggfunc="sum",
        fill_value=0
    )

    if not manager_campaign_pivot.empty:
        manager_campaign_fig = px.imshow(
            manager_campaign_pivot,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale=HEATMAP_COLORSCALE,
            labels={
                "x": "Campaign",
                "y": "Manager",
                "color": "Revenue"
            }
        )

        manager_campaign_fig = apply_chart_layout(
            manager_campaign_fig,
            "Managers × Campaigns: Revenue"
        )

        manager_campaign_fig.update_layout(
            xaxis=dict(tickangle=-35),
            margin=dict(l=80, r=40, t=90, b=110),
            coloraxis_colorbar=dict(
                title="Revenue",
                thickness=14,
                len=0.75
            )
        )

        manager_campaign_fig.update_traces(
            textfont=dict(size=10, color=COLORS["text"]),
            hovertemplate=(
                "Manager: %{y}<br>"
                "Campaign: %{x}<br>"
                "Revenue: %{z:,.2f}<extra></extra>"
            )
        )

    else:
        manager_campaign_fig = create_empty_figure(
            "Managers × Campaigns: Revenue",
            "No data available for manager-campaign analysis"
        )

else:
    manager_campaign_fig = create_empty_figure(
        "Managers × Campaigns: Revenue",
        "Campaign or Deal Owner Name fields not found"
    )

# Products tab

paid_product_summary = (
    paid_deals
    .groupby("Product", dropna=False)
    .agg(
        Paid_Deals=("Id", "count"),
        Revenue=("Initial Amount Paid", "sum"),
        Avg_Revenue=("Initial Amount Paid", "mean"),
        Median_Revenue=("Initial Amount Paid", "median")
    )
    .reset_index()
)

paid_product_summary["Product"] = paid_product_summary["Product"].fillna("Unknown")

paid_product_summary["Paid Deals Share, %"] = (
    paid_product_summary["Paid_Deals"] / paid_product_summary["Paid_Deals"].sum() * 100
).round(2)

paid_product_summary["Revenue Share, %"] = (
    paid_product_summary["Revenue"] / paid_product_summary["Revenue"].sum() * 100
).round(2)

paid_product_summary = paid_product_summary.sort_values("Revenue", ascending=False)

# Exclude low-volume categories so they do not distort product charts

EXCLUDED_PRODUCTS = ["Find yourself in IT"]
EXCLUDED_PAYMENT_TYPES = ["Reservation"]

paid_product_summary_filtered = paid_product_summary[
    ~paid_product_summary["Product"].isin(EXCLUDED_PRODUCTS)
].copy()

paid_deals_products_filtered = paid_deals[
    ~paid_deals["Product"].isin(EXCLUDED_PRODUCTS)
].copy()

if "Payment Type" in paid_deals_products_filtered.columns:
    paid_deals_products_filtered = paid_deals_products_filtered[
        ~paid_deals_products_filtered["Payment Type"].isin(EXCLUDED_PAYMENT_TYPES)
    ].copy()

# Main product metrics

products_count = paid_product_summary_filtered["Product"].nunique()
product_paid_deals_total = paid_product_summary_filtered["Paid_Deals"].sum()
product_revenue_total = paid_product_summary_filtered["Revenue"].sum()
product_avg_check_total = safe_divide(product_revenue_total, product_paid_deals_total)

top_product_row = paid_product_summary_filtered.sort_values(
    "Revenue",
    ascending=False
).head(1)

if not top_product_row.empty:
    top_product_name = top_product_row["Product"].iloc[0]
    top_product_revenue = top_product_row["Revenue"].iloc[0]
else:
    top_product_name = "-"
    top_product_revenue = np.nan

top_product_share = safe_divide(top_product_revenue, product_revenue_total) * 100


products_kpi_cards = dbc.Row(
    [
        dbc.Col(
            create_kpi_card(
                "Products",
                format_int(products_count),
                "Products with Paid Deals"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "Paid Deals",
                format_int(product_paid_deals_total),
                "For main products"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "Revenue",
                format_money(product_revenue_total),
                "Revenue by Product"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "Average Check",
                format_money(product_avg_check_total),
                "Revenue / Paid Deals"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "Revenue Leader",
                top_product_name,
                f"Share: {format_percent(top_product_share)}"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "Excluded",
                "1 product / 1 type",
                "Find yourself in IT, Reservation"
            ),
            md=2
        )
    ],
    className="g-3",
    style={"marginBottom": "24px"}
)

product_share_plot = paid_product_summary_filtered[
    paid_product_summary_filtered["Product"] != "Unknown"
].copy()

product_share_fig = go.Figure()

product_share_fig.add_trace(
    go.Bar(
        x=product_share_plot["Product"],
        y=product_share_plot["Paid Deals Share, %"],
        name="Paid Deals Share, %",
        marker_color=COLORS["primary"]
    )
)

product_share_fig.add_trace(
    go.Bar(
        x=product_share_plot["Product"],
        y=product_share_plot["Revenue Share, %"],
        name="Revenue Share, %",
        marker_color=COLORS["accent"]
    )
)

product_share_fig.update_layout(
    barmode="group",
    xaxis=dict(title="Product"),
    yaxis=dict(title="Share, %")
)

product_share_fig = apply_chart_layout(product_share_fig, "Products: Paid Deals Share and Revenue Share")
product_share_fig = apply_legend_style(product_share_fig, "top")


product_paid_avg_fig = go.Figure()

product_paid_avg_fig.add_trace(
    go.Bar(
        x=product_share_plot["Product"],
        y=product_share_plot["Paid_Deals"],
        name="Paid Deals",
        marker_color=COLORS["primary"],
        opacity=0.85
    )
)

product_paid_avg_fig.add_trace(
    go.Scatter(
        x=product_share_plot["Product"],
        y=product_share_plot["Avg_Revenue"],
        name="Average Check",
        mode="lines+markers",
        line=dict(color=COLORS["accent"], width=3),
        marker=dict(size=8),
        yaxis="y2"
    )
)

product_paid_avg_fig.update_layout(
    xaxis=dict(title="Product"),
    yaxis=dict(title="Paid Deals Count"),
    yaxis2=dict(
        title="Average Check",
        overlaying="y",
        side="right"
    )
)

product_paid_avg_fig = apply_chart_layout(product_paid_avg_fig, "Products: Paid Deals and Average Check")
product_paid_avg_fig = apply_legend_style(product_paid_avg_fig, "top")


payment_type_summary = (
    paid_deals_products_filtered
    .groupby("Payment Type", dropna=False)
    .agg(
        Paid_Deals=("Id", "count"),
        Revenue=("Initial Amount Paid", "sum"),
        Avg_Revenue=("Initial Amount Paid", "mean")
    )
    .reset_index()
)

payment_type_summary["Payment Type"] = payment_type_summary["Payment Type"].fillna("Unknown")
payment_type_summary = payment_type_summary.sort_values("Paid_Deals", ascending=False)

payment_type_fig = go.Figure()

payment_type_fig.add_trace(
    go.Bar(
        x=payment_type_summary["Payment Type"],
        y=payment_type_summary["Paid_Deals"],
        name="Paid Deals",
        marker_color=COLORS["primary"],
        opacity=0.85
    )
)

payment_type_fig.add_trace(
    go.Scatter(
        x=payment_type_summary["Payment Type"],
        y=payment_type_summary["Avg_Revenue"],
        name="Average Check",
        mode="lines+markers",
        line=dict(color=COLORS["accent"], width=3),
        marker=dict(size=8),
        yaxis="y2"
    )
)

payment_type_fig.update_layout(
    xaxis=dict(title="Payment Type"),
    yaxis=dict(title="Paid Deals Count"),
    yaxis2=dict(
        title="Average Check",
        overlaying="y",
        side="right"
    )
)

payment_type_fig = apply_chart_layout(payment_type_fig, "Payment Types: Paid Deals and Average Check")
payment_type_fig = apply_legend_style(payment_type_fig, "top")


product_payment_pivot = (
    paid_deals_products_filtered
    .pivot_table(
        index="Product",
        columns="Payment Type",
        values="Id",
        aggfunc="count",
        fill_value=0
    )
)

product_payment_pivot.index = product_payment_pivot.index.fillna("Unknown")
product_payment_pivot.columns = product_payment_pivot.columns.fillna("Unknown")

if "Unknown" in product_payment_pivot.index:
    product_payment_pivot = product_payment_pivot.drop(index="Unknown")

product_payment_pivot = product_payment_pivot.loc[
    product_payment_pivot.sum(axis=1).sort_values(ascending=False).index
]

product_payment_fig = go.Figure()

for payment_type in product_payment_pivot.columns:
    product_payment_fig.add_trace(
        go.Bar(
            x=product_payment_pivot.index,
            y=product_payment_pivot[payment_type],
            name=str(payment_type),
            marker_color=PAYMENT_TYPE_COLORS.get(str(payment_type), COLORS["secondary"]),
            hovertemplate=(
                "Product: %{x}<br>"
                f"Payment Type: {payment_type}<br>"
                "Paid Deals Count: %{y}<extra></extra>"
            )
        )
    )

product_payment_fig.update_layout(
    barmode="stack",
    xaxis=dict(title="Product"),
    yaxis=dict(title="Paid Deals Count")
)

product_payment_fig = apply_chart_layout(product_payment_fig, "Products: Payment Type Structure")
product_payment_fig = apply_legend_style(product_payment_fig, "top")

# Small note on product-level limitations
products_note = dbc.Alert(
    "Note: Find yourself in IT and Reservation payment type are excluded from product charts because their volume is minimal and does not affect the main conclusions.",
    color="light",
    style={
        "border": "1px solid #E5E7EB",
        "borderRadius": "14px",
        "color": COLORS["muted_text"],
        "fontSize": "13px",
        "marginBottom": "24px"
    }
)

# Product and education type link

education_type_col = find_col(
    paid_deals_products_filtered,
    [
        "Education Type",
        "Learning Type",
        "Study Type",
        "Course Type",
        "Product Type",
        "Training Type",
        "Education Type",
        "Education Format",
        "Course Type"
    ]
)

if education_type_col and "Product" in paid_deals_products_filtered.columns:
    product_education_data = paid_deals_products_filtered.copy()

    product_education_data["Product"] = product_education_data["Product"].fillna("Unknown")
    product_education_data[education_type_col] = product_education_data[education_type_col].fillna("Unknown")

    product_education_data = product_education_data[
        product_education_data["Product"] != "Unknown"
    ].copy()

    product_education_pivot = product_education_data.pivot_table(
        index="Product",
        columns=education_type_col,
        values="Id",
        aggfunc="count",
        fill_value=0
    )

    product_education_pivot = product_education_pivot.loc[
        product_education_pivot.sum(axis=1).sort_values(ascending=False).index
    ]

    if not product_education_pivot.empty:
        product_education_fig = go.Figure()

        education_colors = [
            COLORS["primary"],
            COLORS["secondary"],
            COLORS["accent"],
            COLORS["success"],
            "#A7D8F5",
            "#BFDFF2"
        ]

        for i, education_type in enumerate(product_education_pivot.columns):
            product_education_fig.add_trace(
                go.Bar(
                    x=product_education_pivot.index,
                    y=product_education_pivot[education_type],
                    name=str(education_type),
                    marker_color=education_colors[i % len(education_colors)],
                    hovertemplate=(
                        "Product: %{x}<br>"
                        f"Education Type: {education_type}<br>"
                        "Paid Deals Count: %{y}<extra></extra>"
                    )
                )
            )

        product_education_fig.update_layout(
            barmode="stack",
            xaxis=dict(title="Product"),
            yaxis=dict(title="Paid Deals Count")
        )

        product_education_fig = apply_chart_layout(
            product_education_fig,
            "Products: Education Type Structure"
        )

        product_education_fig = apply_legend_style(product_education_fig, "top")

    else:
        product_education_fig = create_empty_figure(
            "Products: Education Type Structure",
            "No data available for education type analysis"
        )

else:
    product_education_fig = create_empty_figure(
        "Products: Education Type Structure",
        "Education type column not found"
    )

product_table = paid_product_summary_filtered.copy()

product_table = product_table.rename(columns={
    "Paid_Deals": "Paid Deals",
    "Avg_Revenue": "Average Check",
    "Median_Revenue": "Median Check",
    "Product": "Product",
    "Revenue": "Revenue",
    "Paid Deals Share, %": "Paid Deals Share, %",
    "Revenue Share, %": "Revenue Share, %"
})

product_table = product_table[
    [
        "Product",
        "Paid Deals",
        "Revenue",
        "Average Check",
        "Median Check",
        "Paid Deals Share, %",
        "Revenue Share, %"
    ]
]

# Unit economics tab

unit_total_row = unit_economics_total.iloc[0].to_dict()


def get_unit_metric(possible_names):
    """
    Finds a metric in the unit_economics_total row by several possible names.
    This protects the dashboard from alternative column names such as C1, C1 %, ROMI %, ROMI, % and similar variants.
    """
    normalized_map = {
        str(col).strip().lower().replace(" ", "").replace("_", "").replace(",", "").replace("%", ""): col
        for col in unit_total_row.keys()
    }

    for name in possible_names:
        key = str(name).strip().lower().replace(" ", "").replace("_", "").replace(",", "").replace("%", "")
        if key in normalized_map:
            return unit_total_row[normalized_map[key]]

    return np.nan


unit_kpi_cards = dbc.Row(
    [
        dbc.Col(
            create_kpi_card(
                "UA",
                format_int(get_unit_metric(["UA", "Users", "Leads"])),
                "Potential Leads"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "B",
                format_int(get_unit_metric(["B", "Clients", "Customers"])),
                "Clients"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "C1",
                format_percent_auto(get_unit_metric(["C1", "C1 %", "C1, %", "Conversion", "Conversion %"])),
                "Lead-to-Client Conversion"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "AC",
                format_money(get_unit_metric(["AC", "Acquisition Cost"])),
                "Acquisition Cost"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "CPA",
                format_money(get_unit_metric(["CPA", "Cost per Lead"])),
                "Cost per Lead"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "CAC",
                format_money(get_unit_metric(["CAC", "Cost per Customer"])),
                "Customer Acquisition Cost"
            ),
            md=2
        ),
    ],
    className="g-3",
    style={"marginBottom": "16px"}
)


unit_kpi_cards_2 = dbc.Row(
    [
        dbc.Col(
            create_kpi_card(
                "Revenue_I",
                format_money(get_unit_metric(["Revenue_I", "Revenue I", "Revenue", "Calculated Revenue"])),
                "Calculated Revenue"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "CLTV",
                format_money(get_unit_metric(["CLTV"])),
                "Customer Lifetime Value"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "LTV",
                format_money(get_unit_metric(["LTV"])),
                "Lead Value"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "CM",
                format_money(get_unit_metric(["CM", "Contribution Margin"])),
                "Contribution Margin"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "ROMI",
                format_percent_auto(get_unit_metric(["ROMI", "ROMI %", "ROMI, %", "Return on Marketing"])),
                "Marketing ROI"
            ),
            md=2
        ),
        dbc.Col(
            create_kpi_card(
                "Experiment",
                "11 days",
                "Can be run within 14 days"
            ),
            md=2
        ),
    ],
    className="g-3",
    style={"marginBottom": "24px"}
)

product_col = find_col(unit_economics_display, ["Product", "Product"])
revenue_col = find_col(unit_economics_display, ["Revenue", "Revenue_I"])
cm_col = find_col(unit_economics_display, ["CM"])
cac_col = find_col(unit_economics_display, ["CAC"])
cltv_col = find_col(unit_economics_display, ["CLTV"])
b_col = find_col(unit_economics_display, ["B", "Clients", "Clients"])

# Some unit economics values may be stored as text, so they are explicitly converted to numeric

unit_numeric_cols = [
    revenue_col,
    cm_col,
    cac_col,
    cltv_col,
    b_col
]

for col in unit_numeric_cols:
    if col and col in unit_economics_display.columns:
        unit_economics_display[col] = pd.to_numeric(
            unit_economics_display[col]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.replace("%", "", regex=False),
            errors="coerce"
        )

# Keep only products with clients in the calculation base for unit economics charts
UNIT_EXCLUDED_PRODUCTS = ["Data Analytics", "Find yourself in IT", "Unknown"]

if product_col and revenue_col and cm_col:
    unit_product_plot = unit_economics_display[
        ~unit_economics_display[product_col].astype(str).isin(UNIT_EXCLUDED_PRODUCTS)
    ].copy()

    unit_revenue_cm_fig = go.Figure()

    unit_revenue_cm_fig.add_trace(
        go.Bar(
            x=unit_product_plot[product_col],
            y=unit_product_plot[revenue_col],
            name="Revenue",
            marker_color=COLORS["primary"],
            opacity=0.85
        )
    )

    unit_revenue_cm_fig.add_trace(
        go.Scatter(
            x=unit_product_plot[product_col],
            y=unit_product_plot[cm_col],
            name="CM",
            mode="lines+markers",
            line=dict(color=COLORS["accent"], width=3),
            marker=dict(size=8),
            yaxis="y2"
        )
    )

    unit_revenue_cm_fig.update_layout(
        xaxis=dict(title="Product"),
        yaxis=dict(title="Revenue"),
        yaxis2=dict(
            title="CM",
            overlaying="y",
            side="right"
        )
    )

    unit_revenue_cm_fig = apply_chart_layout(unit_revenue_cm_fig, "Products: Calculated Revenue and CM")
    unit_revenue_cm_fig = apply_legend_style(unit_revenue_cm_fig, "top")

else:
    unit_revenue_cm_fig = create_empty_figure("Revenue and CM by Product")


if product_col and cac_col and cltv_col:
    cac_cltv_plot = unit_economics_display.copy()

    cac_cltv_plot = cac_cltv_plot[
        (~cac_cltv_plot[product_col].astype(str).isin(UNIT_EXCLUDED_PRODUCTS)) &
        (cac_cltv_plot[cac_col].notna()) &
        (cac_cltv_plot[cltv_col].notna()) &
        (cac_cltv_plot[cac_col] > 0) &
        (cac_cltv_plot[cltv_col] > 0)
    ].copy()

    if b_col and b_col in cac_cltv_plot.columns:
        cac_cltv_plot["Bubble Size"] = cac_cltv_plot[b_col].clip(lower=1)
    elif cm_col and cm_col in cac_cltv_plot.columns:
        cac_cltv_plot["Bubble Size"] = cac_cltv_plot[cm_col].clip(lower=1)
    else:
        cac_cltv_plot["Bubble Size"] = 1

    if not cac_cltv_plot.empty:
        cac_cltv_fig = px.scatter(
            cac_cltv_plot,
            x=cac_col,
            y=cltv_col,
            color=product_col,
            hover_name=product_col,
            size="Bubble Size",
            size_max=50,
            labels={
                cac_col: "CAC",
                cltv_col: "CLTV",
                product_col: "Product",
                "Bubble Size": "Size"
            }
        )

        cac_cltv_fig.update_traces(
            marker=dict(opacity=0.75, line=dict(width=0.5, color="#1F2937")),
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "CAC: %{x:,.2f}<br>"
                "CLTV: %{y:,.2f}<extra></extra>"
            )
        )

        cac_cltv_fig = apply_chart_layout(cac_cltv_fig, "Products: CAC and CLTV")
        cac_cltv_fig = apply_legend_style(cac_cltv_fig, "top")

    else:
        cac_cltv_fig = create_empty_figure(
            "CAC vs CLTV by Product",
            "No products with positive CAC and CLTV"
        )

else:
    cac_cltv_fig = create_empty_figure("CAC vs CLTV by Product")

# The chart shows the effect of C1 improvement in the scenario analysis

growth_required_cols = [
    "Product",
    "Scenario",
    "Baseline CM",
    "Scenario CM",
    "CM Uplift",
    "CM Uplift, %"
]

if all(col in best_growth_points.columns for col in growth_required_cols):
    growth_plot = best_growth_points.copy()

    # Keep only products with clients in the calculation base
    growth_plot = growth_plot[
        ~growth_plot["Product"].astype(str).isin(
            ["Data Analytics", "Find yourself in IT", "Unknown"]
        )
    ].copy()

    numeric_growth_cols = [
        "Baseline CM",
        "Scenario CM",
        "CM Uplift",
        "CM Uplift, %"
    ]

    for col in numeric_growth_cols:
        growth_plot[col] = pd.to_numeric(growth_plot[col], errors="coerce")

    growth_plot = growth_plot[
        growth_plot["Baseline CM"].notna() &
        growth_plot["Scenario CM"].notna()
    ].copy()

    growth_plot = growth_plot.sort_values("CM Uplift", ascending=True)

    if not growth_plot.empty:
        growth_plot["Label"] = (
            growth_plot["Product"].astype(str)
            + " | "
            + growth_plot["Scenario"].astype(str)
        )

        cm_growth_fig = go.Figure()

        cm_growth_fig.add_trace(
            go.Bar(
                y=growth_plot["Label"],
                x=growth_plot["Baseline CM"],
                name="Baseline CM",
                orientation="h",
                marker_color=COLORS["primary"],
                opacity=0.75,
                customdata=np.stack(
                    [
                        growth_plot["Scenario CM"],
                        growth_plot["CM Uplift"],
                        growth_plot["CM Uplift, %"]
                    ],
                    axis=-1
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Baseline CM: %{x:,.2f}<br>"
                    "Scenario CM: %{customdata[0]:,.2f}<br>"
                    "CM Uplift: %{customdata[1]:,.2f}<br>"
                    "Uplift, %: %{customdata[2]:.2f}%<extra></extra>"
                )
            )
        )

        cm_growth_fig.add_trace(
            go.Bar(
                y=growth_plot["Label"],
                x=growth_plot["Scenario CM"],
                name="Scenario CM",
                orientation="h",
                marker_color=COLORS["accent"],
                opacity=0.75,
                customdata=np.stack(
                    [
                        growth_plot["Baseline CM"],
                        growth_plot["CM Uplift"],
                        growth_plot["CM Uplift, %"]
                    ],
                    axis=-1
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Baseline CM: %{customdata[0]:,.2f}<br>"
                    "Scenario CM: %{x:,.2f}<br>"
                    "CM Uplift: %{customdata[1]:,.2f}<br>"
                    "Uplift, %: %{customdata[2]:.2f}%<extra></extra>"
                )
            )
        )

        # Uplift labels to the right of the bars
        for _, row in growth_plot.iterrows():
            cm_growth_fig.add_annotation(
                x=row["Scenario CM"],
                y=row["Label"],
                text=f"+{row['CM Uplift']:,.0f}".replace(",", " "),
                showarrow=False,
                xanchor="left",
                xshift=8,
                font=dict(
                    size=11,
                    color=COLORS["success"]
                )
            )

        cm_growth_fig.update_layout(
            barmode="group",
            xaxis=dict(title="CM"),
            yaxis=dict(title=""),
            height=360
        )

        cm_growth_fig = apply_chart_layout(
            cm_growth_fig,
            "CM Before and After C1 Improvement"
        )

        cm_growth_fig = apply_legend_style(cm_growth_fig, "top")

    else:
        cm_growth_fig = create_empty_figure(
            "CM Before and After C1 Improvement",
            "No rows with filled Baseline CM and Scenario CM values"
        )

else:
    missing_cols = [
        col for col in growth_required_cols
        if col not in best_growth_points.columns
    ]

    cm_growth_fig = create_empty_figure(
        "CM Before and After C1 Improvement",
        f"Missing columns: {', '.join(missing_cols)}"
    )

experiment_text = html.Div(
    [
        html.H5(
            "HADI Hypothesis",
            style={
                "fontWeight": "700",
                "color": COLORS["text"],
                "marginBottom": "12px"
            }
        ),

        html.Div(
            "Hypothesis",
            style={
                "fontWeight": "700",
                "fontSize": "13px",
                "color": COLORS["text"],
                "marginBottom": "4px"
            }
        ),
        html.P(
            "If an updated first-contact and follow-up script is implemented for new leads, "
            "then lead-to-client conversion C1 will increase by at least 10%.",
            style={
                "fontSize": "13px",
                "color": COLORS["muted_text"],
                "lineHeight": "1.45",
                "marginBottom": "12px"
            }
        ),

        html.Div(
            "Experiment Parameters",
            style={
                "fontWeight": "700",
                "fontSize": "13px",
                "color": COLORS["text"],
                "marginBottom": "6px"
            }
        ),

        html.Div("Metric: C1", style={"fontSize": "13px", "marginBottom": "4px"}),
        html.Div("Expected uplift: +10%", style={"fontSize": "13px", "marginBottom": "4px"}),
        html.Div("Sample size: 560 leads", style={"fontSize": "13px", "marginBottom": "4px"}),
        html.Div("Average flow: 51 leads per day", style={"fontSize": "13px", "marginBottom": "4px"}),
        html.Div("Experiment duration: 11 days", style={"fontSize": "13px", "marginBottom": "8px"}),

        html.P(
            "The scenario analysis above shows the effect of a C1 +5% change. The HADI hypothesis tests a more pronounced experimental effect of C1 +10%.",
            style={
                "fontSize": "12px",
                "color": COLORS["muted_text"],
                "lineHeight": "1.45",
                "marginTop": "10px",
                "marginBottom": "0"
            }
        ),

        html.Div(
            "The experiment can be run within 14 days",
            style={
                "fontSize": "13px",
                "fontWeight": "700",
                "color": COLORS["success"],
                "marginTop": "8px"
            }
        )
    ]
)

experiment_card = dbc.Card(
    dbc.CardBody([experiment_text]),
    style={
        "borderRadius": "14px",
        "border": "1px solid #E5E7EB",
        "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.04)",
        "backgroundColor": COLORS["card"],
        "height": "100%"
    }
)

# Geography and German language level are not treated as separate analysis sections because of the high Unknown share
geo_data_note = dbc.Alert(
    [
        html.Div(
            "Limitation of geographic and language-level analysis",
            style={
                "fontWeight": "700",
                "color": COLORS["text"],
                "marginBottom": "6px"
            }
        ),
        html.Div(
            "City and Level of Deutsch have low completeness: around 90% of city values and around 98% of German level values are Unknown. "
            "Therefore, these dimensions are not included in the main dashboard charts. Berlin and B1 stand out only within the filled subset of the data, so conclusions should be interpreted carefully.",
            style={
                "color": COLORS["muted_text"],
                "fontSize": "13px",
                "lineHeight": "1.45"
            }
        )
    ],
    color="light",
    style={
        "border": "1px solid #E5E7EB",
        "borderRadius": "14px",
        "backgroundColor": COLORS["card"],
        "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.04)",
        "padding": "14px 16px",
        "marginBottom": "0",
        "height": "fit-content"
    }
)

geo_quality_df = pd.DataFrame({
    "Field": ["City", "Level of Deutsch"],
    "Filled, %": [10.02, 2.30],
    "Missing, %": [89.98, 97.70]
})

geo_quality_fig = go.Figure()

geo_quality_fig.add_trace(
    go.Bar(
        y=geo_quality_df["Field"],
        x=geo_quality_df["Filled, %"],
        name="Filled, %",
        orientation="h",
        marker_color="#A7D8F5",
        text=geo_quality_df["Filled, %"],
        texttemplate="%{text:.2f}%",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "Field: %{y}<br>"
            "Filled: %{x:.2f}%<extra></extra>"
        )
    )
)

geo_quality_fig.add_trace(
    go.Bar(
        y=geo_quality_df["Field"],
        x=geo_quality_df["Missing, %"],
        name="Missing, %",
        orientation="h",
        marker_color="#EEF2F7",
        text=geo_quality_df["Missing, %"],
        texttemplate="%{text:.2f}%",
        textposition="inside",
        hovertemplate=(
            "Field: %{y}<br>"
            "Missing: %{x:.2f}%<extra></extra>"
        )
    )
)

geo_quality_fig.update_layout(
    barmode="stack",
    xaxis=dict(
        title="Share, %",
        range=[0, 105]
    ),
    yaxis=dict(
        title="",
        autorange="reversed"
    ),
    height=280
)

geo_quality_fig = apply_chart_layout(
    geo_quality_fig,
    "Completeness of City and Level of Deutsch"
)

geo_quality_fig.update_layout(
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.08,
        xanchor="center",
        x=0.5,
        title_text="",
        font=dict(size=10)
    ),
    margin=dict(l=80, r=40, t=95, b=45)
)
# Dashboard tab assembly

# The main tabs contain many charts, so they are split into shorter subtabs.
# This makes each screen easier to read and better suited for a 16:9 presentation format.

def create_inner_tabs(tabs):
    return dbc.Tabs(
        tabs,
        className="mb-3",
        style={"marginBottom": "16px"}
    )


overview_tab = html.Div(
    [
        create_section_header(
            "Online School Overview",
            "This tab shows the overall business picture: funnel, paid deals, revenue, conversion and monthly dynamics."
        ),

        create_inner_tabs(
            [
                dbc.Tab(
                    label="KPI and Funnel",
                    children=[
                        overview_kpi_cards,
                        dbc.Row(
                            [
                                dbc.Col(create_chart_card(stage_fig), md=5),
                                dbc.Col(create_chart_card(monthly_deals_fig), md=7)
                            ],
                            className="g-3"
                        )
                    ]
                ),
                dbc.Tab(
                    label="Dynamics",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(create_chart_card(monthly_deals_fig), md=7),
                                dbc.Col(create_chart_card(monthly_conversion_fig), md=5)
                            ],
                            className="g-3"
                        )
                    ]
                ),
                dbc.Tab(
                    label="Sources and Products",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(create_chart_card(top_sources_revenue_fig), md=6),
                                dbc.Col(create_chart_card(top_products_revenue_fig), md=6)
                            ],
                            className="g-3"
                        )
                    ]
                ),
                dbc.Tab(
                    label="Data Quality",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(create_chart_card(geo_quality_fig, graph_height=300), md=8),
                                dbc.Col(geo_data_note, md=4)
                            ],
                            className="g-3"
                        )
                    ]
                )
            ]
        )
    ]
)


marketing_tab = html.Div(
    [
        create_section_header(
            "Marketing",
            "This tab shows acquisition channel performance: spend, revenue, conversion, cost per lead, customer acquisition cost and lead quality."
        ),

        create_inner_tabs(
            [
                dbc.Tab(
                    label="Sources",
                    children=[
                        marketing_kpi_cards,
                        dbc.Row(
                            [
                                dbc.Col(create_chart_card(source_bubble_fig), md=12)
                            ],
                            className="g-3"
                        )
                    ]
                ),
                dbc.Tab(
                    label="Spend and ROI",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(create_chart_card(spend_revenue_fig), md=12)
                            ],
                            className="g-3"
                        )
                    ]
                ),
                dbc.Tab(
                    label="Quality and Campaigns",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(create_chart_card(quality_fig), md=5),
                                dbc.Col(create_chart_card(campaign_fig), md=7)
                            ],
                            className="g-3"
                        )
                    ]
                ),
                dbc.Tab(
                    label="Campaign Table",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(create_table_card(campaign_table, page_size=10), md=12)
                            ],
                            className="g-3"
                        )
                    ]
                )
            ]
        )
    ]
)


sales_tab = html.Div(
    [
        create_section_header(
            "Sales",
            "This tab shows sales team performance: workload, conversion, revenue, average check, call activity, SLA and the link between sales and acquisition sources."
        ),

        create_inner_tabs(
            [
                dbc.Tab(
                    label="Managers",
                    children=[
                        sales_kpi_cards,
                        dbc.Row(
                            [
                                dbc.Col(create_chart_card(manager_bubble_fig), md=6),
                                dbc.Col(create_chart_card(manager_share_fig), md=6)
                            ],
                            className="g-3"
                        )
                    ]
                ),
                dbc.Tab(
                    label="Revenue and Check",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(create_chart_card(manager_paid_revenue_fig), md=12)
                            ],
                            className="g-3"
                        )
                    ]
                ),
                dbc.Tab(
                    label="SLA and Calls",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(create_chart_card(manager_calls_fig), md=6),
                                dbc.Col(create_chart_card(sla_fig), md=6)
                            ],
                            className="g-3"
                        )
                    ]
                ),
                dbc.Tab(
                    label="Links",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(create_chart_card(manager_source_fig), md=6),
                                dbc.Col(create_chart_card(manager_campaign_fig), md=6)
                            ],
                            className="g-3"
                        )
                    ]
                )
            ]
        )
    ]
)


products_tab = html.Div(
    [
        create_section_header(
            "Products",
            "This tab shows the product sales structure: paid deals, revenue, average check, payment types and distribution by education type."
        ),

        create_inner_tabs(
            [
                dbc.Tab(
                    label="Products",
                    children=[
                        products_kpi_cards,
                        dbc.Row(
                            [
                                dbc.Col(create_chart_card(product_share_fig), md=6),
                                dbc.Col(create_chart_card(product_paid_avg_fig), md=6)
                            ],
                            className="g-3"
                        )
                    ]
                ),
                dbc.Tab(
                    label="Payment Types",
                    children=[
                        products_note,
                        dbc.Row(
                            [
                                dbc.Col(create_chart_card(payment_type_fig), md=5),
                                dbc.Col(create_chart_card(product_payment_fig), md=7)
                            ],
                            className="g-3"
                        )
                    ]
                ),
                dbc.Tab(
                    label="Education Type",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(create_chart_card(product_education_fig), md=12)
                            ],
                            className="g-3"
                        )
                    ]
                ),
                dbc.Tab(
                    label="Details",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(create_table_card(product_table, page_size=10), md=12)
                            ],
                            className="g-3"
                        )
                    ]
                )
            ]
        )
    ]
)


unit_note = dbc.Alert(
    [
        html.Div(
            "Product Unit Economics Limitation",
            style={
                "fontWeight": "700",
                "color": COLORS["text"],
                "marginBottom": "6px"
            }
        ),
        html.Div(
            "Unit economics charts include only products with clients in the calculation base: Digital Marketing, UX/UI Design and Web Developer. "
            "Data Analytics and Find yourself in IT are not included in the scenario analysis because they have no clients matching the calculation conditions.",
            style={
                "color": COLORS["muted_text"],
                "fontSize": "13px",
                "lineHeight": "1.45"
            }
        )
    ],
    color="light",
    style={
        "border": "1px solid #E5E7EB",
        "borderRadius": "14px",
        "backgroundColor": COLORS["card"],
        "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.04)",
        "padding": "14px 16px",
        "marginBottom": "16px"
    }
)

unit_table_display = unit_economics_display.copy()

unit_table_display = unit_table_display.rename(columns={
    "Product": "Product",
    "UA": "UA",
    "B": "Clients",
    "C1, %": "C1, %",
    "AC": "AC",
    "CPA": "CPA",
    "CAC": "CAC",
    "COGS": "COGS",
    "Revenue": "Revenue_I",
    "Revenue_I": "Revenue_I",
    "T": "T",
    "AOV": "AOV",
    "APC": "APC",
    "CLTV": "CLTV",
    "LTV": "LTV",
    "CM": "CM",
    "ROMI, %": "ROMI, %"
})

# Show products with clients and revenue first; products without a calculation base remain at the bottom

unit_table_display["Clients"] = pd.to_numeric(
    unit_table_display["Clients"],
    errors="coerce"
).fillna(0)

unit_table_display["Revenue_I"] = pd.to_numeric(
    unit_table_display["Revenue_I"],
    errors="coerce"
).fillna(0)

unit_table_display["CM"] = pd.to_numeric(
    unit_table_display["CM"],
    errors="coerce"
).fillna(0)

unit_table_display["Has Clients"] = unit_table_display["Clients"] > 0

unit_table_display = unit_table_display.sort_values(
    by=["Has Clients", "Revenue_I", "CM"],
    ascending=[False, False, False]
).drop(columns=["Has Clients"])

unit_table_intro = html.Div(
    [
        html.H4(
            "Product-level Unit Economics Details",
            style={
                "color": COLORS["text"],
                "fontWeight": "700",
                "marginBottom": "8px"
            }
        ),
        html.P(
            "The table shows calculated unit economics metrics by product: UA, clients, C1, acquisition cost, "
            "CAC, calculated revenue, CLTV, LTV, CM and ROMI. Products without clients in the calculation base are kept in the table as a data limitation, "
            "but are not used in the growth-point scenario analysis.",
            style={
                "color": COLORS["muted_text"],
                "fontSize": "14px",
                "lineHeight": "1.5",
                "marginBottom": "16px"
            }
        )
    ],
    style={"marginTop": "8px"}
)

unit_tab = html.Div(
    [
        create_section_header(
            "Unit Economics",
            "This tab shows overall project unit economics, product-level unit economics, the main growth point and HADI hypothesis parameters."
        ),

        create_inner_tabs(
            [
                dbc.Tab(
                    label="Overall Metrics",
                    children=[
                        unit_kpi_cards,
                        unit_kpi_cards_2,
                        dbc.Row(
                            [
                                dbc.Col(create_chart_card(unit_revenue_cm_fig), md=6),
                                dbc.Col(create_chart_card(cac_cltv_fig), md=6)
                            ],
                            className="g-3"
                        )
                    ]
                ),
                dbc.Tab(
                    label="Growth Point",
                    children=[
                        unit_note,
                        dbc.Row(
                            [
                                dbc.Col(create_chart_card(cm_growth_fig), md=8),
                                dbc.Col(experiment_card, md=4)
                            ],
                            className="g-3"
                        )
                    ]
                ),
                dbc.Tab(
                    label="Details",
                    children=[
                        unit_table_intro,
                        dbc.Row(
                            [
                                dbc.Col(create_table_card(unit_table_display, page_size=10), md=12)
                            ],
                            className="g-3"
                        )
                    ]
                )
            ]
        )
    ]
)


# Dash app setup

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

app.title = "Online School Performance Dashboard"

app.layout = dbc.Container(
    [
        html.Div(
            [
                html.H1(
                    "Online School Performance Dashboard",
                    style={
                        "color": COLORS["text"],
                        "fontWeight": "800",
                        "marginBottom": "8px"
                    }
                ),
                html.P(
                    "Interactive dashboard for analyzing online school sales, marketing, products and unit economics.",
                    style={
                        "color": COLORS["muted_text"],
                        "fontSize": "16px",
                        "marginBottom": "16px"
                    }
                )
            ],
            style={"paddingTop": "16px"}
        ),

        dcc.Tabs(
            id="dashboard-tabs",
            value="overview",
            children=[
                dcc.Tab(label="Overview", value="overview", children=overview_tab),
                dcc.Tab(label="Marketing", value="marketing", children=marketing_tab),
                dcc.Tab(label="Sales", value="sales", children=sales_tab),
                dcc.Tab(label="Products", value="products", children=products_tab),
                dcc.Tab(label="Unit Economics", value="unit", children=unit_tab)
            ],
            style={"marginBottom": "16px"}
        )
    ],
    fluid=True,
    style={
        "backgroundColor": COLORS["background"],
        "minHeight": "100vh",
        "paddingLeft": "28px",
        "paddingRight": "28px",
        "paddingBottom": "20px"
    }
)


# Run the app

if __name__ == "__main__":
    app.run(debug=True)