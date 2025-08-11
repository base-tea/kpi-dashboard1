# streamlit_dashboard.py
# -*- coding: utf-8 -*-
"""
KPI-Dashboard Vorlage mit Streamlit + Plotly

Eigenschaften
- Seitenlayout mit Sidebar-Filtern
- CSV-Upload ODER synthetische Beispieldaten
- KPI-Kacheln mit Vorperioden-Vergleich
- Zeitreihen-, Kategorien- und Kanal-Sicht
- Export der gefilterten Daten

Start
  pip install streamlit pandas numpy plotly python-dateutil
  streamlit run streamlit_dashboard.py

CSV-Schema (wenn eigener Upload)
  date, region, category, channel, product, orders, revenue
  date: ISO-Format (YYYY-MM-DD) oder DD.MM.YYYY
  revenue: Zahl in CHF
"""

from __future__ import annotations
import io
import os
import math
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# Seiteneinstellungen
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="KPI-Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------
def fmt_chf(x: float) -> str:
    try:
        return "CHF " + f"{x:,.0f}".replace(",", "'")
    except Exception:
        return "CHF -"

def parse_date(s: str) -> pd.Timestamp:
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d"):
        try:
            return pd.to_datetime(datetime.strptime(s, fmt))
        except Exception:
            continue
    # Fallback auf pandas Parser
    return pd.to_datetime(s, errors="coerce")


@st.cache_data(show_spinner=False)
def load_example_data(n_months: int = 18, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = (pd.Timestamp.today().normalize() - relativedelta(months=n_months - 1)).replace(day=1)
    dates = pd.date_range(start, periods=n_months, freq="MS")

    regions = ["Nord", "Süd", "Ost", "West", "Zentral"]
    categories = ["Ausrüstung", "Verbrauch", "Service"]
    channels = ["Direkt", "Online", "Partner"]
    products = [f"P{i:02d}" for i in range(1, 21)]

    rows = []
    for d in dates:
        for r in regions:
            for c in categories:
                for ch in channels:
                    base_orders = rng.poisson(12)
                    rev_per_order = rng.normal(450, 120)
                    orders = max(0, base_orders + rng.integers(-3, 4))
                    revenue = max(0.0, orders * max(80, rev_per_order))
                    # Produktmix stichprobenartig
                    for _ in range(max(1, orders // 6)):
                        p = rng.choice(products)
                        rows.append(
                            {
                                "date": d,
                                "region": r,
                                "category": c,
                                "channel": ch,
                                "product": p,
                                "orders": int(max(1, rng.poisson(2))),
                                "revenue": float(max(20, rng.normal(revenue / max(1, orders), 60))),
                            }
                        )
    df = pd.DataFrame(rows)
    # Aggregation auf Monatsebene pro Dimensionenkombi
    df = (
        df.groupby(["date", "region", "category", "channel", "product"], as_index=False)
        .agg({"orders": "sum", "revenue": "sum"})
        .sort_values("date")
    )
    return df

def try_read_csv(upload: bytes | None) -> pd.DataFrame | None:
    if not upload:
        return None
    try:
        df = pd.read_csv(io.BytesIO(upload))
    except Exception:
        try:
            df = pd.read_excel(io.BytesIO(upload))
        except Exception:
            return None
    # Spalten normalisieren
    cols = {c.lower().strip(): c for c in df.columns}
    def col(name):
        for k in cols:
            if k == name:
                return cols[k]
        return None
    required = ["date", "region", "category", "channel", "product", "orders", "revenue"]
    missing = [c for c in required if col(c) is None]
    if missing:
        st.warning("Fehlende Spalten im Upload: " + ", ".join(missing))
        return None
    # Umbenennen in Standardschema
    df = df.rename(columns={cols[k]: k for k in cols})
    df["date"] = df["date"].apply(lambda x: parse_date(str(x)))
    df["orders"] = pd.to_numeric(df["orders"], errors="coerce").fillna(0).astype(int)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

def filter_df(df: pd.DataFrame, d_from: pd.Timestamp, d_to: pd.Timestamp,
              regions, categories, channels) -> pd.DataFrame:
    m = (
        (df["date"] >= d_from) & (df["date"] <= d_to)
        & (df["region"].isin(regions))
        & (df["category"].isin(categories))
        & (df["channel"].isin(channels))
    )
    return df.loc[m].copy()

def kpi_block(df_curr: pd.DataFrame, df_prev: pd.DataFrame):
    revenue_curr = df_curr["revenue"].sum()
    revenue_prev = df_prev["revenue"].sum()
    orders_curr = df_curr["orders"].sum()
    orders_prev = df_prev["orders"].sum()
    aov_curr = revenue_curr / orders_curr if orders_curr else 0
    aov_prev = revenue_prev / orders_prev if orders_prev else 0

    def delta_pct(curr, prev):
        if prev == 0:
            return 0.0 if curr == 0 else 100.0
        return (curr - prev) / prev * 100.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Umsatz", fmt_chf(revenue_curr), f"{delta_pct(revenue_curr, revenue_prev):+.1f}%")
    with c2:
        st.metric("Bestellungen", f"{orders_curr:,}".replace(",", "'"), f"{delta_pct(orders_curr, orders_prev):+.1f}%")
    with c3:
        st.metric("Ø Bestellung", fmt_chf(aov_curr), f"{delta_pct(aov_curr, aov_prev):+.1f}%")
    with c4:
        st.metric("Periode", f"{df_curr['date'].min().date()} – {df_curr['date'].max().date()}")

# -----------------------------------------------------------------------------
# Datenquelle auswählen
# -----------------------------------------------------------------------------
st.sidebar.header("Datenquelle")
uploaded = st.sidebar.file_uploader("CSV oder Excel hochladen", type=["csv", "xlsx"])

if uploaded and uploaded.size > 0:
    df_raw = try_read_csv(uploaded.getvalue())
    source = "Upload"
else:
    df_raw = load_example_data()
    source = "Beispieldaten"

if df_raw is None or df_raw.empty:
    st.error("Keine gültigen Daten geladen")
    st.stop()

st.caption(f"Quelle: {source}")

# -----------------------------------------------------------------------------
# Filter in der Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("Filter")
min_date, max_date = df_raw["date"].min(), df_raw["date"].max()

# Standardperiode: letzte 6 Monate
default_from = (max_date - relativedelta(months=5)).replace(day=1)

d_from, d_to = st.sidebar.date_input(
    "Zeitraum",
    value=(default_from.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)
if isinstance(d_from, tuple):
    # Streamlit < 1.30 fallback
    d_from, d_to = d_from

d_from = pd.to_datetime(d_from)
d_to = pd.to_datetime(d_to)

regions = st.sidebar.multiselect("Region", options=sorted(df_raw["region"].unique()), default=list(sorted(df_raw["region"].unique())))
categories = st.sidebar.multiselect("Kategorie", options=sorted(df_raw["category"].unique()), default=list(sorted(df_raw["category"].unique())))
channels = st.sidebar.multiselect("Kanal", options=sorted(df_raw["channel"].unique()), default=list(sorted(df_raw["channel"].unique())))

# Vorperiode: gleicher Längenbereich unmittelbar davor
period_len = (d_to - d_from).days + 1
prev_to = d_from - timedelta(days=1)
prev_from = prev_to - timedelta(days=period_len - 1)

# -----------------------------------------------------------------------------
# Gefilterte Daten
# -----------------------------------------------------------------------------
df_curr = filter_df(df_raw, d_from, d_to, regions, categories, channels)
df_prev = filter_df(df_raw, prev_from, prev_to, regions, categories, channels)

st.title("KPI-Dashboard")

# KPI-Block
kpi_block(df_curr, df_prev)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Zeit", "Kategorien", "Kanäle", "Tabelle"])

with tab1:
    # Zeitreihe Umsatz
    ts = (
        df_curr.groupby(pd.Grouper(key="date", freq="MS"), as_index=False)
        .agg({"revenue": "sum", "orders": "sum"})
    )
    fig = px.line(ts, x="date", y="revenue", markers=True, title="Umsatz über Zeit")
    fig.update_layout(yaxis_title="Umsatz (CHF)", xaxis_title="Monat")
    st.plotly_chart(fig, use_container_width=True)

    # Vergleich aktuelle vs. Vorperiode
    col1, col2 = st.columns(2)
    with col1:
        by_reg = df_curr.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        st.subheader("Umsatz nach Region")
        st.plotly_chart(px.bar(by_reg, x="region", y="revenue"), use_container_width=True)
    with col2:
        by_cat = df_curr.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        st.subheader("Umsatz nach Kategorie")
        st.plotly_chart(px.bar(by_cat, x="category", y="revenue"), use_container_width=True)

with tab2:
    # Kategorie-Drilldown
    by_cat_month = (
        df_curr.groupby([pd.Grouper(key="date", freq="MS"), "category"], as_index=False)["revenue"].sum()
    )
    st.plotly_chart(px.area(by_cat_month, x="date", y="revenue", color="category", groupnorm=None,
                            title="Umsatz je Kategorie über Zeit"), use_container_width=True)

    # Produkte Top 10 (falls vorhanden)
    if "product" in df_curr.columns and not df_curr["product"].isna().all():
        top_products = (
            df_curr.groupby("product", as_index=False)["revenue"].sum()
            .sort_values("revenue", ascending=False)
            .head(10)
        )
        st.subheader("Top 10 Produkte nach Umsatz")
        st.plotly_chart(px.bar(top_products, x="product", y="revenue"), use_container_width=True)

with tab3:
    by_channel = df_curr.groupby("channel", as_index=False)["revenue"].sum()
    st.plotly_chart(px.pie(by_channel, names="channel", values="revenue", hole=0.5,
                           title="Umsatzanteile nach Kanal"), use_container_width=True)

    conv = df_curr.groupby("channel", as_index=False).agg({"orders": "sum"})
    conv["avg_order_value"] = df_curr.groupby("channel")["revenue"].sum().values / conv["orders"].replace(0, np.nan)
    conv = conv.fillna(0)
    st.subheader("Bestellungen & AOV nach Kanal")
    st.dataframe(conv.rename(columns={"orders": "Bestellungen", "avg_order_value": "Ø Bestellwert"}))

with tab4:
    st.download_button(
        label="Gefilterte Daten als CSV",
        data=df_curr.to_csv(index=False).encode("utf-8"),
        file_name="filtered_data.csv",
        mime="text/csv",
    )
    st.dataframe(df_curr)

# Fusszeile
st.caption("Vorlage – ersetze die Beispieldaten durch deinen CSV-Upload oder binde eine Datenbank an")
