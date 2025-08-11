# streamlit_dashboard.py
# -*- coding: utf-8 -*-
"""
KPI-Dashboard (erweitert & robust)
- Zusätzliche KPI: Marge (optional via cost-Spalte oder Annahme)
- Wachstum: MoM und YoY
- Granularität: Monat oder Woche
- Export: Excel mit mehreren Sheets
- Robuste Guards bei leeren Filterergebnissen
"""
from __future__ import annotations
import io
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="KPI-Dashboard", layout="wide", initial_sidebar_state="expanded")

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
                    for _ in range(max(1, orders // 6)):
                        p = rng.choice(products)
                        rows.append(
                            {"date": d, "region": r, "category": c, "channel": ch, "product": p,
                             "orders": int(max(1, rng.poisson(2))),
                             "revenue": float(max(20, rng.normal(revenue / max(1, orders), 60)))})
    df = pd.DataFrame(rows)
    df = (df.groupby(["date","region","category","channel","product"], as_index=False)
            .agg({"orders":"sum","revenue":"sum"}).sort_values("date"))
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
    cols = {c.lower().strip(): c for c in df.columns}
    def col(name):
        for k in cols:
            if k == name:
                return cols[k]
        return None
    required = ["date","region","category","channel","product","orders","revenue"]
    missing = [c for c in required if col(c) is None]
    if missing:
        st.warning("Fehlende Spalten im Upload: " + ", ".join(missing))
        return None
    df = df.rename(columns={cols[k]: k for k in cols})
    df["date"] = df["date"].apply(lambda x: parse_date(str(x)))
    df["orders"] = pd.to_numeric(df["orders"], errors="coerce").fillna(0).astype(int)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    if "cost" in df.columns:
        df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0.0)
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

def ensure_profit_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "cost" not in df.columns:
        # einfache Annahme: 65% des Umsatzes als Kosten
        df["cost"] = df["revenue"] * 0.65
    df["margin"] = df["revenue"] - df["cost"]
    df["margin_pct"] = np.where(df["revenue"] > 0, df["margin"] / df["revenue"] * 100, 0.0)
    return df

def filter_df(df: pd.DataFrame, d_from: pd.Timestamp, d_to: pd.Timestamp, regions, categories, channels) -> pd.DataFrame:
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
    margin_curr = df_curr.get("margin", pd.Series(dtype=float)).sum() if not df_curr.empty else 0.0
    margin_prev = df_prev.get("margin", pd.Series(dtype=float)).sum() if not df_prev.empty else 0.0

    def delta_pct(curr, prev):
        if prev == 0:
            return 0.0 if curr == 0 else 100.0
        return (curr - prev) / prev * 100.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Umsatz", fmt_chf(revenue_curr), f"{delta_pct(revenue_curr, revenue_prev):+.1f}%")
    c2.metric("Bestellungen", f"{orders_curr:,}".replace(",", "'"), f"{delta_pct(orders_curr, orders_prev):+.1f}%")
    c3.metric("Ø Bestellung", fmt_chf(aov_curr), f"{delta_pct(aov_curr, aov_prev):+.1f}%")
    c4.metric("Marge", fmt_chf(margin_curr), f"{delta_pct(margin_curr, margin_prev):+.1f}%")
    if not df_curr.empty:
        c5.metric("Periode", f"{df_curr['date'].min().date()} – {df_curr['date'].max().date()}")
    else:
        c5.metric("Periode", "keine Daten")

# ------------------------------ App-Logik -------------------------------------
st.sidebar.header("Datenquelle")
uploaded = st.sidebar.file_uploader("CSV oder Excel hochladen", type=["csv","xlsx"])

if uploaded and uploaded.size > 0:
    df_raw = try_read_csv(uploaded.getvalue())
    source = "Upload"
else:
    df_raw = load_example_data()
    source = "Beispieldaten"

if df_raw is None or df_raw.empty:
    st.error("Keine gültigen Daten geladen")
    st.stop()

# Profitspalten sicherstellen
df_raw = ensure_profit_columns(df_raw)

st.caption(f"Quelle: {source}")

# ------------------------------ Filter (robust) -------------------------------
st.sidebar.header("Filter")
min_date, max_date = df_raw["date"].min(), df_raw["date"].max()
suggested_from = (max_date - relativedelta(months=5)).replace(day=1)
default_from = max(min_date, suggested_from)
default_to = max_date
if default_from > default_to:
    default_from = default_to

date_input_value = st.sidebar.date_input(
    "Zeitraum",
    value=(default_from.date(), default_to.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

# Vereinheitlichen
if isinstance(date_input_value, (list, tuple)):
    d_from = pd.to_datetime(date_input_value[0])
    d_to   = pd.to_datetime(date_input_value[-1])
else:
    d_from = pd.to_datetime(date_input_value)
    d_to   = pd.to_datetime(date_input_value)

regions_all = sorted(df_raw["region"].dropna().astype(str).unique().tolist()) if "region" in df_raw.columns else []
categories_all = sorted(df_raw["category"].dropna().astype(str).unique().tolist()) if "category" in df_raw.columns else []
channels_all = sorted(df_raw["channel"].dropna().astype(str).unique().tolist()) if "channel" in df_raw.columns else []

regions = st.sidebar.multiselect("Region", options=regions_all, default=regions_all) if regions_all else []
categories = st.sidebar.multiselect("Kategorie", options=categories_all, default=categories_all) if categories_all else []
channels = st.sidebar.multiselect("Kanal", options=channels_all, default=channels_all) if channels_all else []

if not regions and regions_all:
    regions = regions_all
if not categories and categories_all:
    categories = categories_all
if not channels and channels_all:
    channels = channels_all

# ------------------------------ Daten / Vorperiode ----------------------------
df_curr = filter_df(df_raw, d_from, d_to, regions, categories, channels)

period_len = max(1, (d_to - d_from).days + 1)
prev_to = d_from - timedelta(days=1)
prev_from = prev_to - timedelta(days=period_len - 1)
df_prev = filter_df(df_raw, prev_from, prev_to, regions, categories, channels)

st.title("KPI-Dashboard")

# KPI-Block
kpi_block(df_curr, df_prev)

# ------------------------------ Tabs -----------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Zeit", "Kategorien", "Kanäle", "Tabelle"])

with tab1:
    if df_curr.empty:
        st.info("Keine Daten im gewählten Zeitraum/Filter – bitte Filter anpassen")
    else:
        # Granularität
        gran = st.radio("Granularität", ["Monat", "Woche"], horizontal=True)
        freq = "MS" if gran == "Monat" else "W-MON"
        ts = df_curr.groupby(pd.Grouper(key="date", freq=freq), as_index=False).agg({"revenue":"sum","orders":"sum","margin":"sum"})
        if ts.empty or "date" not in ts.columns or "revenue" not in ts.columns:
            st.info("Zeitreihenansicht nicht verfügbar für die aktuelle Auswahl")
        else:
            fig = px.line(ts, x="date", y="revenue", markers=True, title=f"Umsatz über Zeit – {gran}")
            fig.update_layout(yaxis_title="Umsatz (CHF)", xaxis_title="Datum")
            st.plotly_chart(fig, use_container_width=True)

        # MoM
        ts_m = df_curr.groupby(pd.Grouper(key="date", freq="MS"), as_index=False).agg({"revenue":"sum"})
        ts_m["revenue_lag1m"] = ts_m["revenue"].shift(1)
        ts_m["mom_growth_pct"] = np.where(ts_m["revenue_lag1m"]>0, (ts_m["revenue"]-ts_m["revenue_lag1m"])/ts_m["revenue_lag1m"]*100, 0)

        # YoY
        ts_m["date_last_year"] = ts_m["date"] - pd.DateOffset(years=1)
        yoy_lookup = ts_m[["date","revenue"]].rename(columns={"date":"date_last_year","revenue":"revenue_last_year"})
        ts_yoy = ts_m.merge(yoy_lookup, on="date_last_year", how="left")
        ts_yoy["yoy_growth_pct"] = np.where(ts_yoy["revenue_last_year"]>0,
                                            (ts_yoy["revenue"]-ts_yoy["revenue_last_year"])/ts_yoy["revenue_last_year"]*100, 0)

        with st.expander("MoM- und YoY-Wachstum (Monatsbasis)"):
            st.dataframe(ts_yoy[["date","revenue","mom_growth_pct","revenue_last_year","yoy_growth_pct"]])

        col1, col2 = st.columns(2)
        with col1:
            by_reg = df_curr.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
            if by_reg.empty:
                st.info("Keine Daten für Regionen")
            else:
                st.subheader("Umsatz nach Region")
                st.plotly_chart(px.bar(by_reg, x="region", y="revenue"), use_container_width=True)
        with col2:
            by_cat = df_curr.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
            if by_cat.empty:
                st.info("Keine Daten für Kategorien")
            else:
                st.subheader("Umsatz nach Kategorie")
                st.plotly_chart(px.bar(by_cat, x="category", y="revenue"), use_container_width=True)

with tab2:
    if df_curr.empty:
        st.info("Keine Daten für Kategorien-Drilldown")
    else:
        by_cat_month = df_curr.groupby([pd.Grouper(key="date", freq="MS"), "category"], as_index=False)["revenue"].sum()
        if by_cat_month.empty:
            st.info("Kein Kategorienverlauf für die Auswahl")
        else:
            st.plotly_chart(px.area(by_cat_month, x="date", y="revenue", color="category",
                                    title="Umsatz je Kategorie über Zeit"), use_container_width=True)
        if "product" in df_curr.columns and not df_curr["product"].isna().all():
            top_products = df_curr.groupby("product", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False).head(10)
            if not top_products.empty:
                st.subheader("Top 10 Produkte nach Umsatz")
                st.plotly_chart(px.bar(top_products, x="product", y="revenue"), use_container_width=True)

with tab3:
    if df_curr.empty:
        st.info("Keine Daten für Kanäle")
    else:
        by_channel = df_curr.groupby("channel", as_index=False).agg({"orders":"sum","revenue":"sum","margin":"sum"})
        if by_channel.empty:
            st.info("Keine Kanalumsätze für die Auswahl")
        else:
            st.plotly_chart(px.pie(by_channel, names="channel", values="revenue", hole=0.5,
                                   title="Umsatzanteile nach Kanal"), use_container_width=True)

        conv = df_curr.groupby("channel", as_index=False).agg({"orders":"sum"})
        if not conv.empty:
            orders_sum = conv["orders"].replace(0, np.nan)
            conv["avg_order_value"] = df_curr.groupby("channel")["revenue"].sum().values / orders_sum
            conv = conv.fillna(0)
            st.subheader("Bestellungen & Ø Bestellwert nach Kanal")
            st.dataframe(conv.rename(columns={"orders":"Bestellungen","avg_order_value":"Ø Bestellwert"}))

with tab4:
    if df_curr.empty:
        st.info("Keine Daten für Tabelle")
    else:
        # Excel-Export mit mehreren Sheets
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df_curr.to_excel(writer, sheet_name="Daten", index=False)
            df_curr.groupby("region", as_index=False)["revenue"].sum().to_excel(writer, sheet_name="Regionen", index=False)
            df_curr.groupby("category", as_index=False)["revenue"].sum().to_excel(writer, sheet_name="Kategorien", index=False)
            df_curr.groupby("channel", as_index=False)["revenue"].sum().to_excel(writer, sheet_name="Kanäle", index=False)
        st.download_button("Export als Excel", data=buf.getvalue(),
                           file_name="export.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.download_button("Gefilterte Daten als CSV",
                           data=df_curr.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_data.csv",
                           mime="text/csv")
        st.dataframe(df_curr)

st.caption("Neue Features: Marge, MoM/YoY, Wochen/Monat-Umschalter, Excel-Export mit Tabs")
