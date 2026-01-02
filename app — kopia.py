import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import re
import numpy as np
import plotly.express as px
import plotly.io as pio
from io import BytesIO
import tempfile
import os
import copy
import plotly.colors as pc

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm
# =====================
# Ustawienia domyślne Plotly
# =====================
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = px.colors.qualitative.Set2

import base64



# =====================
# Konfiguracja strony
# =====================
st.set_page_config(page_title="Analiza danych", layout="wide")
# --- wczytanie obrazka i zamiana na base64 --- logo w prawym górnym rogu
def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = img_to_base64("logo.png")  # ← ścieżka do pliku PNG

st.markdown(
    f"""
    <style>
    .app-logo {{
        position: fixed;
        top: 60px;
        right: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        z-index: 99999;
        background: white;
        padding: 6px 10px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        font-family: "Segoe UI", sans-serif;
    }}

    .app-logo img {{
        height: 28px;
        width: auto;
    }}

    .app-logo span {{
        font-size: 18px;
        font-weight: 700;
        color: #111;
        white-space: nowrap;
    }}
    </style>

    <div class="app-logo">
        <img src="data:image/png;base64,{logo_base64}">
        <span>by: Bart</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("📊 Analiza danych")

# =====================
# Stałe
# =====================
MONTHS_PL = {
    "sty": 1, "lut": 2, "mar": 3, "kwi": 4, "maj": 5, "cze": 6,
    "lip": 7, "sie": 8, "wrz": 9, "paź": 10, "lis": 11, "gru": 12
}

STANDARD_COLUMNS = {
    "wiek": "age",
    "lat": "age",
    "years_of_experience": "experience_years",
    "doświadczenie": "experience_years",
    "płeć": "gender",
    "gender": "gender",
    "edu_level": "edu_level",
    "edukacja": "edu_level",
    "industry": "industry",
    "branża": "industry",
    "class": "class"
}

# =====================
# Funkcje pomocnicze
# =====================
def parse_experience(value):
    today = dt.date.today()
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    if value == "0-2":
        return 1
    if value.startswith(">="):
        return int(value.replace(">=", ""))
    match = re.match(r"(\d{2})?\.*([a-ząćęłńóśżź]{3})", value)
    if match:
        year_str, month_str = match.groups()
        month = MONTHS_PL.get(month_str)
        if not month or not year_str:
            return None
        year = int(year_str)
        year += 2000 if year < 30 else 1900
        start_date = dt.date(year, month, 1)
        return round((today - start_date).days / 365.25, 1)
    return None


def load_and_map_file(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file, sep=None, engine="python")
    elif file.name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file)
    else:
        st.error(f"Nieobsługiwany format pliku: {file.name}")
        return None

    df.columns = [str(c).strip().lower() for c in df.columns]

    mapped_cols = {}
    for col in df.columns:
        for key, std_name in STANDARD_COLUMNS.items():
            if key in col:
                mapped_cols[col] = std_name
                break

    return df.rename(columns=mapped_cols)


@st.cache_data
def preprocess_data(df):
    df = df.copy()

    if "age" in df.columns and df["age"].notna().any():
        age_clean = (
            df["age"]
            .astype(str)
            .str.replace("<", "0-", regex=False)
            .str.replace("+", "-100", regex=False)
        )
        age_split = age_clean.str.split("-", expand=True).apply(pd.to_numeric, errors="coerce")
        df["age"] = age_split.mean(axis=1)

    if "experience_years" in df.columns and df["experience_years"].notna().any():
        df["experience_years"] = df["experience_years"].apply(parse_experience)

    # for col in ["gender", "edu_level", "industry", "class"]:
    #     if col not in df.columns:
    #         df[col] = None

    return df

# =====================
# Upload plików
# =====================
st.sidebar.subheader("📁 Wczytaj plik(y)")
uploaded_files = st.sidebar.file_uploader(
    "Wybierz pliki CSV lub Excel",
    type=["csv", "xls", "xlsx"],
    accept_multiple_files=True
)

df = None
if uploaded_files:
    dfs = []
    for file in uploaded_files:
        part = load_and_map_file(file)
        if part is not None:
            dfs.append(part)
    if dfs:
        df = preprocess_data(pd.concat(dfs, ignore_index=True))
        st.success(f"Wczytano {len(dfs)} plik(y), łącznie {df.shape[0]} wierszy")
else:
    st.markdown(
        "<div style='text-align:center; padding:40px; font-size:20px;'>"
        "💾 Czekam na załadowanie pliku/ów.<br>Życzę udanej analizy!"
        "</div>",
        unsafe_allow_html=True
    )

if df is None or df.empty:
    st.stop()

# =====================
# Sidebar – filtry
# =====================
st.sidebar.header("🔎 Filtry")
filtered_df = df.copy()
filters = {}

for col in df.columns:
    col_data = df[col]
    if pd.api.types.is_numeric_dtype(col_data):
        valid = col_data.dropna()
        if valid.nunique() < 2:
            continue
        lo, hi = float(valid.min()), float(valid.max())
        selected = st.sidebar.slider(col, lo, hi, (lo, hi))
        filters[col] = ("num", selected, lo, hi)
    else:
        unique_vals = col_data.dropna().unique().tolist()
        if 2 <= len(unique_vals) <= 30:
            selected = st.sidebar.multiselect(col, sorted(unique_vals), unique_vals)
            filters[col] = ("cat", selected, unique_vals)

for col, cfg in filters.items():
    if cfg[0] == "num":
        _, (lo, hi), _, _ = cfg
        filtered_df = filtered_df[
            filtered_df[col].between(lo, hi) | filtered_df[col].isna()
        ]
    else:
        _, selected, _ = cfg
        filtered_df = filtered_df[
            filtered_df[col].isin(selected) | filtered_df[col].isna()
        ]

st.write(f"✅ Liczba wierszy po filtrach: {filtered_df.shape[0]}")

# =====================
# Sidebar – wizualizacje
# =====================
st.sidebar.header("📊 Wizualizacje")
show_random = st.sidebar.checkbox("Losowe wiersze", True)
show_hist = st.sidebar.checkbox("Histogram", True)
show_scatter = st.sidebar.checkbox("Wykres punktowy", True)
show_corr = st.sidebar.checkbox("Heatmapa korelacji", True)
show_box = st.sidebar.checkbox("Boxplot", False)
show_violin = st.sidebar.checkbox("Violin plot", False)
show_count = st.sidebar.checkbox("Countplot", False)
show_matrix = st.sidebar.checkbox("Scatter matrix", False)

# =====================
# Losowe wiersze
# =====================
if show_random and not filtered_df.empty:
    st.subheader("🔀 Podgląd losowych wierszy")
    n_rows = st.slider(
        "Liczba wierszy",
        5,
        min(50, len(filtered_df)),
        value=min(10, len(filtered_df))
    )
    st.dataframe(filtered_df.sample(n=n_rows, random_state=42), hide_index=True)

# =====================
# Przygotowanie kolumn
# =====================
numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
cat_cols = filtered_df.select_dtypes(exclude=np.number).columns.tolist()

x_col = st.sidebar.selectbox("Oś X", numeric_cols)
y_col = st.sidebar.selectbox("Oś Y", numeric_cols, index=min(1, len(numeric_cols)-1))
color_col = st.sidebar.selectbox("Kolor (opcjonalnie)", [None] + cat_cols)

# =====================
# Wizualizacje + komentarze
# =====================
plots_for_pdf = []
comments = []

def get_comment_box(title):
    add = st.checkbox(f"Dodaj komentarz do: {title}")
    return st.text_area(f"Analiza: {title}") if add else ""

if show_hist and numeric_cols:
    st.subheader(f"📊 Histogram: {x_col}")
    fig = px.histogram(filtered_df, x=x_col, nbins=20)
    st.plotly_chart(fig, use_container_width=True)
    plots_for_pdf.append(fig)
    comments.append(get_comment_box(f"Histogram: {x_col}"))

if show_scatter and numeric_cols:
    st.subheader(f"📈 Wykres punktowy: {x_col} vs {y_col}")
    fig = px.scatter(filtered_df, x=x_col, y=y_col, color=color_col)
    st.plotly_chart(fig, use_container_width=True)
    plots_for_pdf.append(fig)
    comments.append(get_comment_box(f"Scatter: {x_col} vs {y_col}"))

if show_corr and len(numeric_cols) >= 2:
    st.subheader("🔥 Heatmapa korelacji")
    corr = filtered_df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    plots_for_pdf.append(fig)
    comments.append(get_comment_box("Heatmapa korelacji"))

if show_box and numeric_cols:
    st.subheader("📦 Boxplot")
    col = st.selectbox("Kolumna (Boxplot)", numeric_cols)
    fig = px.box(filtered_df, y=col, points="outliers")
    st.plotly_chart(fig, use_container_width=True)
    plots_for_pdf.append(fig)
    comments.append(get_comment_box(f"Boxplot: {col}"))

if show_violin and numeric_cols and cat_cols:
    st.subheader("🎻 Violin plot")
    num_col = st.selectbox("Kolumna numeryczna (Violin)", numeric_cols)
    cat_col = st.selectbox("Kolumna kategoryczna (Violin)", cat_cols)
    fig = px.violin(filtered_df, x=cat_col, y=num_col, box=True, points="outliers")
    st.plotly_chart(fig, use_container_width=True)
    plots_for_pdf.append(fig)
    comments.append(get_comment_box(f"Violin: {num_col} wg {cat_col}"))

if show_count and cat_cols:
    st.subheader("🧮 Countplot")
    col = st.selectbox("Kolumna kategoryczna (Countplot)", cat_cols)
    counts = filtered_df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(counts, x=col, y="count", text="count")
    st.plotly_chart(fig, use_container_width=True)
    plots_for_pdf.append(fig)
    comments.append(get_comment_box(f"Countplot: {col}"))

if show_matrix and len(numeric_cols) >= 2:
    st.subheader("🔬 Scatter Matrix")
    selected = st.multiselect("Kolumny", numeric_cols, default=numeric_cols[:4])
    if len(selected) >= 2:
        fig = px.scatter_matrix(filtered_df, dimensions=selected)
        st.plotly_chart(fig, use_container_width=True)
        plots_for_pdf.append(fig)
        comments.append(get_comment_box("Scatter Matrix"))

# =====================
# Eksport CSV / Excel
# =====================
st.subheader("⬇️ Eksport danych")

csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Pobierz CSV", csv, "dane_po_filtrach.csv", "text/csv")

excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    filtered_df.to_excel(writer, index=False, sheet_name="Dane")
st.download_button(
    "📥 Pobierz Excel",
    excel_buffer.getvalue(),
    "dane_po_filtrach.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
# dodano dla kolorów w pdf= ale też nie działa
# def prepare_fig_for_pdf(fig):
#     fig_copy = copy.deepcopy(fig)
#     fig_copy.update_layout(
#         template="plotly_white",
#         paper_bgcolor="white",
#         plot_bgcolor="white",
#         font=dict(color="black"),
#         colorway=px.colors.qualitative.Set2
#     )
#     return fig_copy
# ====================
# Przygotowanie wykresów do HTML
# ====================
# def prepare_fig_for_html(fig):
#     fig = copy.deepcopy(fig)
#     palette = px.colors.qualitative.Set2

#     fig.update_layout(
#         template="plotly_white",
#         paper_bgcolor="white",
#         plot_bgcolor="white",
#         font=dict(color="black"),
#         showlegend=True
#     )

#     for i, trace in enumerate(fig.data):
#         color = palette[i % len(palette)]

#         # ===== SCATTER / LINE =====
#         if trace.type == "scatter":
#             if trace.mode and "markers" in trace.mode:
#                 trace.marker.color = color
#                 trace.marker.opacity = 0.8
#             if trace.mode and "lines" in trace.mode:
#                 trace.line.color = color

#         # ===== HISTOGRAM =====
#         elif trace.type == "histogram":
#             trace.marker.color = color
#             trace.marker.line.color = color
#             trace.marker.opacity = 0.8

#         # ===== BOX / VIOLIN =====
#         elif trace.type in ("box", "violin"):
#             trace.fillcolor = color
#             trace.line.color = color

#         # ===== HEATMAP / IMSHOW =====
#         elif trace.type in ("heatmap", "imshow"):
#             trace.colorscale = "RdBu"
#             trace.showscale = True

#     return fig

# def prepare_fig_for_html(fig):
#     fig = copy.deepcopy(fig)

#     palette = px.colors.qualitative.Set2

#     fig.update_layout(
#         template="plotly_white",
#         paper_bgcolor="white",
#         plot_bgcolor="white",
#         font=dict(color="black"),
#         showlegend=True
#     )

#     for i, trace in enumerate(fig.data):
#         color = palette[i % len(palette)]

#         # scatter, bar, histogram, box, violin
#         if hasattr(trace, "marker") and trace.marker is not None:
#             trace.marker.color = color
#             trace.marker.opacity = 0.8

#         # linie
#         if hasattr(trace, "line") and trace.line is not None:
#             trace.line.color = color

#         # heatmap / imshow
#         if trace.type in ("heatmap", "imshow"):
#             trace.colorscale = "RdBu"
#             trace.showscale = True

#     return fig



# ====================
# Przygotowanie wykresów do PDF
# ====================

def prepare_fig_for_pdf(fig):
    fig_copy = copy.deepcopy(fig)

    # Jawna paleta (kolorowa, stabilna)
    palette = pc.qualitative.Plotly

    # Ustaw tło
    fig_copy.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        showlegend=True
    )

    # === WYMUSZENIE KOLORÓW NA KAŻDYM TRACE ===
    for i, trace in enumerate(fig_copy.data):
        color = palette[i % len(palette)]

        if hasattr(trace, "marker"):
            trace.marker.color = color

        if hasattr(trace, "line"):
            trace.line.color = color

        if trace.type in ["heatmap", "imshow"]:
            trace.colorscale = "RdBu"

    return fig_copy
# ====================
# HTML – GENEROWANY AUTOMATYCZNIE
# =====================
def generate_html_report(df, figs, comments, applied_filters):
    import plotly.io as pio
    from datetime import datetime

    sections = []
    toc = []

    # === Nagłówek dokumentu ===
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Raport danych</title>
        # === 
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ margin-bottom: 0; }}
            h2 {{ margin-top: 40px; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
            ul {{ line-height: 1.6; }}
            .comment {{ background: #f7f7f7; padding: 10px; margin-top: 10px; }}
            .meta {{ color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>Raport danych po filtrach</h1>
        <div class="meta">Wygenerowano: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    """

    # === Filtry ===
    toc.append("<li><a href='#filters'>Zastosowane filtry</a></li>")
    filters_html = "<h2 id='filters'>Zastosowane filtry</h2><ul>"
    for col, cfg in applied_filters.items():
        if cfg[0] == "num":
            _, (lo, hi), _, _ = cfg
            filters_html += f"<li><b>{col}</b>: {lo} – {hi}</li>"
        else:
            _, sel, _ = cfg
            filters_html += f"<li><b>{col}</b>: {', '.join(map(str, sel))}</li>"
    filters_html += "</ul>"
    sections.append(filters_html)

    # === Podgląd danych ===
    toc.append("<li><a href='#table'>Podgląd danych</a></li>")
    sections.append(
        "<h2 id='table'>Podgląd danych (pierwsze 10 wierszy)</h2>" +
        df.head(10).to_html(index=False)
    )

    # === Wykresy ===
    for i, (fig, comment) in enumerate(zip(figs, comments), start=1):
        section_id = f"plot-{i}"
        title = fig.layout.title.text if fig.layout.title.text else f"Wykres {i}"

        toc.append(f"<li><a href='#{section_id}'>{title}</a></li>")

        fig_html = pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs="cdn",
            config={
                "responsive": True,
                "displayModeBar": True,
                "scrollZoom": True
            }
)





        section = f"<h2 id='{section_id}'>{title}</h2>{fig_html}"

        if comment:
            section += f"<div class='comment'><b>Komentarz:</b><br>{comment}</div>"

        sections.append(section)

    # === Spis treści ===
    html += "<h2>Spis treści</h2><ul>" + "".join(toc) + "</ul>"

    # === Treść ===
    html += "".join(sections)

    html += "</body></html>"

    return html.encode("utf-8")


# =====================
# PDF – GENEROWANY TYLKO NA KLIK
# =====================
def generate_pdf(df, figs, comments, applied_filters):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    temp_imgs = []

    story.append(Paragraph("Raport danych po filtrach", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Zastosowane filtry:", styles["Heading2"]))
    for col, cfg in applied_filters.items():
        if cfg[0] == "num":
            _, (lo, hi), _, _ = cfg
            story.append(Paragraph(f"{col}: {lo} – {hi}", styles["Normal"]))
        else:
            _, sel, _ = cfg
            story.append(Paragraph(f"{col}: {', '.join(map(str, sel))}", styles["Normal"]))

    story.append(PageBreak())

    table_df = df.head(10).astype(str)
    table = Table([table_df.columns.tolist()] + table_df.values.tolist())
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("FONTSIZE", (0,0), (-1,-1), 8)
    ]))
    story.append(table)
    story.append(PageBreak())

    for fig, comment in zip(figs, comments):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        prepare_fig_for_pdf(fig).write_image(tmp.name, scale=2) # zmieniona linia        
        tmp.close()
        temp_imgs.append(tmp.name)

        story.append(Image(tmp.name, width=16*cm, height=9*cm))
        if comment:
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"Komentarz: {comment}", styles["Italic"]))
        story.append(PageBreak())

    doc.build(story)

    for p in temp_imgs:
        try:
            os.remove(p)
        except Exception:
            pass

    buffer.seek(0)
    return buffer


st.subheader("⬇️ Eksport PDF")

if st.button("📄 Generuj raport PDF"):
    with st.spinner("Generowanie PDF..."):
        pdf_buffer = generate_pdf(filtered_df, plots_for_pdf, comments, filters)

    st.download_button(
        "📥 Pobierz PDF",
        pdf_buffer,
        "raport_danych.pdf",
        "application/pdf"
    )
st.subheader("⬇️ Eksport HTML")
# =====================
# HTML – GENEROWANY TYLKO NA KLIK
# =====================
if st.button("📄 Generuj raport HTML"):
    with st.spinner("Generowanie raportu HTML..."):
        html_bytes = generate_html_report(
            filtered_df,
            plots_for_pdf,
            comments,
            filters
        )

    st.download_button(
        "📥 Pobierz HTML",
        html_bytes,
        "raport_danych.html",
        "text/html"
    )
