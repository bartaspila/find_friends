# =========================================================
# Wyszukaj znajomych – kompletna wersja z Dark / Light mode
# =========================================================

import streamlit as st
st.set_page_config(page_title="Wyszukaj znajomych", layout="wide")

import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
import base64
import json

# ------------------ SESSION STATE ------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ------------------ CONSTANTS ------------------
MODEL_NAME = "welcome_survey_clustering_pipeline_v2"
DATA = "welcome_survey_simple_v2.csv"
CLUSTER_NAMES_AND_DESCRIPTIONS = "welcome_survey_cluster_names_and_descriptions_v2.json"

# ------------------ CACHE ------------------
@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding="utf-8") as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants(model):
    all_df = pd.read_csv(DATA, sep=";")
    return predict_model(model, data=all_df)

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("Ustawienia wyglądu")

    st.session_state.dark_mode = st.checkbox(
        "🌙 Dark mode",
        st.session_state.dark_mode
    )

    if st.session_state.dark_mode:
        bg_color = "#1E1E2F"
        secondary_bg = "#2C2C3E"
        text_color = "#E5E5E5"
        sidebar_text = "#E5E5E5"
        metric_text_color = "#E5E5E5"
        plotly_template = "plotly_dark"
        logo_bg = "rgba(255,255,255,0.9)"
        logo_text = "#111"
    else:
        bg_color = "#FFFFFF"
        secondary_bg = "#F0F0F0"
        text_color = "#111111"
        sidebar_text = "#111111"
        metric_text_color = "#111111"
        plotly_template = "plotly_white"
        logo_bg = "#FFFFFF"
        logo_text = "#111"

    st.divider()
    st.header("Powiedz nam coś o sobie")

    age = st.selectbox(
        "Wiek",
        ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown']
    )
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox(
        "Ulubione zwierzęta",
        ['Brak ulubionych', 'Psy', 'Koty', 'Koty i Psy', 'Inne']
    )
    fav_place = st.selectbox(
        "Ulubione miejsce",
        ['Nad wodą', 'W lesie', 'W górach', 'Inne']
    )
    gender = st.radio("Płeć", ['Kobieta', 'Mężczyzna'])

    person_df = pd.DataFrame([{
        "age": age,
        "edu_level": edu_level,
        "fav_animals": fav_animals,
        "fav_place": fav_place,
        "gender": gender
    }])

# ================== GLOBAL CSS (TU JEST KLUCZ) ==================
# st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-color: {bg_color};
#         color: {text_color};
#     }}

#     section[data-testid="stSidebar"] {{
#         background-color: {secondary_bg};
#     }}

#     section[data-testid="stSidebar"] * {{
#         color: {sidebar_text} !important;
#     }}

#     div[data-testid="stMetric"] {{
#         --metric-value-color: {metric_text_color} !important;
#         --metric-label-color: {metric_text_color} !important;
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {secondary_bg};
    }}

    section[data-testid="stSidebar"] * {{
        color: {sidebar_text} !important;
    }}

    /* METRIC */
    div[data-testid="stMetric"] {{
        --metric-value-color: {metric_text_color} !important;
        --metric-label-color: {metric_text_color} !important;
    }}

    /* ✅ CHECKBOX – TO JEST KLUCZ */
    input[type="checkbox"] {{
        accent-color: #4da3ff;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ================== MODEL ==================
model = get_model()
all_df = get_all_participants(model)
cluster_names = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names[predicted_cluster_id]
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]

# ================== LOGO ==================
def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = img_to_base64("logo.png")

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
        background: {logo_bg};
        padding: 6px 10px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        font-family: "Segoe UI", sans-serif;
    }}

    .app-logo img {{
        height: 28px;
    }}

    .app-logo span {{
        font-size: 16px;
        font-weight: 700;
        color: {logo_text};
    }}
    </style>

    <div class="app-logo">
        <img src="data:image/png;base64,{logo_base64}">
        <span>by Bart</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ================== CONTENT ==================
st.title("🤝 Wyszukaj znajomych – analiza danych")

st.header(f"Najbliżej Ci do grupy: {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data["description"])

st.metric("Liczba twoich znajomych", len(same_cluster_df))

# ================== WYKRESY ==================
def show_hist(df, x, title, xlabel):
    fig = px.histogram(df, x=x, title=title, template=plotly_template)
    fig.update_layout(xaxis_title=xlabel, yaxis_title="Liczba osób")
    st.plotly_chart(fig, use_container_width=True)

show_hist(same_cluster_df.sort_values("age"), "age", "Rozkład wieku", "Wiek")
show_hist(same_cluster_df, "edu_level", "Rozkład wykształcenia", "Wykształcenie")
show_hist(same_cluster_df, "fav_animals", "Ulubione zwierzęta", "Zwierzęta")
show_hist(same_cluster_df, "fav_place", "Ulubione miejsca", "Miejsca")
show_hist(same_cluster_df, "gender", "Płeć", "Płeć")

# ================== PORÓWNANIE ==================
st.header("👤 Ty na tle swojej grupy")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Twoje dane")
    st.dataframe(person_df, use_container_width=True)

with c2:
    st.subheader("Najczęstsze cechy w grupie")
    summary = same_cluster_df.drop(columns=["Cluster"]).mode().iloc[0]
    st.dataframe(summary.to_frame("Najczęściej"), use_container_width=True)

# ================== PIE ==================
st.header("📊 Struktura grupy")

c1, c2 = st.columns(2)
with c1:
    fig = px.pie(
        same_cluster_df,
        names="gender",
        hole=0.4,
        title="Płeć",
        template=plotly_template
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.pie(
        same_cluster_df,
        names="edu_level",
        hole=0.4,
        title="Wykształcenie",
        template=plotly_template
    )
    st.plotly_chart(fig, use_container_width=True)
