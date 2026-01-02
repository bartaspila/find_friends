# zmieniono lub dodano: nazwa strony, tytuł, logo, wygenerowano nowy model, 
# nowy plik z danymi (większa grupa), uporządkowanie wieku w sidebarze, 
# dodano wizualizacje, 

import streamlit as st
st.set_page_config(page_title="Wyszukaj znajomych", layout="wide")

# ustawienie dark mode
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  # domyślnie dark

import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
import base64
import json

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'

DATA = 'welcome_survey_simple_v2.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'


@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters

with st.sidebar:
    st.sidebar.header("Ustawienie trybu wyświetlania")
    st.session_state.dark_mode = st.sidebar.checkbox("Dark Mode", st.session_state.dark_mode)
    if st.session_state.dark_mode:
        bg_color = "#1E1E2F"
        secondary_bg = "#2C2C3E"
        text_color = "#E5E5E5"
        sidebar_text = "#E5E5E5"
        metric_text_color = "#E5E5E5"
    else:
        bg_color = "#FFFFFF"
        secondary_bg = "#F0F0F0"   # lekko szary sidebar
        text_color = "#111111"
        sidebar_text = "#111111"
        metric_text_color = "#111111"

#     # --- CSS ---
#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-color: {bg_color};
#         color: {text_color};
#     }}
#     .stSidebar {{
#         background-color: {secondary_bg};
#         color: {sidebar_text};
#     }}
#     .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar label, .stSidebar .css-1v3fvcr {{
#         color: {sidebar_text} !important;
#     }}
   

#     </style>
#     """,
#     unsafe_allow_html=True
# )
#     #===== METRIC FINAL FIX (CSS VARIABLE) =====

#     div[data-testid="stMetric"] {
#         --metric-value-color: {metric_text_color} !important;
#         --metric-label-color: {metric_text_color} !important;
# }
    # st.markdown(
    #     f"""
    #     <style>
    #     .stApp {{
    #         background-color: {bg_color};
    #         color: {text_color};
    #     }}

    #     .stSidebar {{
    #         background-color: {secondary_bg};
    #         color: {sidebar_text};
    #     }}

    #     .stSidebar h1,
    #     .stSidebar h2,
    #     .stSidebar h3,
    #     .stSidebar label {{
    #         color: {sidebar_text} !important;
    #     }}

    #     /* ===== METRIC FINAL FIX ===== */
    #     div[data-testid="stMetric"] {{
    #         --metric-value-color: {metric_text_color} !important;
    #         --metric-label-color: {metric_text_color} !important;
    #     }}
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )


# opcje w sidebarze
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Koty i Psy','Inne'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Kobieta','Mężczyzna'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]


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
st.title("🤝 Wyszukaj znajomych – analiza danych")
st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}

        .stSidebar {{
            background-color: {secondary_bg};
            color: {sidebar_text};
        }}

        .stSidebar h1,
        .stSidebar h2,
        .stSidebar h3,
        .stSidebar label {{
            color: {sidebar_text} !important;
        }}

        /* ===== METRIC FINAL FIX ===== */
        div[data-testid="stMetric"] {{
            --metric-value-color: {metric_text_color} !important;
            --metric-label-color: {metric_text_color} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
st.header(f"Najbliżej Ci do grupy: {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

st.header("Osoby z grupy")
fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

# Sekcja: Ty vs Twoja grupa (porównanie)
st.header("👤 Ty na tle swojej grupy")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Twoje dane")
    st.dataframe(person_df, use_container_width=True)

with col2:
    st.subheader("Najczęstsze cechy w grupie")
    summary = same_cluster_df.drop(columns=["Cluster"]).mode().iloc[0]
    st.dataframe(summary.to_frame("Najczęściej"), use_container_width=True)

# Wykres kołowy – struktura grupy (%)
st.header("📊 Struktura grupy (udziały %)")

col1, col2 = st.columns(2)

with col1:
    fig = px.pie(
        same_cluster_df,
        names="gender",
        title="Płeć w grupie",
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.pie(
        same_cluster_df,
        names="edu_level",
        title="Wykształcenie w grupie",
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

# Heatmapa preferencji (🔥)
st.header("🔥 Heatmapa zależności (wybierz osie)")

col1, col2 = st.columns(2)

categorical_columns = {
    "Ulubione zwierzęta": "fav_animals",
    "Ulubione miejsce": "fav_place",
    "Wykształcenie": "edu_level",
    "Płeć": "gender",
    "Wiek": "age",
}

with col1:
    x_label = st.selectbox(
        "Oś X",
        list(categorical_columns.keys()),
        index=0
    )

with col2:
    y_label = st.selectbox(
        "Oś Y",
        list(categorical_columns.keys()),
        index=1
    )

x_col = categorical_columns[x_label]
y_col = categorical_columns[y_label]

if x_col == y_col:
    st.warning("⚠️ Wybierz różne zmienne na osie X i Y")
else:
    heatmap_df = (
        same_cluster_df
        .groupby([x_col, y_col])
        .size()
        .reset_index(name="count")
    )

    fig = px.density_heatmap(
        heatmap_df,
        x=x_col,
        y=y_col,
        z="count",
        color_continuous_scale="Blues",
        title=f"{x_label} vs {y_label}"
    )

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label
    )

    st.plotly_chart(fig, use_container_width=True)


# Radar – „profil typowej osoby w grupie”
st.header("🧭 Profil typowej osoby z grupy")

profile_counts = {
    "Nad wodą": (same_cluster_df["fav_place"] == "Nad wodą").mean(),
    "Las": (same_cluster_df["fav_place"] == "W lesie").mean(),
    "Góry": (same_cluster_df["fav_place"] == "W górach").mean(),
    "Psy": same_cluster_df["fav_animals"].isin(["Psy", "Koty i Psy"]).mean(),
    "Koty": same_cluster_df["fav_animals"].isin(["Koty", "Koty i Psy"]).mean(),
}

radar_df = pd.DataFrame(
    dict(
        r=list(profile_counts.values()),
        theta=list(profile_counts.keys())
    )
)

fig = px.line_polar(
    radar_df,
    r="r",
    theta="theta",
    line_close=True,
    title="Profil zainteresowań grupy"
)

fig.update_traces(fill="toself")
st.plotly_chart(fig, use_container_width=True)

# Ranking TOP 5 cech w grupie
st.header("🏆 TOP cechy w Twojej grupie")

fav_place_top = same_cluster_df["fav_place"].value_counts().head(5)
fig = px.bar(
    fav_place_top,
    x=fav_place_top.values,
    y=fav_place_top.index,
    orientation="h",
    title="Najpopularniejsze miejsca",
    labels={"x": "Liczba osób", "y": "Miejsce"}
)

st.plotly_chart(fig, use_container_width=True)
