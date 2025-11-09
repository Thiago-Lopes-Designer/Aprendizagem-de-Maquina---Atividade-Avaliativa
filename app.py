# app.py — Dashboard Streamlit (tema azul escuro)
import os
import joblib
import pandas as pd
import streamlit as st

# ----------------- Config -----------------
st.set_page_config(
    page_title="Fake News Detector - Reprodução Científica",
    page_icon="🧠",
    layout="wide",
)

# ----------------- Estilo -----------------
st.markdown(
    """
    <style>
    :root { --blue:#00B4FF; }
    html, body, [class*="css"]  { background-color:#0E1117; color:#FAFAFA; }
    h1,h2,h3 { color: var(--blue) !important; }
    section.main > div { padding-top: 1rem; }
    .block-container { padding-top: 1.2rem; }
    .dataframe tbody tr th, .dataframe thead th { color:#FAFAFA !important; }
    .stButton>button { background:#122033; border:1px solid #1f3a5b; color:#fff; }
    .stButton>button:hover { background:#16304d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Helpers -----------------
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Carrega o pipeline treinado (TF-IDF + SVM) salvo em modelo.pkl
    """
    candidate_paths = [
        "modelo.pkl",
        "./modelo.pkl",
        "/content/modelo.pkl",              # caso tenha feito upload via Colab
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return joblib.load(p)
    raise FileNotFoundError(
        "Arquivo 'modelo.pkl' não encontrado no diretório do app. "
        "Envie o modelo para a raiz do repositório."
    )

def classify_text(text: str, model):
    """
    Retorna (pred_int, label_str). Captura erro de TF-IDF não-ajustado.
    """
    try:
        pred = int(model.predict([text])[0])
        label = "🟩 NOTÍCIA REAL" if pred == 1 else "🟥 FAKE NEWS"
        return pred, label, None
    except Exception as e:
        return None, None, e

# ----------------- UI -----------------
st.title("🧠 Detecção de Fake News em Português")
st.subheader("Reprodução científica baseada em Fagundes et al. (SBC, 2024)")

st.header("📊 Resultados dos Modelos")
dados = {
    "Modelo": ["Baseline (TF-IDF)", "POS + TF-IDF"],
    "Acurácia": [0.98, 0.98],
    "Precisão": [0.98, 0.98],
    "Recall": [0.98, 0.98],
    "F1-Score": [0.98, 0.98],
}
st.table(pd.DataFrame(dados))

st.header("📊 Visões lado a lado")
col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("📈 F1-Score (comparação)")
    st.image("grafico_comparativo.png", use_column_width=True)
with col2:
    st.subheader("🧮 Matriz de Confusão")
    st.image("matriz_confusao.png", use_column_width=True)

st.header("🗞️ Teste uma notícia")
texto = st.text_area(
    "Cole uma notícia em português para classificar:",
    height=180,
    placeholder="Ex.: 'Governo anuncia que...'"
)

col_a, col_b = st.columns([1, 3])

with col_a:
    if st.button("Classificar notícia", use_container_width=True):
        if texto.strip():
            try:
                modelo = load_model()
            except FileNotFoundError as e:
                st.error(str(e))
            else:
                pred, label, err = classify_text(texto, modelo)
                if err is None:
                    st.subheader(f"Resultado: {label}")
                else:
                    st.error(
                        "Falha ao classificar. Se aparecer algo como "
                        "`idf vector is not fitted`, garanta que **modelo.pkl** "
                        "é o *pipeline completo* (TF-IDF + SVM) salvo via `joblib.dump(pipeline, 'modelo.pkl')`."
                    )
        else:
            st.warning("Por favor, digite um texto antes de classificar.")

with col_b:
    st.info(
        "Este classificador usa um **pipeline TF-IDF + LinearSVM** treinado no corpus **Fake.Br** "
        "(3.600 reais / 3.600 falsas). Os resultados de reprodução indicaram **F1 ≈ 0.98** tanto no "
        "baseline quanto com **POS**, confirmando as conclusões do artigo (ganho marginal com sintaxe superficial)."
    )

st.markdown(
    """
    ---
    Projeto para **Aprendizagem de Máquina — UNIFESSPA (2025)**.  
    Reprodução científica baseada em *Fagundes, Roman & Digiampietri (2024), SBC*.
    """
)
