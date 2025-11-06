# app.py — Dashboard Streamlit (tema azul escuro)
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Fake News Detector - Reprodução Científica",
    page_icon="🧠",
    layout="wide",
)

# ====== Estilo (tema escuro/azul) ======
st.markdown("""
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
""", unsafe_allow_html=True)

# ====== Cabeçalho ======
st.title("🧠 Detecção de Fake News em Português")
st.subheader("Reprodução científica baseada em Fagundes et al. (SBC, 2024)")

# ====== Seção 1: Tabela de métricas ======
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
    
# ====== Seção 4: Teste de notícia ======
st.header("🗞️ Teste uma notícia")
texto = st.text_area("Cole uma notícia em português para classificar:", height=180,
                     placeholder="Ex.: 'Governo anuncia que...'")

col_a, col_b = st.columns([1,3])
with col_a:
    if st.button("Classificar notícia"):
        if texto.strip():
            # Carrega o pipeline completo (TF-IDF + SVM)
            modelo = joblib.load("modelo.pkl")
            # Predição direta a partir do texto cru
            pred = int(modelo.predict([texto])[0])
            label = "🟩 NOTÍCIA REAL" if pred == 1 else "🟥 FAKE NEWS"
            st.subheader(f"Resultado: {label}")
        else:
            st.warning("Por favor, digite um texto antes de classificar.")

with col_b:
    st.info(
        "Este classificador usa um **pipeline TF-IDF + LinearSVM** treinado no corpus **Fake.Br** "
        "(3.600 reais / 3.600 falsas). Os resultados de reprodução indicaram **F1 ≈ 0.98** tanto no "
        "baseline quanto com **POS**, confirmando as conclusões do artigo (ganho marginal com sintaxe superficial)."
    )

# ====== Rodapé ======
st.markdown("""
---
Projeto para **Aprendizagem de Máquina — UNIFESSPA (2025)**.  
Reprodução científica baseada em *Fagundes, Roman & Digiampietri (2024), SBC*.
""")

