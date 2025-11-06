import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Configurações de layout e tema
st.set_page_config(
    page_title="Fake News Detector - Reprodução Científica",
    page_icon="🧠",
    layout="wide",
)

# ====== Estilo ======
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: #FAFAFA;
}
h1, h2, h3 {
    color: #00B4FF;
}
.sidebar .sidebar-content {
    background: #0E1117;
}
.metric-table td, .metric-table th {
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

# ====== Cabeçalho ======
st.title("🧠 Detecção de Fake News em Português")
st.subheader("Reprodução científica baseada no artigo da SBC (Fagundes et al., 2024)")

# ====== Seção 1: Tabela de métricas ======
st.header("📊 Resultados dos Modelos")

dados = {
    "Modelo": ["Baseline (TF-IDF)", "POS + TF-IDF"],
    "Acurácia": [0.98, 0.98],
    "Precisão": [0.98, 0.98],
    "Recall": [0.98, 0.98],
    "F1-Score": [0.98, 0.98]
}

df = pd.DataFrame(dados)
st.table(df)

# ====== Seção 2: Gráfico de F1 ======
st.header("📈 Comparação de F1-Score")
st.image("grafico_comparativo.png", use_column_width=True)

# ====== Seção 3: Matriz de confusão ======
st.header("🧮 Matriz de Confusão (Modelo POS + TF-IDF)")
st.image("matriz_confusao.png", caption="Modelo POS + TF-IDF", use_column_width=True)

# ====== Seção 4: Teste de notícia ======
st.header("🗞️ Teste uma Notícia")
texto = st.text_area("Digite ou cole uma notícia em português para classificar:")

if st.button("Classificar notícia"):
    if texto.strip():
        modelo = joblib.load("modelo.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        vetor = vectorizer.transform([texto])
        pred = modelo.predict(vetor)[0]
        resultado = "🟥 FAKE NEWS" if pred == 0 else "🟩 NOTÍCIA REAL"
        st.subheader(f"Resultado: {resultado}")
    else:
        st.warning("Por favor, digite um texto antes de classificar.")

# ====== Rodapé ======
st.markdown("""
---
Projeto desenvolvido para a disciplina **Aprendizagem de Máquina - UNIFESSPA (2025)**  
Reprodução científica baseada em *Fagundes et al. (2024)* – Sociedade Brasileira de Computação.
""")
