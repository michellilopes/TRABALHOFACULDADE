# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import random
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Previsão do PIB per capita", page_icon="🌎", layout="wide")

st.title("🌎 Predição do PIB per capita")
st.write(
    "Aplicação de análise e previsão do PIB per capita utilizando indicadores "
    "econômicos do Banco Mundial e modelos de Machine Learning."
)

PAISES = {
    "BRA": "Brasil", "ARG": "Argentina", "CHL": "Chile", "COL": "Colômbia",
    "MEX": "México", "USA": "Estados Unidos", "CAN": "Canadá", "DEU": "Alemanha",
    "FRA": "França", "GBR": "Reino Unido", "ITA": "Itália", "ESP": "Espanha",
    "PRT": "Portugal", "NLD": "Países Baixos", "SWE": "Suécia", "CHE": "Suíça",
    "NOR": "Noruega", "POL": "Polônia", "JPN": "Japão", "CHN": "China",
    "KOR": "Coreia do Sul", "IND": "Índia", "IDN": "Indonésia", "SGP": "Singapura",
    "AUS": "Austrália", "NZL": "Nova Zelândia", "ZAF": "África do Sul",
    "EGY": "Egito", "NGA": "Nigéria", "ARE": "Emirados Árabes Unidos"
}

INDICADORES = {
    "NY.GDP.PCAP.CD": "PIB_per_capita",
    "SP.POP.TOTL": "Populacao",
    "NY.GDP.MKTP.KD.ZG": "Crescimento_PIB",
    "FP.CPI.TOTL.ZG": "Inflacao",
    "SL.UEM.TOTL.ZS": "Desemprego",
    "SP.URB.TOTL.IN.ZS": "Urbanizacao",
    "IT.NET.USER.ZS": "Acesso_Internet"
}

FEATURES = [
    "Populacao", "Crescimento_PIB", "Inflacao",
    "Desemprego", "Urbanizacao", "Acesso_Internet"
]
TARGET = "PIB_per_capita"

@st.cache_data(show_spinner=False, ttl=86400)
def buscar_indicador(codigo, nome):
    url = (
        f"https://api.worldbank.org/v2/country/all/indicator/{codigo}"
        "?format=json&per_page=20000"
    )
    resposta = requests.get(url, timeout=60)
    resposta.raise_for_status()
    dados = resposta.json()
    if not isinstance(dados, list) or len(dados) < 2 or dados[1] is None:
        return pd.DataFrame()
    registros = []
    for item in dados[1]:
        codigo_pais = item.get("countryiso3code")
        if codigo_pais in PAISES and item.get("date") and item.get("value") is not None:
            registros.append({
                "Pais": PAISES[codigo_pais],
                "Codigo": codigo_pais,
                "Ano": int(item["date"]),
                nome: item["value"]
            })
    return pd.DataFrame(registros)

@st.cache_data(show_spinner="Consultando a API do Banco Mundial...")
def carregar_dados(ano_inicio, ano_fim):
    dfs = []
    for codigo, nome in INDICADORES.items():
        df_temp = buscar_indicador(codigo, nome)
        if not df_temp.empty:
            dfs.append(df_temp)
    if not dfs:
        raise ValueError("A API do Banco Mundial não retornou dados.")
    df = dfs[0]
    for df_temp in dfs[1:]:
        df = pd.merge(df, df_temp, on=["Pais", "Codigo", "Ano"], how="outer")
    df = df.sort_values(["Pais", "Ano"]).reset_index(drop=True)
    return df[(df["Ano"] >= ano_inicio) & (df["Ano"] <= ano_fim)].copy()

def criar_pipeline(parametros=None):
    parametros = parametros or {}
    rf_params = {
        "n_estimators": parametros.get("n_estimators", 300),
        "max_depth": parametros.get("max_depth", 10),
        "min_samples_leaf": parametros.get("min_samples_leaf", 2),
        "min_samples_split": parametros.get("min_samples_split", 2),
        "max_features": parametros.get("max_features", 1.0),
        "random_state": 42,
        "n_jobs": -1
    }
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("modelo", RandomForestRegressor(**rf_params))
    ])

def treinar_modelos(df, n_iter=20, tamanho_populacao=8, geracoes=5):
    dados = df.dropna(subset=[TARGET]).copy()
    treino = dados[dados["Ano"] <= 2020].copy()
    teste = dados[dados["Ano"] > 2020].copy()

    if treino.empty or teste.empty:
        raise ValueError(
            "Não há dados suficientes para separar treino (até 2020) e teste (após 2020). "
            "Verifique o período selecionado e a disponibilidade dos dados."
        )
    if treino[FEATURES].dropna(how="all").empty:
        raise ValueError("Não há indicadores disponíveis para treinar o modelo.")

    X_train, y_train = treino[FEATURES], treino[TARGET]
    X_test, y_test = teste[FEATURES], teste[TARGET]
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    modelo_base = criar_pipeline({
        "n_estimators": 300, "max_depth": 10, "min_samples_leaf": 2
    })
    modelo_base.fit(X_train, y_train)

    espaco_parametros = {
        "modelo__n_estimators": [100, 200, 300, 400, 500],
        "modelo__max_depth": [None, 5, 8, 10, 15, 20],
        "modelo__min_samples_leaf": [1, 2, 3, 4, 5],
        "modelo__min_samples_split": [2, 5, 10],
        "modelo__max_features": [1.0, "sqrt", 0.8]
    }
    busca = RandomizedSearchCV(
        estimator=criar_pipeline(),
        param_distributions=espaco_parametros,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    busca.fit(X_train, y_train)
    modelo_random = busca.best_estimator_

    espaco_genetico = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 5, 8, 10, 15, 20],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "min_samples_split": [2, 5, 10],
        "max_features": [1.0, "sqrt", 0.8]
    }
    random.seed(42)
    np.random.seed(42)

    def individuo_aleatorio():
        return {k: random.choice(v) for k, v in espaco_genetico.items()}

    def aptidao(individuo):
        estimador = criar_pipeline(individuo)
        scores = cross_val_score(
            estimador, X_train, y_train,
            scoring="neg_mean_absolute_error", cv=cv, n_jobs=1
        )
        return -scores.mean()

    def cruzamento(pai1, pai2):
        return {k: random.choice([pai1[k], pai2[k]]) for k in espaco_genetico}

    def mutacao(individuo, taxa=0.2):
        novo = individuo.copy()
        for chave, opcoes in espaco_genetico.items():
            if random.random() < taxa:
                novo[chave] = random.choice(opcoes)
        return novo

    populacao = [individuo_aleatorio() for _ in range(tamanho_populacao)]
    historico_genetico = []
    for geracao in range(geracoes):
        pontuados = sorted(
            [(aptidao(ind), ind) for ind in populacao], key=lambda x: x[0]
        )
        historico_genetico.append({
            "Geração": geracao + 1,
            "Melhor MAE de validação": pontuados[0][0]
        })
        elite = [ind for _, ind in pontuados[:2]]
        nova_populacao = elite.copy()
        while len(nova_populacao) < tamanho_populacao:
            candidatos = pontuados[:max(2, tamanho_populacao // 2)]
            pai1, pai2 = random.sample(candidatos, 2)
            nova_populacao.append(mutacao(cruzamento(pai1[1], pai2[1])))
        populacao = nova_populacao

    pontuados = sorted(
        [(aptidao(ind), ind) for ind in populacao], key=lambda x: x[0]
    )
    melhores_parametros_genetico = pontuados[0][1]
    modelo_genetico = criar_pipeline(melhores_parametros_genetico)
    modelo_genetico.fit(X_train, y_train)

    modelos = [
        ("Random Forest base", modelo_base),
        ("RandomizedSearchCV", modelo_random),
        ("Algoritmo Genético", modelo_genetico)
    ]
    avaliacoes = []
    previsoes = {}
    for nome, estimador in modelos:
        pred = estimador.predict(X_test)
        previsoes[nome] = pred
        avaliacoes.append({
            "Método": nome,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred)
        })

    metricas = pd.DataFrame(avaliacoes)
    resultado = teste[["Pais", "Codigo", "Ano", TARGET]].copy()
    resultado["PIB_Previsto"] = previsoes["Random Forest base"]
    resultado["Erro"] = resultado[TARGET] - resultado["PIB_Previsto"]
    comparacao = resultado.copy()
    comparacao["Erro_%"] = (
        (comparacao[TARGET] - comparacao["PIB_Previsto"]).abs()
        / comparacao[TARGET].replace(0, np.nan)
    ) * 100

    return {
        "dados_modelo": dados, "treino": treino, "teste": teste,
        "X_train": X_train, "X_test": X_test, "y_test": y_test,
        "modelo_base": modelo_base, "modelo_random": modelo_random,
        "modelo_genetico": modelo_genetico, "metricas": metricas,
        "resultado": resultado, "comparacao": comparacao,
        "historico_genetico": pd.DataFrame(historico_genetico),
        "melhores_parametros_random": busca.best_params_,
        "melhores_parametros_genetico": melhores_parametros_genetico
    }

with st.sidebar:
    st.header("⚙️ Configurações")
    ano_inicio = st.number_input("Ano inicial dos dados", min_value=2000, max_value=2025, value=2005)
    ano_fim = st.number_input("Ano final dos dados", min_value=int(ano_inicio), max_value=2025, value=2025)
    st.caption("Treinamento: até 2020 | Teste: após 2020")
    st.divider()
    st.write("Ajustes dos modelos")
    n_iter = st.slider("Iterações do RandomizedSearchCV", 5, 30, 20)
    tamanho_populacao = st.slider("Tamanho da população genética", 4, 12, 8)
    geracoes = st.slider("Gerações do algoritmo genético", 2, 8, 5)
    executar = st.button("🚀 Carregar dados e executar análise", type="primary", use_container_width=True)

if executar:
    try:
        with st.spinner("Buscando os indicadores no Banco Mundial..."):
            dados = carregar_dados(int(ano_inicio), int(ano_fim))
        st.session_state["dados"] = dados
        with st.spinner("Treinando e comparando os três modelos. Esta etapa pode levar alguns minutos..."):
            resultados = treinar_modelos(
                dados, n_iter=n_iter,
                tamanho_populacao=tamanho_populacao, geracoes=geracoes
            )
        st.session_state["resultados"] = resultados
        st.success("Análise concluída!")
    except Exception as e:
        st.error(f"Não foi possível concluir a análise: {e}")

if "dados" not in st.session_state:
    st.info("Configure o período na barra lateral e clique em **Carregar dados e executar análise** para começar.")
    st.markdown("""
    **Etapas da aplicação**
    1. Consulta dos indicadores na API do Banco Mundial.
    2. Organização dos dados dos 30 países.
    3. Treinamento com dados até 2020 e teste após 2020.
    4. Comparação dos três métodos de ajuste.
    """)
    st.stop()

df = st.session_state["dados"]
st.subheader("📊 Base de dados")
c1, c2, c3 = st.columns(3)
c1.metric("Países", df["Codigo"].nunique())
c2.metric("Registros", f"{len(df):,}".replace(",", "."))
c3.metric("Período disponível", f"{df['Ano'].min()}–{df['Ano'].max()}")
with st.expander("Visualizar dados e valores ausentes"):
    st.dataframe(df, use_container_width=True)
    ausentes = df.isna().sum().rename("Valores ausentes").to_frame()
    st.dataframe(ausentes, use_container_width=True)
st.download_button(
    "⬇️ Baixar base de dados (CSV)",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="dados_pib_mundo_banco_mundial.csv",
    mime="text/csv"
)

# Gráfico da evolução do Brasil
st.subheader("📈 Evolução do PIB per capita — Brasil")
brasil = df[df["Codigo"] == "BRA"].dropna(subset=[TARGET]).sort_values("Ano")
if not brasil.empty:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(brasil["Ano"], brasil[TARGET], marker="o")
    ax.set_xlabel("Ano")
    ax.set_ylabel("PIB per capita (US$)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("Não há dados de PIB per capita do Brasil no período selecionado.")

if "resultados" not in st.session_state:
    st.warning("A base foi carregada. Execute novamente a análise pela barra lateral para treinar os modelos.")
    st.stop()

r = st.session_state["resultados"]
st.divider()
st.subheader("🤖 Comparação dos modelos")
st.dataframe(
    r["metricas"].style.format({"MAE": "US$ {:,.2f}", "RMSE": "US$ {:,.2f}", "R²": "{:.4f}"}),
    use_container_width=True
)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, metrica, titulo in zip(
    axes, ["MAE", "RMSE", "R²"],
    ["MAE (menor é melhor)", "RMSE (menor é melhor)", "R² (maior é melhor)"]
):
    ax.bar(r["metricas"]["Método"], r["metricas"][metrica])
    ax.set_title(titulo)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

melhor_mae = r["metricas"].loc[r["metricas"]["MAE"].idxmin()]
melhor_rmse = r["metricas"].loc[r["metricas"]["RMSE"].idxmin()]
melhor_r2 = r["metricas"].loc[r["metricas"]["R²"].idxmax()]
st.markdown("**Interpretação automática das métricas**")
st.write(f"- Menor MAE: **{melhor_mae['Método']}** (US$ {melhor_mae['MAE']:,.2f}).")
st.write(f"- Menor RMSE: **{melhor_rmse['Método']}** (US$ {melhor_rmse['RMSE']:,.2f}).")
st.write(f"- Maior R²: **{melhor_r2['Método']}** (R² = {melhor_r2['R²']:.4f}).")

with st.expander("Ver melhores parâmetros encontrados"):
    st.write("**RandomizedSearchCV**")
    st.json(r["melhores_parametros_random"])
    st.write("**Algoritmo Genético**")
    st.json(r["melhores_parametros_genetico"])
    st.write("**Evolução do algoritmo genético**")
    st.dataframe(r["historico_genetico"], use_container_width=True)

st.subheader("🎯 PIB per capita: real x previsto")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(r["y_test"], r["modelo_base"].predict(r["X_test"]), alpha=0.6)
limite_min = min(r["y_test"].min(), r["modelo_base"].predict(r["X_test"]).min())
limite_max = max(r["y_test"].max(), r["modelo_base"].predict(r["X_test"]).max())
ax.plot([limite_min, limite_max], [limite_min, limite_max], linestyle="--")
ax.set_xlabel("PIB per capita real (US$)")
ax.set_ylabel("PIB per capita previsto (US$)")
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close(fig)

st.subheader("🔎 Importância dos indicadores")
rf = r["modelo_base"].named_steps["modelo"]
importancias = pd.DataFrame({
    "Indicador": FEATURES,
    "Importância": rf.feature_importances_
}).sort_values("Importância", ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(importancias["Indicador"], importancias["Importância"])
ax.invert_yaxis()
ax.set_xlabel("Importância")
st.pyplot(fig)
plt.close(fig)
st.dataframe(importancias, use_container_width=True)

st.subheader("🇧🇷 Previsão para o Brasil")
brasil_modelo = r["dados_modelo"][r["dados_modelo"]["Codigo"] == "BRA"].sort_values("Ano")
if not brasil_modelo.empty:
    ultimo = brasil_modelo.tail(1)
    previsao_brasil = float(r["modelo_base"].predict(ultimo[FEATURES])[0])
    a, b, c = st.columns(3)
    a.metric("Ano dos indicadores", int(ultimo["Ano"].iloc[0]))
    b.metric("PIB per capita real", f"US$ {ultimo[TARGET].iloc[0]:,.2f}")
    c.metric("PIB per capita previsto", f"US$ {previsao_brasil:,.2f}")
else:
    st.warning("Não há dados suficientes para gerar a previsão do Brasil.")

st.subheader("📄 Resultados detalhados")
st.dataframe(r["comparacao"].head(100), use_container_width=True)
st.download_button(
    "⬇️ Baixar previsões em CSV",
    data=r["resultado"].to_csv(index=False).encode("utf-8-sig"),
    file_name="previsoes_pib_per_capita.csv",
    mime="text/csv"
)
