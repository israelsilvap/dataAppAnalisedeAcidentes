import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Configurar o layout da página para wide mode
st.set_page_config(layout="wide")

# Equipe:
# Dnilson Correia de Sousa - 495986
# Francisco Silvan Felipe do Carmo - 496641
# Francisco Wendel Alves Ribeiro - 510424
# Gabriel Araújo Texeira - 511476
# Israel da Silva Pereira - 497145
# João Artur Sales Rocha - 511375
# Matheus Nunes Vieira - 510011

# 1. Introdução
# A primeira etapa envolve o carregamento do dataset e a importação das bibliotecas necessárias.

# Carregar o dataset
df = pd.read_csv('datatran2023.csv', encoding='latin1', delimiter=';')

# Título do App
st.title("Análise de Acidentes de Trânsito 2023")

# Visualizar as primeiras linhas do dataset
st.header("Visualizar as primeiras linhas do dataset")
st.write(df.head())

# Adicionando uma linha de separação
st.markdown("---")  # Linha horizontal de separação

# 2. Obtenção dos Dados (Obtain)
# Aqui, primeiramente importamos as bibliotecas e na sequência obtemos e carregamos o dataset para começarmos o processo de exploração.

# 3. Limpeza dos Dados (Scrub)
# Nesta etapa, removemos valores nulos, tratamos colunas com formato incorreto (ex.: latitude, longitude, datas) e verificamos se há valores inválidos.

# Remover linhas com valores nulos nas colunas críticas
df_cleaned = df.dropna(subset=['classificacao_acidente', 'regional', 'delegacia', 'uop']).copy()

# Converter a coluna 'data_inversa' para o tipo datetime
df_cleaned['data_inversa'] = pd.to_datetime(df_cleaned['data_inversa'], format='%Y-%m-%d')
df_cleaned['horario'] = pd.to_datetime(df_cleaned['horario'], errors='coerce', format='%H:%M:%S').dt.hour

# Converter latitude e longitude para float
df_cleaned['latitude'] = df_cleaned['latitude'].str.replace(',', '.').astype(float)
df_cleaned['longitude'] = df_cleaned['longitude'].str.replace(',', '.').astype(float)

# Análise descritiva
st.header("Exploração dos Dados")
st.subheader("Resumo estatístico das colunas numéricas:")
st.write(df_cleaned.describe())

# Adicionando uma linha de separação
st.markdown("---")  # Linha horizontal de separação

# 4. Exploração dos Dados (Explore)
# Antes de fazermos as visualizações, verificamos a distribuição dos valores numéricos (ex.: mortos, feridos).

# Distribuição das Condições Climáticas
# st.subheader("Distribuição das Condições Climáticas")
# Distribuição das Condições Climáticas
st.markdown("<h3 style='text-align: center;'>Distribuição das Condições Climáticas</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(10, 7))
    df_cleaned['condicao_metereologica'].value_counts().plot(kind='bar', color='green', ax=ax)
    ax.set_title('Distribuição das Condições Climáticas em Acidentes')
    ax.set_xlabel('Condições Climáticas')
    ax.set_ylabel('Frequência')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

with col2:
    # Criar uma nova coluna com o total de vítimas (mortos + feridos graves + feridos leves)
    df_cleaned['total_vitimas'] = df_cleaned['mortos'] + df_cleaned['feridos_graves'] + df_cleaned['feridos_leves']

    # Agrupar por condicao_metereologica e somar os valores
    df_grouped_clima = df_cleaned.groupby('condicao_metereologica')[['mortos', 'total_vitimas']].sum()

    # Calcular a porcentagem de mortos em relação ao total de vítimas
    df_grouped_clima['percentual_mortos'] = (df_grouped_clima['mortos'] / df_grouped_clima['total_vitimas']) * 100

    # Substituir valores NaN por 0, que ocorrem quando o total de vítimas é zero
    df_grouped_clima['percentual_mortos'].fillna(0, inplace=True)

    # Criar uma classe "Outros" para Granizo e Neve
    condicoes_agrupadas = df_grouped_clima.copy()
    condicoes_agrupadas.loc['Outros'] = condicoes_agrupadas.loc[['Granizo', 'Neve']].sum()

    # Remover as entradas originais de "Granizo" e "Neve"
    condicoes_agrupadas = condicoes_agrupadas.drop(['Granizo', 'Neve'], errors='ignore')

    # Ordenar os valores pela porcentagem de mortos (ordem decrescente)
    df_grouped_clima_sorted = condicoes_agrupadas.sort_values(by='percentual_mortos', ascending=False)

    # Função para formatar valores muito pequenos como uma string vazia
    def autopct_format(pct):
        return ('' if pct < 0.1 else f'{pct:.1f}%')

    # Plotar o gráfico de pizza com a porcentagem de mortos
    fig, ax = plt.subplots(figsize=(8.3, 8.3))

    # Criação do gráfico de pizza
    wedges, texts, autotexts = ax.pie(
        df_grouped_clima_sorted['percentual_mortos'],
        labels=None,
        autopct=autopct_format,
        startangle=90,
        colors=[
            '#FF9999',
            '#006400',
            '#99FF99',
            '#FFCC99',
            '#BA55D3',
            '#FF4500',
            '#1E90FF',
            '#FFD700'
        ][:len(df_grouped_clima_sorted)],
        pctdistance=0.85,
        explode=[0.05] * len(df_grouped_clima_sorted)
    )

    # Adicionar título
    ax.set_title('Distribuição de Mortes por Condição Climática')

    # Melhorar o espaçamento das porcentagens e texto
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)

    # Centralizar o gráfico para melhor visualização
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    # Adicionar legenda para "Outros"
    ax.legend(
        wedges,
        df_grouped_clima_sorted.index,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Condições Climáticas",
    )

    # Adicionar explicação para a classe "Outros"
    ax.text(1.1, -1.1, 'Outros: Neve e Granizo', fontsize=12, ha='center')

    # Ajustar o layout
    fig.tight_layout()
    st.pyplot(fig)

# Adicionando uma linha de separação
st.markdown("---")  # Linha horizontal de separação

# 4.3 Top 10 Causas de Acidentes
st.markdown("<h3 style='text-align: center;'>Top 10 Causas de Acidentes</h3>", unsafe_allow_html=True)
plt.figure(figsize=(10, 6))
df_cleaned['causa_acidente'].value_counts().head(10).plot(kind='bar', color='blue')
plt.xlabel('Causa do Acidente')
plt.ylabel('Frequência')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(plt)

# Adicionando uma linha de separação
st.markdown("---")  # Linha horizontal de separação

st.markdown("<h3 style='text-align: center;'>Top 10 Causas de Acidentes Relacionadas a Mortes</h3>", unsafe_allow_html=True)
# Causas de acidentes relacionadas a mortes
plt.figure(figsize=(10, 6))
df_grouped_causa_mortos = df_cleaned.groupby('causa_acidente')['mortos'].sum().sort_values(ascending=False).head(10)
df_grouped_causa_mortos.plot(kind='bar', color='red')
plt.xlabel('Causa do Acidente')
plt.ylabel('Total de Mortos')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
st.pyplot(plt)

# Adicionando uma linha de separação
st.markdown("---")  # Linha horizontal de separação

#
st.markdown("<h3 style='text-align: center;'>Top 10 Tipos de Acidente e Gravidade (Somatório de Mortos e Feridos)</h3>", unsafe_allow_html=True)
# Criar uma nova coluna com o somatório dos feridos graves, leves e mortos
df_cleaned['total_impacto'] = df_cleaned['mortos'] + df_cleaned['feridos_graves'] + df_cleaned['feridos_leves']

# Agrupar por tipo de acidente e ordenar pelo somatório
df_grouped_tipo_acidente_impacto = df_cleaned.groupby('tipo_acidente')[['mortos', 'feridos_graves', 'feridos_leves', 'total_impacto']].sum().sort_values(by='total_impacto', ascending=False)

# Plotar o gráfico
df_grouped_tipo_acidente_impacto[['mortos', 'feridos_graves', 'feridos_leves']].plot(kind='bar', stacked=True, color=['red', 'blue', 'green'], figsize=(10, 6))
plt.xlabel('Tipo de Acidente')
plt.ylabel('Total de Mortos / Feridos')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
st.pyplot(plt)

# Adicionando uma linha de separação
st.markdown("---")  # Linha horizontal de separação

# Gravidade dos Acidentes ao Longo do Dia
st.markdown("<h3 style='text-align: center;'>Gravidade dos Acidentes ao Longo do Dia (Mortos ou Feridos Graves)</h3>", unsafe_allow_html=True)
plt.figure(figsize=(10, 6))
data = df_cleaned.groupby('horario')[['mortos', 'feridos_graves']].sum()
plt.plot(data.index, data['mortos'], marker='o', label='Mortos')
plt.plot(data.index, data['feridos_graves'], marker='o', label='Feridos Graves')
plt.xlabel('Hora do Dia')
plt.ylabel('Total de Mortos / Feridos Graves')
plt.legend()
plt.tight_layout()
st.pyplot(plt)

# Adicionando uma linha de separação
st.markdown("---")  # Linha horizontal de separação

#
st.markdown("<h3 style='text-align: center;'>Distribuição de Acidentes por Estado (UF)</h3>", unsafe_allow_html=True)
# Distribuição de acidentes por estado (UF)
plt.figure(figsize=(10, 6))
df_cleaned['uf'].value_counts().plot(kind='bar', color='purple')
plt.xlabel('Estado (UF)')
plt.ylabel('Número de Acidentes')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
st.pyplot(plt)

# Adicionando uma linha de separação
st.markdown("---")  # Linha horizontal de separação

#
st.markdown("<h3 style='text-align: center;'>Top 10 BRs com Mais Acidentes</h3>", unsafe_allow_html=True)
br_accident_count = df_cleaned['br'].value_counts().reset_index()

# Ordenar as BRs pela contagem de acidentes e pegar as 10 mais frequentes
br_accident_count  = br_accident_count.sort_values(by='count', ascending=False).head(10)

# Criar gráfico
plt.figure(figsize=(10, 6))
plt.bar(br_accident_count['br'].astype(str), br_accident_count['count'], color='blue')
plt.xlabel('BR')
plt.ylabel('Número de Acidentes')
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar o gráfico
plt.show()
st.pyplot(plt)

# Adicionando uma linha de separação
st.markdown("---")  # Linha horizontal de separação

# 4.7 Distribuição Geográfica dos Acidentes
st.markdown("<h3 style='text-align: center;'>Distribuição Geográfica dos Acidentes de Trânsito por Tipo de Acidente</h3>", unsafe_allow_html=True)

# Mapa interativo dos acidentes
fig = px.scatter_mapbox(df_cleaned, lat="latitude", lon="longitude", hover_name="uf",
                        hover_data=["causa_acidente", "mortos", "feridos_graves"],
                        color="tipo_acidente", zoom=4, height=600,
                        labels={"tipo_acidente": "Tipos de Acidentes"})

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(
    legend_title="Tipos de Acidentes",
    legend=dict(
        title="Tipos de Acidentes",
        traceorder="normal",
        itemsizing="constant",
        itemclick="toggleothers"
    )
)
st.plotly_chart(fig)

# Adicionando uma linha de separação
st.markdown("---")  # Linha horizontal de separação

# 5. Modelagem Preditiva (Model)
# Agora que temos uma boa compreensão dos dados, podemos tentar prever a gravidade dos acidentes usando modelos de aprendizado de máquina. Vamos utilizar o Random Forest, um modelo poderoso para classificação.

# Agrupando os dados com as colunas relevantes para predição
df_grouped = df_cleaned[['latitude', 'longitude', 'dia_semana', 'horario', 'causa_acidente', 'tipo_acidente', 'classificacao_acidente']]

# Preparar os dados
X = df_grouped.drop(['classificacao_acidente'], axis=1)

# Codificar variáveis categóricas
label_encoder = LabelEncoder()
for column in ['dia_semana', 'causa_acidente', 'tipo_acidente']:
    X[column] = label_encoder.fit_transform(X[column])

# Codificar a variável alvo 'classificacao_acidente'
y_class = label_encoder.fit_transform(df_grouped['classificacao_acidente'])

# Aplicar OverSampling para balancear as classes
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y_class)

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train_class, y_test_class = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Agora podemos normalizar os dados
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Treinar o modelo Random Forest
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_normalized, y_train_class)

# Fazer previsões
y_pred_class = rf_classifier.predict(X_test_normalized)

# Relatório de Classificação
st.subheader("Relatório de Classificação")
st.text(classification_report(y_test_class, y_pred_class, target_names=label_encoder.classes_))

# Adicionando uma linha de separação
st.markdown("---")  # Linha horizontal de separação

# Matriz de Confusão

st.subheader("Matriz de Confusão")
conf_matrix = confusion_matrix(y_test_class, y_pred_class)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Previsto")
plt.ylabel("Verdadeiro")
st.pyplot(plt)

# Adicionando uma linha de separação
st.markdown("---")  # Linha horizontal de separação

# Importância das Features
st.subheader("Importância das Features no Modelo Random Forest")
importances = rf_classifier.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Visualizar a importância das features com um gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.tight_layout()
st.pyplot(plt)

# Adicionando uma linha de separação
st.markdown("---")  # Linha horizontal de separação

# Curva ROC corrigida
st.subheader("Curva ROC")

# Binarizar as classes para a curva ROC (somente as classes alvo, não atributos como latitude e longitude)
y_test_bin = label_binarize(y_test_class, classes=[0, 1, 2])  # Ajuste as classes conforme necessário, se houver mais classes
y_pred_bin = label_binarize(y_pred_class, classes=[0, 1, 2])

# Tamanhos para melhorar a visualização
plt.figure(figsize=(8, 6))

# Gerar curvas ROC para cada classe (ignorar atributos como latitude e longitude)
for i in range(y_test_bin.shape[1]):  # Certificar que estamos iterando pelas classes binarizadas
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc = auc(fpr, tpr)
    
    # Adicionar a curva para cada classe
    plt.plot(fpr, tpr, lw=2, label=f'Classe {label_encoder.classes_[i]} (AUC = {roc_auc:.2f})')

# Adicionar a linha diagonal que representa uma classificação aleatória
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Melhorar títulos e layout
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.legend(loc="lower right")
plt.tight_layout()

# Exibir o gráfico na interface Streamlit
st.pyplot(plt)

# Adicionando uma linha de separação
st.markdown("---")  # Linha horizontal de separação

# 6. Interação com o Usuário para Inserir Valores dos Atributos
st.markdown("<h3 style='text-align: center;'>Faça uma Previsão com o Modelo</h3>", unsafe_allow_html=True)

# Listar todas as categorias possíveis (incluindo as que podem não estar no conjunto de treinamento)
all_dia_semana = ['segunda-feira', 'terça-feira', 'quarta-feira', 'quinta-feira', 'sexta-feira', 'sábado', 'domingo']
all_causa_acidente = list(df_cleaned['causa_acidente'].unique())
all_tipo_acidente = list(df_cleaned['tipo_acidente'].unique())

# Ajustar o LabelEncoder com todas as categorias possíveis
label_encoder_dia_semana = LabelEncoder().fit(all_dia_semana)
label_encoder_causa_acidente = LabelEncoder().fit(all_causa_acidente)
label_encoder_tipo_acidente = LabelEncoder().fit(all_tipo_acidente)

# Listar opções disponíveis para o usuário selecionar
dia_semana_options = all_dia_semana
causa_acidente_options = all_causa_acidente
tipo_acidente_options = all_tipo_acidente

# Campos de entrada para os atributos
latitude = st.number_input("Latitude (ex: -3.69)")
longitude = st.number_input("Longitude (ex: -40.35)")
dia_semana = st.selectbox("Dia da Semana", dia_semana_options)
horario = st.slider("Horário (em horas)", 0, 23, 12)
causa_acidente = st.selectbox("Causa do Acidente", causa_acidente_options)
tipo_acidente = st.selectbox("Tipo de Acidente", tipo_acidente_options)

# Botão para fazer a previsão
if st.button("Fazer Previsão"):
    try:
        # Codificar as variáveis categóricas com o LabelEncoder ajustado
        dia_semana_encoded = label_encoder_dia_semana.transform([dia_semana])[0]
        causa_acidente_encoded = label_encoder_causa_acidente.transform([causa_acidente])[0]
        tipo_acidente_encoded = label_encoder_tipo_acidente.transform([tipo_acidente])[0]
    except ValueError as e:
        st.error(f"Erro: {e}. Valor não reconhecido nas opções fornecidas.")
        st.stop()  # Parar a execução se houver um erro de codificação

    # Criar um DataFrame com os valores inseridos pelo usuário
    user_data = pd.DataFrame({
        'latitude': [latitude],
        'longitude': [longitude],
        'dia_semana': [dia_semana_encoded],
        'horario': [horario],
        'causa_acidente': [causa_acidente_encoded],
        'tipo_acidente': [tipo_acidente_encoded]
    })

    # Normalizar os dados de entrada usando o scaler ajustado no treinamento
    user_data_normalized = scaler.transform(user_data)

    # Fazer a previsão com o modelo Random Forest
    prediction = rf_classifier.predict(user_data_normalized)

    # Exibir o resultado da previsão
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    st.subheader(f"A previsão do modelo é: {predicted_class}")
