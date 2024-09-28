import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from imblearn.over_sampling import RandomOverSampler
import textwrap

# Configurações da página
st.set_page_config(layout="wide", page_title="Análise de Acidentes de Trânsito 2023 🚗💥", initial_sidebar_state="expanded")

# Cache para carregar o dataset
@st.cache_data
def load_data():
    df = pd.read_csv('datatran2023.csv', encoding='latin1', delimiter=';')
    df_cleaned = df.dropna(subset=['classificacao_acidente', 'regional', 'delegacia', 'uop']).copy()
    df_cleaned['data_inversa'] = pd.to_datetime(df_cleaned['data_inversa'], format='%Y-%m-%d')
    df_cleaned['mes'] = df_cleaned['data_inversa'].dt.month  # Extrair mês para filtro
    df_cleaned['horario'] = pd.to_datetime(df_cleaned['horario'], errors='coerce', format='%H:%M:%S').dt.hour
    df_cleaned['latitude'] = df_cleaned['latitude'].str.replace(',', '.').astype(float)
    df_cleaned['longitude'] = df_cleaned['longitude'].str.replace(',', '.').astype(float)
    return df_cleaned

# Cache para treinar o modelo e escalonador
@st.cache_resource
def train_model():
    # Preparar os dados para o modelo
    df_cleaned = load_data()
    df_grouped = df_cleaned[['latitude', 'longitude', 'dia_semana', 'horario', 'causa_acidente', 'tipo_acidente', 'classificacao_acidente']]

    # Preparar os dados
    X = df_grouped.drop(['classificacao_acidente'], axis=1)

    # Codificar variáveis categóricas
    label_encoder = LabelEncoder()

    # Dicionário para armazenar o LabelEncoder de cada coluna
    label_encoders = {}
    # Codificar cada coluna e armazenar o LabelEncoder correspondente
    for column in ['dia_semana', 'causa_acidente', 'tipo_acidente']:
        label_encoders[column] = LabelEncoder()  # Cria um LabelEncoder para a coluna
        X[column] = label_encoders[column].fit_transform(X[column])  # Aplica o LabelEncoder à coluna

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

    y_pred_class = rf_classifier.predict(X_test_normalized)

    return rf_classifier, scaler, label_encoders, label_encoder, X, y_pred_class, y_test_class


# Título e equipe
st.markdown("<h1 style='text-align: center;'>Análise de Acidentes de Trânsito 2023 🚗💥</h1>", unsafe_allow_html=True)

# Adicionando uma linha de separação
st.markdown("---")  # Linha horizontal de separação

# Inicialização de estado
if 'exibir_sobre' not in st.session_state:
    st.session_state['exibir_sobre'] = False  # Define o estado inicial como False

# Definir o nome do botão com base no estado atual
botao_label = "🏠 Início" if st.session_state.exibir_sobre else " ℹ️ Sobre"

# Layout de colunas dentro da sidebar para centralizar o botão
col1, col2, col3 = st.sidebar.columns([1, 1, 1])  # Ajustar a proporção das colunas

with col2:  # Coluna central, onde ficará o botão
    if st.button(botao_label):
        st.session_state.exibir_sobre = not st.session_state.exibir_sobre
        # Forçar a atualização imediata da página
        st.rerun()  # Força uma nova execução com o estado atualizado

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
# Aba 1: Exibição dos Dados
if st.session_state.exibir_sobre:
    st.markdown("""

        Este **Data App** foi desenvolvido com o objetivo de fornecer uma **análise interativa e abrangente** dos acidentes de trânsito ocorridos no Brasil em 2023, utilizando os dados fornecidos pela **Polícia Rodoviária Federal**. Através de **gráficos dinâmicos**, **mapas geoespaciais** e um **modelo preditivo de aprendizado de máquina**, o objetivo é permitir que o usuário explore os padrões de acidentes, suas causas principais e a gravidade dos mesmos de maneira **intuitiva**.

        ---
                      
        ## ⚙️ Funcionalidades e Seções Interativas:

        - **Exibição de Dados** 📊: Nesta seção, você pode **visualizar as primeiras linhas do dataset** para uma análise rápida dos registros ou consultar um **resumo estatístico** das variáveis numéricas. Isso oferece uma visão geral dos dados coletados, como datas, localidades, condições dos acidentes e gravidade.
        
        <br>
        
        - **Gráficos de Análise** 📈: Esta seção apresenta gráficos interativos para **explorar os padrões** de acidentes de trânsito. Aqui você pode visualizar:     
            - A **distribuição das condições climáticas** durante os acidentes 🌦️.
            - As **principais causas** que levaram a acidentes ⚠️.
            - A relação entre **tipos de acidentes e a gravidade** dos mesmos, com base no número de vítimas 🚨.
            - O padrão de **acidentes ao longo do dia**, para identificar os horários de maior risco 🕒.
            - A **distribuição de acidentes por estado** (UF) 🗺️ e as **rodovias federais (BRs)** com maior número de ocorrências 🛣️.
        
        <br>
        
        - **Mapa de Distribuição** 🌍: Esta funcionalidade traz um **mapa interativo** que permite visualizar a **localização geográfica dos acidentes**, categorizados por tipo de acidente e severidade. Ele utiliza um mapa geoespacial para destacar os **pontos mais críticos**, proporcionando uma análise visual detalhada da distribuição dos incidentes.
        
        <br>
        
        - **Avaliações da Modelagem** 🤖: Nesta aba, é possível visualizar as avaliações do **modelo Random Forest**, treinado com os dados de acidentes para **prever a gravidade dos acidentes**. Isso inclui:
            - **Matriz de Confusão** 🧮: Uma visualização que compara previsões corretas e incorretas do modelo.
            - **Curva ROC** 📉: Avalia o desempenho do modelo em termos de sensibilidade e especificidade.
            - **Importância das Features** 🔍: Mostra as variáveis que mais influenciam as previsões do modelo.

        <br>
        
        - **Previsão do Modelo** 🔮: Aqui você pode **fazer previsões personalizadas** sobre a gravidade dos acidentes com base em parâmetros fornecidos pelo usuário, como **localização**, **causas** e **tipos de acidentes**. Isso permite simular cenários e obter insights mais práticos.
        
        --- 

        ## 🛠️ Como Funcionam os Filtros:

        Para melhorar a **interatividade** e **precisão da análise**, o Data App oferece **filtros dinâmicos** na barra lateral, que permitem ajustar a visualização de acordo com seus interesses específicos:

        - **Filtro por Estado** 🏙️: Você pode selecionar um estado específico do Brasil ou optar por "Todos os estados" para visualizar os dados a nível nacional.
        
        
        - **Filtro por Período de Tempo** ⏳: Com um **slider de meses**, é possível ajustar o intervalo de tempo para analisar acidentes que ocorreram entre um mês específico ou em todo o ano de 2023.
                
        ---
                
        ## 🎯 Utilidade e Objetivo Final:

        O objetivo principal deste Data App é **auxiliar autoridades, pesquisadores e o público em geral** a entender os padrões dos acidentes de trânsito no Brasil. Essa análise permite a identificação de **pontos críticos**, auxiliando na criação de **políticas públicas** mais eficazes, além de fornecer uma **ferramenta educativa** para aumentar a conscientização sobre segurança no trânsito.

        Ao combinar dados robustos com técnicas de **visualização interativa** e **aprendizado de máquina**, esta aplicação tem o potencial de **gerar insights significativos** para reduzir o número de acidentes e melhorar a **segurança viária** no país.
    """, unsafe_allow_html=True)

else:

    # Filtros interativos
    df_cleaned = load_data()

    # Filtros na barra lateral
    st.sidebar.markdown("# Filtros de Análise")
    st.sidebar.markdown("Determine abaixo os parâmetros de análises")

    # Filtro por Estado
    estados = df_cleaned['uf'].unique().tolist()
    estados.insert(0, "Todos os estados")  # Adicionar a opção de análise nacional
    estado_selecionado = st.sidebar.selectbox("Escolha o estado", estados)

    # Filtro por faixa de tempo (meses)
    mes_inicio, mes_fim = st.sidebar.slider(
        "Selecione a faixa de meses",
        min_value=1, max_value=12, value=(1, 12)
    )

    # Filtrar os dados com base no estado e meses selecionados
    if estado_selecionado != "Todos os estados":
        df_filtrado = df_cleaned[(df_cleaned['uf'] == estado_selecionado) & (df_cleaned['mes'] >= mes_inicio) & (df_cleaned['mes'] <= mes_fim)]
    else:
        df_filtrado = df_cleaned[(df_cleaned['mes'] >= mes_inicio) & (df_cleaned['mes'] <= mes_fim)]

    # Atualizar abas com base na escolha do estado
    if estado_selecionado == "Todos os estados" and (mes_inicio == 1 and mes_fim == 12):
        abas_disponiveis = ["Exibição de Dados", "Gráficos de Análise", "Mapa com Distribuição de Acidentes", "Avaliações da Modelagem", "Previsão do Modelo"]
    else:
        abas_disponiveis = ["Exibição de Dados", "Gráficos de Análise", "Mapa com Distribuição de Acidentes"]

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("# Guias de analises")

    selected_tab = st.sidebar.selectbox("Escolha a seção", abas_disponiveis)

    if selected_tab == "Exibição de Dados" :
        # Somente para aba 1
        #st.sidebar.title("Opções - Exibição de Dados")
        option = st.sidebar.radio(
            "Selecione a visualização:",
            ('Visualizar as primeiras linhas do dataset', 'Resumo estatístico')
        )

        # Verificar a opção escolhida e exibir a respectiva visualização
        if option == 'Visualizar as primeiras linhas do dataset':
            # Centralizando o título e o DataFrame
            st.markdown("<h3 style='text-align: center;'>Visualizar as primeiras linhas do dataset</h3>", unsafe_allow_html=True)
            # Para centralizar o DataFrame visualmente, usamos colunas vazias
            col1, col2, col3 = st.columns([0.2, 3, 0.2])  # Ajusta a proporção para centralizar
            with col2:
                st.write(df_filtrado.head(10))
                num_rows = len(df_filtrado)
                st.markdown("---")  # Linha horizontal de separação
                st.markdown(f"<p style='text-align: center;'><strong>{num_rows}</strong> ocorrências.</p>", unsafe_allow_html=True)

        elif option == 'Resumo estatístico':
            # Centralizando o título e o DataFrame
            st.markdown("<h3 style='text-align: center;'>Resumo estatístico das colunas numéricas</h3>", unsafe_allow_html=True)
            # Para centralizar o DataFrame visualmente, usamos colunas vazias
            col1, col2, col3 = st.columns([1, 3, 1])  # Ajusta a proporção para centralizar
            with col2:
                st.write(df_filtrado.describe())


    # Aba 2: Gráficos de Análise
    elif selected_tab == "Gráficos de Análise" :
        # Somente para aba 2
        #st.sidebar.title("Opções - Gráficos de Análise")
        if estado_selecionado != "Todos os estados":
                option2 = st.sidebar.radio(
            "Selecione o gráfico:",
            (
                'Distribuição das Condições Climáticas',
                'Top Causas de Acidentes',
                'Top 10 Tipos de Acidente e Gravidade',
                'Gravidade dos Acidentes ao Longo do Dia',
                'Top 10 BRs com Mais Acidentes'
            )
        )
        else:
            option2 = st.sidebar.radio(
            "Selecione o gráfico:",
            (
                'Distribuição das Condições Climáticas',
                'Top Causas de Acidentes',
                'Top 10 Tipos de Acidente e Gravidade',
                'Gravidade dos Acidentes ao Longo do Dia',
                'Distribuição de Acidentes por Estado (UF)',
                'Top 10 BRs com Mais Acidentes'
            )
        )
        

        # Gráfico: Distribuição das Condições Climáticas
        if option2 == 'Distribuição das Condições Climáticas':
            st.markdown("<h3 style='text-align: center;'>Distribuição das Condições Climáticas</h3>", unsafe_allow_html=True)

            st.markdown('''
                            <style>
                            .justified-text {
                                text-align: justify;
                            }
                            </style>
                            <br>
                            <div class="justified-text">
                                Os gráficos a seguir apresentam dados sobre acidentes de trânsito em diferentes condições climáticas. 
                                O da esquerda mostra a Distribuição das Condições Climáticas em Acidentes, ilustrando a 
                                frequência de acidentes sob cada condição específica, como Céu Claro, Nublado e Chuva. 
                                Já o da direita exibe a Distribuição de Mortes por Condição Climática, representando a proporção 
                                de fatalidades associadas a cada tipo de condição climática, permitindo uma comparação visual entre 
                                a quantidade de acidentes e a gravidade das consequências em termos de mortalidade para cada condição.
                            </div>
                            <br>
                        ''', unsafe_allow_html=True)


            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(10, 7))
                df_filtrado['condicao_metereologica'].value_counts().plot(kind='bar', color='green', ax=ax)
                ax.set_title('Distribuição das Condições Climáticas em Acidentes')
                ax.set_xlabel('Condições Climáticas')
                ax.set_ylabel('Frequência')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                st.pyplot(fig)

            with col2:
                df_filtrado['total_vitimas'] = df_filtrado['mortos'] + df_filtrado['feridos_graves'] + df_filtrado['feridos_leves']
                df_grouped_clima = df_filtrado.groupby('condicao_metereologica')[['mortos', 'total_vitimas']].sum()
                df_grouped_clima['percentual_mortos'] = (df_grouped_clima['mortos'] / df_grouped_clima['total_vitimas']) * 100
                df_grouped_clima['percentual_mortos'].fillna(0, inplace=True)
                df_grouped_clima_sorted = df_grouped_clima.sort_values(by='percentual_mortos', ascending=False)

                fig, ax = plt.subplots(figsize=(10, 8))
                wedges, texts, autotexts = ax.pie(
                    df_grouped_clima_sorted['percentual_mortos'],
                    labels=None,
                    autopct=lambda pct: '' if pct < 0.1 else f'{pct:.1f}%',
                    startangle=90,
                    pctdistance=0.85,
                    explode=[0.05] * len(df_grouped_clima_sorted)
                )
                ax.set_title('Distribuição de Mortes por Condição Climática')
                centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                fig.gca().add_artist(centre_circle)
                ax.legend(wedges, df_grouped_clima_sorted.index, loc="center left", bbox_to_anchor=(1, 0.5))
                st.pyplot(fig)

            

        elif option2 == 'Top Causas de Acidentes':
            st.markdown("<h3 style='text-align: center;'>Top Causas de Acidentes </h3>", unsafe_allow_html=True)

            # Definir o comprimento máximo de caracteres por linha
            max_label_length = 20  # Ajuste esse valor conforme necessário

            # Obter os rótulos originais e quebrar automaticamente os longos para o primeiro gráfico
            labels_causas = df_filtrado['causa_acidente'].value_counts().head(10).index
            wrapped_labels_causas = [textwrap.fill(label, max_label_length) for label in labels_causas]

            # Obter os rótulos originais e quebrar automaticamente os longos para o segundo gráfico
            labels_mortes = df_filtrado.groupby('causa_acidente')['mortos'].sum().sort_values(ascending=False).head(10).index
            wrapped_labels_mortes = [textwrap.fill(label, max_label_length) for label in labels_mortes]

            st.markdown('''
                <style>
                .justified-text {
                    text-align: justify;
                }
                </style>
                <br>
                <div class="justified-text">
                    Os gráficos apresentam as principais causas de acidentes de trânsito e as principais causas 
                    de acidentes relacionados a mortes. No primeiro gráfico, vemos as infrações mais frequentes 
                    que resultam em acidentes, enquanto no segundo gráfico são destacadas as causas que mais resultam 
                    em fatalidades.
                </div>
                <br>
            ''', unsafe_allow_html=True)

            # Criar duas colunas lado a lado
            col1, col2 = st.columns(2)

            # Gráfico: Top 10 Causas de Acidentes
            with col1:

                fig, ax = plt.subplots(figsize=(10, 8))  # Ajustando o tamanho do gráfico
                df_filtrado['causa_acidente'].value_counts().head(10).plot(kind='bar', color='blue', ax=ax)
                
                plt.title('Top 10 Causas de Acidentes')
                plt.xlabel('Causa do Acidente')
                plt.ylabel('Frequência')

                # Aplicar os rótulos quebrados automaticamente no eixo x
                ax.set_xticklabels(wrapped_labels_causas, rotation=45, ha="right")
                
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

            # Gráfico: Top 10 Causas de Acidentes Relacionadas a Mortes
            with col2:

                fig, ax = plt.subplots(figsize=(10, 8))  # Ajustando o tamanho do gráfico
                df_grouped_causa_mortos = df_filtrado.groupby('causa_acidente')['mortos'].sum().sort_values(ascending=False).head(10)
                df_grouped_causa_mortos.plot(kind='bar', color='red', ax=ax)

                plt.title('Top 10 Causas de Acidentes Relacionadas a Mortes')
                plt.xlabel('Causa do Acidente')
                plt.ylabel('Total de Mortos')

                # Aplicar os rótulos quebrados automaticamente no eixo x
                ax.set_xticklabels(wrapped_labels_mortes, rotation=45, ha="right")

                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

        # Gráfico: Top 10 Tipos de Acidente e Gravidade
        elif option2 == 'Top 10 Tipos de Acidente e Gravidade':
            st.markdown("<h3 style='text-align: center;'>Top 10 Tipos de Acidente e Gravidade (Somatório de Mortos e Feridos)</h3>", unsafe_allow_html=True)
            
            # Calcular o total de impacto por tipo de acidente
            df_filtrado['total_impacto'] = df_filtrado['mortos'] + df_filtrado['feridos_graves'] + df_filtrado['feridos_leves']
            df_grouped_tipo_acidente_impacto = df_filtrado.groupby('tipo_acidente')[['mortos', 'feridos_graves', 'feridos_leves', 'total_impacto']].sum().sort_values(by='total_impacto', ascending=False)

            # Definir o comprimento máximo de caracteres por linha
            max_label_length = 20  # Ajuste esse valor conforme necessário

            # Obter os rótulos originais e quebrar automaticamente os longos
            labels = df_grouped_tipo_acidente_impacto.index
            wrapped_labels = [textwrap.fill(label, max_label_length) for label in labels]

            st.markdown('''
                            <style>
                            .justified-text {
                                text-align: justify;
                            }
                            </style>
                            <br>
                            <div class="justified-text">
                                O gráfico apresenta os 10 principais tipos de acidentes de trânsito, classificados de acordo 
                                com a gravidade das consequências, incluindo o número total de mortos, feridos graves e feridos leves. 
                                As categorias estão organizadas com base no tipo de acidente, e as barras representam o somatório 
                                de cada tipo de consequência (mortos, feridos graves e feridos leves), permitindo uma visualização 
                                das diferentes magnitudes de danos causados por cada tipo de ocorrência.
                            </div>
                            <br>
                        ''', unsafe_allow_html=True)

            # Configurar as colunas para limitar a área do gráfico
            col1, col2, col3 = st.columns([1, 5, 1])
            with col2:
                # Gerar o gráfico de barras empilhadas
                fig, ax = plt.subplots(figsize=(10, 6))
                df_grouped_tipo_acidente_impacto[['mortos', 'feridos_graves', 'feridos_leves']].plot(kind='bar', stacked=True, color=['red', 'blue', 'green'], ax=ax)
                
                # Aplicar os novos rótulos com quebra de linha automática
                ax.set_xticklabels(wrapped_labels, rotation=45, ha="right")

                plt.xlabel('Tipo de Acidente')
                plt.ylabel('Total de Mortos / Feridos')
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

        # Gráfico: Gravidade dos Acidentes ao Longo do Dia
        elif option2 == 'Gravidade dos Acidentes ao Longo do Dia':
            st.markdown("<h3 style='text-align: center;'>Gravidade dos Acidentes ao Longo do Dia (Mortos ou Feridos Graves)</h3>", unsafe_allow_html=True)
            
            # Calcular a gravidade dos acidentes ao longo do dia
            data = df_filtrado.groupby('horario')[['mortos', 'feridos_graves']].sum()

            st.markdown('''
                            <style>
                            .justified-text {
                                text-align: justify;
                            }
                            </style>
                            <br>
                            <div class="justified-text">
                               O gráfico apresenta a distribuição da gravidade dos acidentes de trânsito ao longo do dia, 
                                separando os dados entre mortos e feridos graves. O eixo horizontal indica as horas do dia, 
                                enquanto o eixo vertical representa o total de mortos e feridos graves. As duas linhas 
                                traçadas ilustram como esses dois tipos de consequências variam em quantidade conforme a hora do dia, 
                                permitindo uma visualização das faixas horárias com maior ou menor número de ocorrências graves.
                            </div>
                            <br>
                        ''', unsafe_allow_html=True)

            # Configurar as colunas para limitar a área do gráfico
            col1, col2, col3 = st.columns([1, 5, 1])
            with col2:
                # Gerar o gráfico de linha
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(data.index, data['mortos'], marker='o', label='Mortos')
                plt.plot(data.index, data['feridos_graves'], marker='o', label='Feridos Graves')
                plt.xlabel('Hora do Dia')
                plt.ylabel('Total de Mortos / Feridos Graves')
                plt.legend()
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

        # Gráfico: Distribuição de Acidentes por Estado (UF)
        elif option2 == 'Distribuição de Acidentes por Estado (UF)':
            st.markdown("<h3 style='text-align: center;'>Distribuição de Acidentes por Estado (UF)</h3>", unsafe_allow_html=True)

            st.markdown('''
                            <style>
                            .justified-text {
                                text-align: justify;
                            }
                            </style>
                            <br>
                            <div class="justified-text">
                               O gráfico apresentado ilustra a distribuição de acidentes por estado (UF) no Brasil. 
                               No eixo horizontal, estão listadas as unidades federativas (UFs) representando os estados 
                               brasileiros, enquanto o eixo vertical indica o número de acidentes ocorridos em cada estado. 
                               As barras verticais, de cor roxa, representam a quantidade de acidentes por estado, proporcionando 
                               uma visão geral de quais estados possuem os maiores e menores números de incidentes registrados.
                            </div>
                            <br>
                        ''', unsafe_allow_html=True)
            
            # Configurar as colunas para limitar a área do gráfico
            col1, col2, col3 = st.columns([1, 5, 1])
            with col2:
                # Gerar o gráfico de barras para distribuição por estado
                fig, ax = plt.subplots(figsize=(10, 6))
                df_filtrado['uf'].value_counts().plot(kind='bar', color='purple', ax=ax)
                plt.xlabel('Estado (UF)')
                plt.ylabel('Número de Acidentes')
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

        # Gráfico: Top 10 BRs com Mais Acidentes
        elif option2 == 'Top 10 BRs com Mais Acidentes':
            st.markdown("<h3 style='text-align: center;'>Top 10 BRs com Mais Acidentes</h3>", unsafe_allow_html=True)

            st.markdown('''
                            <style>
                            .justified-text {
                                text-align: justify;
                            }
                            </style>
                            <br>
                            <div class="justified-text">
                                O gráfico exibe o "Top 10 BRs com Mais Acidentes", mostrando as 10 rodovias federais (BRs) 
                                com maior número de acidentes registrados. No eixo horizontal, estão representadas as rodovias 
                                identificadas por seus números, enquanto o eixo vertical mostra a quantidade de acidentes ocorridos 
                                em cada uma delas. As barras azuis indicam a frequência de acidentes em cada BR, permitindo uma 
                                comparação visual entre as rodovias mais perigosas em termos de incidentes.
                            </div>
                            <br>
                        ''', unsafe_allow_html=True)

            # Contagem de acidentes por BR
            br_accident_count = df_filtrado['br'].value_counts().reset_index()
            br_accident_count.columns = ['BR', 'count']
            br_accident_count = br_accident_count.sort_values(by='count', ascending=False).head(10)

            # Configurar as colunas para limitar a área do gráfico
            col1, col2, col3 = st.columns([1, 5, 1])
            with col2:
                # Gerar o gráfico de barras para as BRs com mais acidentes
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.bar(br_accident_count['BR'].astype(str), br_accident_count['count'], color='blue')
                plt.xlabel('BR')
                plt.ylabel('Número de Acidentes')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)


    # Aba 3: Mapa de Distribuição de Acidentes
    elif selected_tab == "Mapa com Distribuição de Acidentes":
        # Somente para aba 3
        #st.sidebar.title("Opções - Mapa de Acidentes")
        st.sidebar.write("Visualize o mapa com os acidentes de trânsito.")

        st.markdown("<h3 style='text-align: center;'>Distribuição Geográfica dos Acidentes de Trânsito por Tipo de Acidente</h3>", unsafe_allow_html=True)
        fig = px.scatter_mapbox(df_filtrado, lat="latitude", lon="longitude", hover_name="uf",
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

    # Aba 4: Avaliações da Modelagem
    elif selected_tab == "Avaliações da Modelagem":
        # Somente para aba 4
        option4 = st.sidebar.radio(
            "Selecione a avaliação:",
            ('Matriz de Confusão', 'Curva ROC', 'Importância das Features')
        )

        # Treinamento do modelo
        rf_classifier, scaler, label_encoders, label_encoder, X, y_pred_class, y_test_class = train_model()

        if option4 == 'Matriz de Confusão':
            st.markdown("<h3 style='text-align: center;'>Matriz de Confusão</h3>", unsafe_allow_html=True)

            st.markdown('''
                            <style>
                            .justified-text {
                                text-align: justify;
                            }
                            </style>
                            <br>
                            <div class="justified-text">
                                A partir da matriz de confusão abaixo, é possível inferirmos que o modelo tem um desempenho sólido, 
                                com uma alta acurácia geral (95,25%). Isso pode ser observado principalmente na correta classificação 
                                de acidentes "Com Vítimas Fatais" e "Sem Vítimas". Esses dois grupos possuem números muito baixos de erros, 
                                indicando que o modelo tem uma boa compreensão desses extremos. Entretanto, na classe "Com Vítimas Feridas" 
                                encontra-se o maior desafio do modelo. Embora o número de classificações corretas (8.984) seja alto, 
                                essa classe apresentou 1.320 erros, o que mostra um valor considerável de imprecisão.
                            </div>
                            <br>
                        ''', unsafe_allow_html=True)

            # Configurar as colunas para limitar a área do gráfico
            col1, col2, col3 = st.columns([1, 5, 1])
            with col2:
                # Gerar a matriz de confusão
                conf_matrix = confusion_matrix(y_test_class, y_pred_class)
                
                # Quebra automática dos rótulos das classes para caberem melhor
                max_label_length = 15  # Definir o comprimento máximo de cada linha
                wrapped_labels = [textwrap.fill(label, max_label_length) for label in label_encoder.classes_]

                # Plotar a matriz de confusão com os rótulos ajustados
                fig, ax = plt.subplots(figsize=(6, 3.6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                            xticklabels=wrapped_labels, yticklabels=wrapped_labels, ax=ax)
                
                plt.xlabel("Previsto")
                plt.ylabel("Verdadeiro")
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

        elif option4 == 'Importância das Features':
            st.markdown("<h3 style='text-align: center;'>Importância das Features no Modelo Random Forest</h3>", unsafe_allow_html=True)

            st.markdown('''
                            <style>
                            .justified-text {
                                text-align: justify;
                            }
                            </style>
                            <br>
                            <div class="justified-text">
                                A interpretação do gráfico sugere que as variáveis relacionadas à localização geográfica
                                 (latitude, longitude) e ao tipo de acidente são as mais importantes para o modelo prever 
                                a gravidade dos acidentes. Outros fatores como causa do acidente e horário também têm impacto 
                                significativo. Isso nos permite inferir que fatores tanto geográficos quanto temporais desempenham
                                 um papel crucial na severidade dos acidentes. Essa análise pode guiar futuras investigações e ações 
                                preventivas, focando em áreas de risco ou em causas que aparecem frequentemente associadas a 
                                acidentes graves.
                            </div>
                            <br>
                        ''', unsafe_allow_html=True)

            # Configurar as colunas para limitar a área do gráfico
            col1, col2, col3 = st.columns([1, 5, 1])
            with col2:
                # Gerar gráfico de importância das features
                importances = rf_classifier.feature_importances_
                feature_names = X.columns
                feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', ax=ax)
                plt.xlabel('Importância')
                plt.ylabel('Feature')
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

        elif option4 == 'Curva ROC':
            st.markdown("<h3 style='text-align: center;'>Curva ROC</h3>", unsafe_allow_html=True)

            st.markdown('''
                            <style>
                            .justified-text {
                                text-align: justify;
                            }
                            </style>
                            <br>
                            <div class="justified-text">
                                A curva ROC presente abaixo mostra que o modelo é eficiente para distinguir entre as classes 
                                de acidentes, sendo mais preciso para acidentes "Com Vítimas Fatais". No entanto, para a classe 
                                "Com Vítimas Feridas", há uma leve queda na performance, sugerindo que esse grupo pode ser mais 
                                difícil de classificar corretamente. Isso pode ser usado para ajustar o modelo e melhorar ainda 
                                mais sua precisão para essa classe específica.
                            </div>
                            <br>
                        ''', unsafe_allow_html=True)

            # Configurar as colunas para limitar a área do gráfico
            col1, col2, col3 = st.columns([1, 5, 1])
            with col2:
                # Gerar a curva ROC para cada classe
                y_test_bin = label_binarize(y_test_class, classes=[0, 1, 2])
                y_pred_bin = label_binarize(y_pred_class, classes=[0, 1, 2])

                fig, ax = plt.subplots(figsize=(8, 4.8))
                for i in range(y_test_bin.shape[1]):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, label=f'Classe {label_encoder.classes_[i]} (AUC = {roc_auc:.2f})')

                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Taxa de Falsos Positivos')
                plt.ylabel('Taxa de Verdadeiros Positivos')
                plt.legend(loc="lower right")
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)


    # Aba 5: Previsão do Modelo
    elif selected_tab == "Previsão do Modelo" :
        # Somente para aba 5
        st.sidebar.write("Insira os dados para prever a gravidade dos acidentes.")

    
        st.markdown("<h3 style='text-align: center;'>Previsão do Modelo</h3> ", unsafe_allow_html=True)
        st.markdown("<h6 style='text-align: center;'>Utilize o modelo Random Forest para prever a gravidade dos acidentes</h6>", unsafe_allow_html=True)
        # Carregar o modelo e os encoders
        rf_classifier, scaler, label_encoders, label_encoder, X, y_pred_class, y_test_class = train_model()

        # Formulário para inserção dos dados
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                latitude = st.number_input("Latitude (Ex.: -3,69)", min_value=-90.0, max_value=90.0, step=0.01)
                longitude = st.number_input("Longitude (Ex.: -40,35)", min_value=-180.0, max_value=180.0, step=0.01)
            with col2:
                dia_semana = st.selectbox("Dia da Semana", df_filtrado['dia_semana'].unique())
                horario = st.slider("Horário do Acidente", 0, 23)
            with col3:
                causa_acidente = st.selectbox("Causa do Acidente", df_filtrado['causa_acidente'].unique())
                tipo_acidente = st.selectbox("Tipo de Acidente", df_filtrado['tipo_acidente'].unique())
            submit_button = st.form_submit_button("Fazer Previsão")

        if submit_button:
            # Codificar as variáveis categóricas da entrada do usuário usando o dicionário 'label_encoders'
            dia_semana_encoded = label_encoders['dia_semana'].transform([dia_semana])[0]
            causa_acidente_encoded = label_encoders['causa_acidente'].transform([causa_acidente])[0]
            tipo_acidente_encoded = label_encoders['tipo_acidente'].transform([tipo_acidente])[0]
            
            # Criar o DataFrame com os dados do usuário
            user_data = pd.DataFrame({
                'latitude': [latitude],
                'longitude': [longitude],
                'dia_semana': [dia_semana_encoded],
                'horario': [horario],
                'causa_acidente': [causa_acidente_encoded],
                'tipo_acidente': [tipo_acidente_encoded]
            })

            # Normalizar os dados do usuário
            user_data_normalized = scaler.transform(user_data)

            # Fazer a previsão com o modelo treinado
            prediction = rf_classifier.predict(user_data_normalized)

            # Decodificar a classe prevista
            predicted_class = label_encoder.inverse_transform(prediction)[0]

            # Exibir o resultado da previsão
            st.subheader(f"A previsão do modelo é: {predicted_class}")
        
