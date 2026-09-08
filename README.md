# Projeto Evcomx

Projeto de machine learning aplicado ao contexto industrial da Evcomx, com foco na previsão da temperatura de saída do processo produtivo. O objetivo principal é modelar a variável alvo `temperaturasaidafp` com base em dados operacionais, qualidade do material, características da panela e outros indicadores de processo.

## Visão geral

Este repositório reúne scripts e notebooks usados para:

- carregar e limpar dados históricos de produção;
- gerar features e variáveis derivadas do processo;
- separar conjuntos de treino, validação e dados considerados "elite";
- aplicar encoders para variáveis categóricas;
- treinar modelos de regressão e ensemble para prever temperatura;
- avaliar o erro de previsão com métricas como RMSE.

A proposta é auxiliar a análise operacional, identificar desvios de processo e melhorar a previsibilidade da temperatura final em cenários industriais.

## Objetivo do projeto

Predizer a temperatura de saída da linha de produção, usando informações como:

- qualidade do aço;
- características da panela;
- sequência do processo;
- variáveis químicas e operacionais;
- temperatura objetiva e outros indicadores relacionados.

A modelagem busca reduzir a diferença entre a temperatura prevista e a temperatura real observada, permitindo melhor controle e tomada de decisão na operação.

## Estrutura do repositório

- `dados_fundo_do_amanha_evcomx.csv` — base de dados principal em CSV com dados do processo.
- `randomforest.py` — implementação com Random Forest e ensemble de modelos.
- `modelstackingultimate.py` — versão com stacking e modelos mais robustos (XGBoost, LightGBM, CatBoost, KNN, GPR).
- `NotebookOrganizado.ipynb` — notebook organizado do fluxo de análise e modelagem.
- `Projeto_Evcomx.ipynb` — notebook principal do projeto.
- `StackingModel.ipynb` e outros notebooks — experimentações com diferentes abordagens.
- `FazendoRandomForest.py`, `sorandomforest.py`, `usingcatboost.py`, `ULTIMATEXGBOOST.py` — scripts com variações de treinamento e avaliação.

## Dados

A base contém registros operacionais do processo industrial, incluindo colunas como:

- `corrida`
- `secao`
- `qualidade`
- `panela`
- `temperaturasaidafp` (alvo)
- `temperaturaobjetivada`
- `temperaturamediareal`
- `velocidadereal`
- `velocidadeobjetivada`
- `temperaturaliquidus`
- `sequencia`
- `tempociclo`
- `c_min`, `c_max`, `al_min`, `n_min`, `s_min`

O projeto realiza tratamentos como:

- remoção de duplicatas;
- conversão de colunas numéricas;
- criação de features derivadas;
- codificação de variáveis categóricas;
- seleção de subconjuntos de dados para treino e validação.

## Pipeline de modelagem

O fluxo principal inclui:

1. Carregamento dos dados
2. Limpeza e pré-processamento
3. Criação de features derivadas
4. Separação dos dados em treino e validação
5. Filtragem de dados de alta qualidade para treinamento (estratégia "elite")
6. Codificação de categorias com TargetEncoder e OneHotEncoder
7. Treinamento de algoritmos de regressão
8. Avaliação do modelo por erro e comportamento em diferentes cenários

## Modelos utilizados

Os scripts exploram diferentes abordagens, incluindo:

- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost
- KNN
- Gaussian Process Regressor
- Stacking Regressor

A ideia é comparar desempenho e combinar previsões em ensembles para reduzir variabilidade e melhorar a robustez do modelo.

## Requisitos

Para executar o projeto, recomenda-se:

- Python 3.9+
- pandas
- numpy
- scikit-learn
- category_encoders
- matplotlib
- seaborn
- xgboost
- lightgbm
- catboost
- statsmodels

### Instalação

```bash
pip install pandas numpy scikit-learn category_encoders matplotlib seaborn xgboost lightgbm catboost statsmodels
```

Se necessário, também pode ser usado um ambiente com Jupyter Notebook para rodar os arquivos `.ipynb`.

## Como executar

### Opção 1: rodar o script principal de Random Forest

```bash
python randomforest.py
```

### Opção 2: rodar a versão de ensemble/stacking

```bash
python modelstackingultimate.py
```

### Opção 3: abrir os notebooks

Abra os arquivos `.ipynb` no Jupyter Notebook ou VS Code com suporte a notebooks para explorar o processo passo a passo.

## Métricas e avaliação

Os scripts fazem uso de métricas como:

- RMSE (Root Mean Squared Error)
- comparação do erro de previsão em dados de validação
- análise de desempenho por cenário e segmentação de dados

## Observações

- O projeto foi desenvolvido em contexto acadêmico/experimental, com foco em aprendizado e experimentação de modelagem preditiva.
- A base de dados e os parâmetros podem ser ajustados para otimizar resultado de acordo com o cenário industrial real.
- A implementação central da modelagem envolve feature engineering, encoding e combinação de modelos.

## Futuras melhorias

- comparar resultados com validação temporal mais estrita;
- testar tuning de hiperparâmetros com GridSearch ou Optuna;
- analisar importância das variáveis;
- automatizar a geração de relatório de desempenho;
- integrar a previsão em um fluxo de decisão operacional.

## Contribuição

Este é um projeto de estudo e experimentação. Sugestões de melhoria, ajustes na modelagem e novas métricas são bem-vindas.

## Licença

O projeto é mantido para fins educacionais e de pesquisa aplicada, sem uma licença comercial específica definida no repositório.
