import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from copy import deepcopy
from sklearn.base import clone
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================
TARGET = 'temperaturasaidafp'
WOE_COL = ['qualidade']
OHE_COL = ['panela']
# Colunas ignoradas no treino
COLS_TO_EXCLUDE_FROM_X = ['sugestaomodelolegado', 'temperaturamediareal', 'temperaturaobjetivada','Desvio_Legado_Target','C_Medio', 'al_min', 'n_min', 's_min']

VAL_INICIO = 7000
LIMITE_USOU = 5
DELTA_NEG = 5
DELTA_POS = 10

# ==============================================================================
# 2. CARREGAMENTO E PREPARAÇÃO
# ==============================================================================
print("--- 1. Carregando Dados ---")
df = pd.read_csv('dados_fundo_do_amanha_evcomx.csv', delimiter=';')

# REMOÇÃO DE DUPLICATAS
df = df.drop_duplicates(subset=['corrida'], keep='first')

num_cols = ['al_min','c_min','c_max','n_min','s_min','sequencia','sequenciatotal',
            'vidapanela','tempociclo','tempovacuototal','temperaturaliquidus',
            'velocidadeobjetivada','velocidadereal','sugestaomodelolegado',
            'temperaturasaidafp','temperaturaobjetivada','temperaturamediareal']
for c in num_cols:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

id_cols = ['corrida','secao','acoatual','qualidade','panela']
for c in id_cols:
    if c in df.columns: df[c] = df[c].fillna('MISSING').astype(str)

df['C_Medio'] = (df['c_max'] + df['c_min']) / 2
df['T_Liquid_Desvio'] = df['temperaturaobjetivada'] - df['temperaturaliquidus']

# ==============================================================================
# 3.1 CLUSTERING QUÍMICO (A SUA SOLUÇÃO PARA DESCONHECIDOS)
# ==============================================================================
print("--- Criando Clusters Químicos (Inteligência para Aços Novos) ---")

# Selecionamos as colunas que definem "O que é o aço" (Física/Química)
cols_quimica = ['C_Medio', 'al_min', 'n_min', 's_min']

# Preenche NaNs com a média para o K-Means não quebrar
X_cluster = df[cols_quimica].fillna(df[cols_quimica].mean())

# Padroniza os dados (K-Means precisa de escala igual)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Cria 10 Grupos de Aços (Você pode testar 5, 10, 20...)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster_Quimico'] = kmeans.fit_predict(X_scaled)

# Transforma em categoria para o modelo entender que é um "Grupo"
df['Cluster_Quimico'] = df['Cluster_Quimico'].astype(str)

print(f"Clusters criados: {df['Cluster_Quimico'].nunique()} grupos de comportamento químico.")

# Adiciona o Cluster nas colunas a serem codificadas (One-Hot é bom aqui pois são poucos grupos)
OHE_COL.append('Cluster_Quimico') 

# ==============================================================================
# ... (O RESTO DO CÓDIGO CONTINUA IGUAL: DROP, SPLIT, ENCODER, TREINO) ...
# ==============================================================================
cols_to_drop_early = ['acoatual','corrida','secao','c_min','c_max', 'al_min','s_min'
                      'temperaturaliquidus','velocidadeobjetivada','velocidadereal']
df_clean = df.drop(columns=[c for c in cols_to_drop_early if c in df.columns], errors='ignore')

# ==============================================================================
# 3. SPLIT GOLDEN BATCH (ELITE)
# ==============================================================================
print(f"\n--- 2. Separando Conjuntos (Estratégia Elite) ---")

# 1. Validação (Futuro)
df_valid = df_clean.iloc[VAL_INICIO:].copy()

# 2. Treino Total (Passado)
df_treino_total = df_clean.iloc[:VAL_INICIO].copy()

# 3. Filtro Elite (Golden Batch)
# Critério: O Resultado REAL ficou bom? (Independente do Legado)
erro_operacional_real = df_treino_total['temperaturamediareal'] - df_treino_total['temperaturaobjetivada']

# Máscara: Sucesso Real (-5 a +10)
mask_gold = (erro_operacional_real >= -DELTA_NEG) & (erro_operacional_real <= DELTA_POS)

df_treino_gold = df_treino_total[mask_gold].copy()    # ELITE (Treino)
df_treino_dirty = df_treino_total[~mask_gold].copy()  # SUJOS (Teste)

print(f"Treino ELITE (Sucessos Reais): {len(df_treino_gold)}")
print(f"Dados SUJOS (Erros Reais): {len(df_treino_dirty)}")
print(f"Validação (Futuro): {len(df_valid)}")

# Função auxiliar para separar X e Y
def split_XY(df_part):
    Y = df_part[TARGET].copy()
    X = df_part.drop(columns=[TARGET] + COLS_TO_EXCLUDE_FROM_X, errors='ignore')
    # Remove colunas auxiliares que possam ter sobrado
    X = X.drop(columns=['usage'], errors='ignore')
    return X, Y

# Criação dos Datasets
X_elite, Y_elite = split_XY(df_treino_gold)
X_valid, Y_valid = split_XY(df_valid)
X_dirty, Y_dirty = split_XY(df_treino_dirty)

# ==============================================================================
# 4. ENCODING (OBRIGATÓRIO PARA RANDOM FOREST)
# ==============================================================================
print("\n--- 3. Aplicando Encoders (Estratégia Delta) ---")

def encode_target_and_ohe(X_train, y_train, list_X_apply, df_train_full, woe_cols, ohe_cols):
    # 1. CRIAR O TARGET "DELTA" (Real - Objetivo)
    temp_obj_train = df_train_full.loc[y_train.index, 'temperaturaobjetivada']
    y_delta = y_train - temp_obj_train

    # 2. FIT NO DELTA
    te = ce.TargetEncoder(cols=woe_cols)
    X_train_te = te.fit_transform(X_train, y_delta)
    X_train_final = pd.get_dummies(X_train_te, columns=ohe_cols, drop_first=True)

    # 3. APLICA NOS OUTROS
    outputs = [X_train_final]
    colunas_mestras = X_train_final.columns

    for X_app in list_X_apply:
        X_app_te = te.transform(X_app.copy())
        X_app_final = pd.get_dummies(X_app_te, columns=ohe_cols, drop_first=True)
        X_app_final = X_app_final.reindex(columns=colunas_mestras, fill_value=0)
        outputs.append(X_app_final)

    return outputs

# Lista de datasets para transformar
lista_transf = [X_valid, X_dirty]

outputs = encode_target_and_ohe(
    X_elite, Y_elite,
    lista_transf,
    df_treino_gold, # DF completo do Treino (pra pegar a Temp Objetivo)
    WOE_COL, OHE_COL
)

X_elite_final = outputs[0]  # Treino pronto
X_valid_final = outputs[1]  # Validação pronta
X_dirty_final = outputs[2]  # Dados Sujos prontos

# ==============================================================================
# 5. TREINO (RANDOM FOREST)
# ==============================================================================
best_params = {
    'n_estimators': 1500, # Um pouco mais de árvores para compensar a unificação
    'learning_rate': 0.10,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 15,
    'reg_lambda': 0.01,
    'alpha': 0.01,
    "booster": "dart",
    "tree_method": "hist", "device": "cuda", "random_state": 42,
    "verbosity": 0, "n_jobs": -1
}

base_models = [

    # 2. LightGBM
    ("lgbm", LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        num_leaves=31,
        random_state=42
    )),

    # 3. CatBoost
    ("cat", CatBoostRegressor(
        depth=8,
        iterations=2000,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=False,
        random_seed=42
    )),

    # 4. XGBoost
    ("xgb", XGBRegressor(**best_params)),

    # 5. KNN (normalizado)
    ("knn", Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(
            n_neighbors=15,
            weights="distance",
            metric="manhattan"
        ))
    ])),

    # 6. Gaussian Process Regressor (suaviza previsões)
    ("gpr", GaussianProcessRegressor(
        kernel=RBF(length_scale=1.0),
        alpha=1.0,
        normalize_y=True
    ))
]

# ============================================================
# 2. META-MODELO (FINAL ESTIMATOR)
# ============================================================

# META-MODELO — LightGBM funciona melhor com previsões baseadas em sensores
model_unique = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42
)

model_unique.fit(X_elite_final, Y_elite)

# ==============================================================================
# 7. FUNÇÃO DE AVALIAÇÃO DETALHADA (5 FAIXAS)
# ==============================================================================
def avaliar_cenario_unico(X_input, df_orig, Y_orig, nome_dataset, usar_vies=False):
    print(f"\n>>> AVALIANDO: {nome_dataset} (Correção: {usar_vies})")

    # 1. Predição Única
    pred_raw = model_unique.predict(X_input)
    # 2. Aplicação do Viés
    if usar_vies:
        # Se quiser testar manual: pred_final = pred_raw - 1.5
        pred_final = pred_raw +0.75
    else:
        pred_final = pred_raw

    # 3. Cálculo dos Erros (Fórmula Complexa)
    # Legado
    diff_legado = df_orig['sugestaomodelolegado'] - df_orig[TARGET]
    erro_legado = (df_orig['temperaturamediareal'] + diff_legado) - df_orig['temperaturaobjetivada']

    # Novo Modelo
    diff_novo = pred_final - df_orig[TARGET]
    erro_novo = (df_orig['temperaturamediareal'] + diff_novo) - df_orig['temperaturaobjetivada']

    # 4. Categorização (5 Faixas)
    df_plot = pd.DataFrame({'Legado': erro_legado, 'Novo': erro_novo})

    def categorizar(val):
        if val < -15:           return '1. Extremo Frio (< -15)'
        elif val >= -15 and val < -5: return '2. Frio (-15 a -5)'
        elif val >= -5 and val <= 10: return '3. Acerto (-5 a +10)'
        elif val > 10 and val <= 20:  return '4. Quente (+10 a +20)'
        else:                   return '5. Extremo Calor (> +20)'

    df_plot['Cat_Legado'] = df_plot['Legado'].apply(categorizar)
    df_plot['Cat_Novo'] = df_plot['Novo'].apply(categorizar)

    stats_legado = df_plot['Cat_Legado'].value_counts(normalize=True) * 100
    stats_novo = df_plot['Cat_Novo'].value_counts(normalize=True) * 100

    # 5. Plotagem
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # --- GRÁFICO 1: HISTOGRAMA ---
    sns.histplot(df_plot['Legado'], color='red', label='Legado', kde=True, ax=axes[0], alpha=0.3, element="step")
    sns.histplot(df_plot['Novo'], color='green', label='Random Forest', kde=True, ax=axes[0], alpha=0.3, element="step")

    axes[0].axvline(-5, color='black', linestyle='--', linewidth=2, label='Meta (-5)')
    axes[0].axvline(10, color='black', linestyle='--', linewidth=2, label='Meta (+10)')
    axes[0].axvline(-10, color='red', linestyle=':', linewidth=2)
    axes[0].axvline(20, color='red', linestyle=':', linewidth=2)

    axes[0].set_title(f"Distribuição de Erros - {nome_dataset}")
    axes[0].set_xlabel("Desvio do Objetivo (°C)")
    axes[0].legend()

    # --- GRÁFICO 2: BARRAS (5 CATEGORIAS) ---
    categorias = [
        '1. Extremo Frio (< -15)',
        '2. Frio (-15 a -5)',
        '3. Acerto (-5 a +10)',
        '4. Quente (+10 a +20)',
        '5. Extremo Calor (> +20)'
    ]

    vals_leg = [stats_legado.get(c, 0) for c in categorias]
    vals_nov = [stats_novo.get(c, 0) for c in categorias]
    x = np.arange(len(categorias))

    bars1 = axes[1].bar(x - 0.17, vals_leg, 0.35, label='Legado', color='red', alpha=0.7)
    bars2 = axes[1].bar(x + 0.17, vals_nov, 0.35, label='Random Forest', color='green', alpha=0.7)

    axes[1].set_title(f"Comparação Detalhada - {nome_dataset}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['EXT. FRIO', 'Frio', 'ACERTO', 'Quente', 'EXT. CALOR'], fontsize=10)

    axes[1].bar_label(bars1, fmt='%.1f%%', padding=3, fontsize=9)
    axes[1].bar_label(bars2, fmt='%.1f%%', padding=3, fontsize=9)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # 6. Tabela Resumo
    resumo = pd.DataFrame({'% Legado': vals_leg, '% Novo': vals_nov}, index=categorias)
    print(f"--- Resumo Numérico: {nome_dataset} ---")
    print(resumo.round(2))
    print("="*60)

# ==============================================================================
# 8. EXECUÇÃO
# ==============================================================================

# 1. ELITE (Aprendizado Puro - Sem Viés)
avaliar_cenario_unico(X_elite_final, df_treino_gold, Y_elite,
                      "1. TREINO ELITE (Sem Viés)", usar_vies=False)

# 2. DADOS SUJOS (Recuperação - Com Viés)
avaliar_cenario_unico(X_dirty_final, df_treino_dirty, Y_dirty,
                      "2. DADOS SUJOS (Com Correção)", usar_vies=False)

# 3. VALIDAÇÃO FUTURA (Teste Real - Com Viés)
avaliar_cenario_unico(X_valid_final, df_valid, Y_valid,
                      "3. VALIDAÇÃO FUTURA (Com Correção)", usar_vies=True)

