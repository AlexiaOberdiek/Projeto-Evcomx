import numpy as np
import pandas as pd
import category_encoders as ce
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================
TARGET = 'temperaturasaidafp'
# Definimos colunas base. O Cluster será adicionado dinamicamente depois.
WOE_COL = ['qualidade']
OHE_COL = ['panela'] 

# Colunas que não entram no X final
COLS_TO_EXCLUDE_FROM_X = ['sugestaomodelolegado', 'temperaturamediareal', 'temperaturaobjetivada', 'Desvio_Legado_Target','C_Medio', 'al_min', 'n_min', 's_min'] 

VAL_INICIO = 7000  
LIMITE_USOU = 5
DELTA_NEG = 5   
DELTA_POS = 10  

# ==============================================================================
# 2. FUNÇÃO DE PERDA (XGBOOST)
# ==============================================================================
def custom_asymmetric_loss(y_true, y_pred):
    resid = y_pred - y_true
    alpha = 1.3
    factor = np.where(resid < 0, alpha, 1.0)
    grad = 2.0 * resid * factor
    hess = 2.0 * factor
    return grad, hess

# ==============================================================================
# 3. CARREGAMENTO E FE
# ==============================================================================
print("--- 1. Carregando Dados ---")
df = pd.read_csv('dados_fundo_do_amanha_evcomx.csv', delimiter=';')
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

# Nota: Não dropamos as colunas químicas ainda, pois precisamos delas pro cluster
cols_to_drop_early = ['acoatual','corrida','secao','c_min','c_max',
                    'velocidadeobjetivada','velocidadereal']
df_clean = df.drop(columns=[c for c in cols_to_drop_early if c in df.columns], errors='ignore')

# ==============================================================================
# 4. SPLIT GOLDEN BATCH
# ==============================================================================
print(f"\n--- 2. Separando Conjuntos ---")

# 1. Validação (Futuro)
df_valid = df_clean.iloc[VAL_INICIO:].copy()

# 2. Treino Total (Passado)
df_treino_total = df_clean.iloc[:VAL_INICIO].copy()

# 3. Filtro Elite
erro_real = df_treino_total['temperaturamediareal'] - df_treino_total['temperaturaobjetivada']
mask_gold = (erro_real >= -DELTA_NEG) & (erro_real <= DELTA_POS)

df_treino_gold = df_treino_total[mask_gold].copy()    # ELITE (Treino)
df_treino_dirty = df_treino_total[~mask_gold].copy()  # SUJOS (Teste)

print(f"Treino ELITE: {len(df_treino_gold)}")
print(f"Dados SUJOS: {len(df_treino_dirty)}")
print(f"Validação: {len(df_valid)}")

# ==============================================================================
# 5. CLUSTERING (TREINADO SÓ NA ELITE)
# ==============================================================================
print("\n--- 3. Criando Clusters Químicos (Baseado na Elite) ---")

# Colunas que definem a química
cols_quimica = ['C_Medio', 'al_min', 'n_min', 's_min']

# 1. Preparar dados da Elite
X_cluster_train = df_treino_gold[cols_quimica].fillna(0)

# 2. Fit Scaler (Só na Elite)
scaler = StandardScaler()
X_cluster_train_scaled = scaler.fit_transform(X_cluster_train)

# 3. Fit K-Means (Só na Elite)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_cluster_train_scaled)

# 4. Aplicar Clusters (Predict) em TODOS os conjuntos
def aplicar_cluster(df_target, nome):
    # Preenche NaN
    X_c = df_target[cols_quimica].fillna(0)
    # Escala usando o scaler da Elite
    X_c_scaled = scaler.transform(X_c)
    # Prevê o cluster usando o K-Means da Elite
    clusters = kmeans.predict(X_c_scaled)
    # Atribui ao DF
    df_target['Cluster_Quimico'] = clusters.astype(str)
    print(f"Clusters aplicados em: {nome}")
    return df_target

df_treino_gold = aplicar_cluster(df_treino_gold, "Elite")
df_treino_dirty = aplicar_cluster(df_treino_dirty, "Sujos")
df_valid = aplicar_cluster(df_valid, "Validação")

# Adiciona o Cluster na lista de colunas para One-Hot Encoding
OHE_COL.append('Cluster_Quimico')

# Agora podemos dropar as colunas químicas originais se quisermos (opcional)
# Vou mantê-las pois vimos que o modelo gosta de C_Medio e Al_min.

# ==============================================================================
# 6. SEPARAÇÃO X/Y
# ==============================================================================
def split_XY(df_part):
    Y = df_part[TARGET].copy()
    X = df_part.drop(columns=[TARGET] + COLS_TO_EXCLUDE_FROM_X, errors='ignore')
    return X, Y

X_elite, Y_elite = split_XY(df_treino_gold)
X_valid, Y_valid = split_XY(df_valid)
X_dirty, Y_dirty = split_XY(df_treino_dirty)

# ==============================================================================
# 7. ENCODING (DELTA STRATEGY)
# ==============================================================================
print("\n--- 4. Aplicando Encoders (Delta) ---")

def encode_target_and_ohe(X_train, y_train, list_X_apply, df_train_full, woe_cols, ohe_cols):
    temp_obj_train = df_train_full.loc[y_train.index, 'temperaturaobjetivada']
    y_delta = y_train - temp_obj_train
    
    # Fit Target Encoder (Delta)
    te = ce.TargetEncoder(cols=woe_cols)
    X_train_te = te.fit_transform(X_train, y_delta) 
    
    # Fit One-Hot (Agora inclui o Cluster_Quimico)
    X_train_final = pd.get_dummies(X_train_te, columns=ohe_cols, drop_first=True)
    
    outputs = [X_train_final]
    colunas_mestras = X_train_final.columns
    
    for X_app in list_X_apply:
        X_app_te = te.transform(X_app.copy())
        X_app_final = pd.get_dummies(X_app_te, columns=ohe_cols, drop_first=True)
        X_app_final = X_app_final.reindex(columns=colunas_mestras, fill_value=0)
        outputs.append(X_app_final)
        
    return outputs

lista_transf = [X_valid, X_dirty]
outputs = encode_target_and_ohe(X_elite, Y_elite, lista_transf, df_treino_gold, WOE_COL, OHE_COL)

X_elite_final, X_valid_final, X_dirty_final = outputs

# ==============================================================================
# 8. TREINO (XGBOOST)
# ==============================================================================
best_params = {
    'n_estimators': 1500, 'learning_rate': 0.10, 'max_depth': 8, 
    'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_weight': 15, 
    'reg_lambda': 0.01, 'alpha': 0.01, 
    "objective": custom_asymmetric_loss, "booster": "gbtree",
    "tree_method": "hist", "device": "cuda", "random_state": 42,
    "verbosity": 0, "n_jobs": -1
}

print("\n--- 5. Treinando Modelo Único (XGBoost) ---")
model_unique = XGBRegressor(**best_params)
model_unique.fit(X_elite_final, Y_elite)

# ==============================================================================
# 10. AVALIAÇÃO E GRÁFICOS
# ==============================================================================
def avaliar_cenario_unico(X_input, df_orig, Y_orig, nome_dataset, usar_vies=False):
    print(f"\n>>> AVALIANDO: {nome_dataset} (Correção: {usar_vies})")
    
    pred_raw = model_unique.predict(X_input)
    
    if usar_vies:
        # pred_final = pred_raw - 1.5 # Opção Manual
        pred_final = pred_raw + 0.75
    else:
        pred_final = pred_raw
        
    diff_legado = df_orig['sugestaomodelolegado'] - df_orig[TARGET]
    erro_legado = (df_orig['temperaturamediareal'] + diff_legado) - df_orig['temperaturaobjetivada']
    
    diff_novo = pred_final - df_orig[TARGET]
    erro_novo = (df_orig['temperaturamediareal'] + diff_novo) - df_orig['temperaturaobjetivada']
    
    df_plot = pd.DataFrame({'Legado': erro_legado, 'Novo': erro_novo})
    
    def categorizar(val):
        if val < -15: return '1. Extremo Frio (< -15)'
        elif val >= -15 and val < -5: return '2. Frio (-15 a -5)'
        elif val >= -5 and val <= 10: return '3. Acerto (-5 a +10)'
        elif val > 10 and val <= 20:  return '4. Quente (+10 a +20)'
        else: return '5. Extremo Calor (> +20)'

    df_plot['Cat_Legado'] = df_plot['Legado'].apply(categorizar)
    df_plot['Cat_Novo'] = df_plot['Novo'].apply(categorizar)
    
    stats_legado = df_plot['Cat_Legado'].value_counts(normalize=True) * 100
    stats_novo = df_plot['Cat_Novo'].value_counts(normalize=True) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    sns.histplot(df_plot['Legado'], color='red', label='Legado', kde=True, ax=axes[0], alpha=0.3, element="step")
    sns.histplot(df_plot['Novo'], color='purple', label='XGBoost', kde=True, ax=axes[0], alpha=0.3, element="step")
    axes[0].axvline(-5, color='black', linestyle='--'); axes[0].axvline(10, color='black', linestyle='--')
    axes[0].set_title(f"Distribuição: {nome_dataset}"); axes[0].legend()
    
    categorias = ['1. Extremo Frio (< -15)', '2. Frio (-15 a -5)', '3. Acerto (-5 a +10)', '4. Quente (+10 a +20)', '5. Extremo Calor (> +20)']
    vals_leg = [stats_legado.get(c, 0) for c in categorias]
    vals_nov = [stats_novo.get(c, 0) for c in categorias]
    x = np.arange(len(categorias))
    
    bars1 = axes[1].bar(x - 0.17, vals_leg, 0.35, label='Legado', color='red', alpha=0.7)
    bars2 = axes[1].bar(x + 0.17, vals_nov, 0.35, label='XGBoost', color='purple', alpha=0.7)
    
    axes[1].set_title(f"Comparação - {nome_dataset}")
    axes[1].set_xticks(x); axes[1].set_xticklabels(['EXT. FRIO', 'Frio', 'ACERTO', 'Quente', 'EXT. CALOR'], fontsize=10)
    axes[1].bar_label(bars1, fmt='%.1f%%', padding=3); axes[1].bar_label(bars2, fmt='%.1f%%', padding=3)
    axes[1].legend(); plt.tight_layout(); plt.show()
    
    print(f"--- Resumo: {nome_dataset} ---")
    print(pd.DataFrame({'% Legado': vals_leg, '% Novo': vals_nov}, index=categorias).round(2))
    print("="*60)

avaliar_cenario_unico(X_elite_final, df_treino_gold, Y_elite, "1. TREINO ELITE ", False)
avaliar_cenario_unico(X_dirty_final, df_treino_dirty, Y_dirty, "2. DADOS SUJOS ", True)
avaliar_cenario_unico(X_valid_final, df_valid, Y_valid, "3. VALIDAÇÃO FUTURA ", True)

# ==============================================================================
# 11. DIAGNÓSTICO (VIF E IMPORTÂNCIA)
# ==============================================================================
print("\n--- 6. Diagnóstico de Features ---")

# Importância
importances = model_unique.feature_importances_
try: feature_names = model_unique.get_booster().feature_names
except: feature_names = X_elite_final.columns

df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
# Mostra top 20 para ver onde o Cluster aparece
sns.barplot(x='Importance', y='Feature', data=df_imp.head(20), palette='magma') 
plt.title("Feature Importance (XGBoost + Clusters)")
plt.tight_layout(); plt.show()

# VIF (Só pra confirmar se os clusters não estouraram a colinearidade)
df_vif = X_elite_final.copy().dropna().select_dtypes(include=[np.number])
# Pegamos uma amostra se for muito grande pra não demorar
if len(df_vif) > 5000: df_vif = df_vif.sample(5000) 
X_vif = add_constant(df_vif)
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
vif_data = vif_data[vif_data['Feature'] != 'const'].sort_values(by='VIF', ascending=False)

print("\n=== VIF (Top 10) ===")
print(vif_data.head(10))