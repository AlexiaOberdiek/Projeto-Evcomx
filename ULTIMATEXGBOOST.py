import numpy as np
import pandas as pd
import category_encoders as ce
from xgboost import XGBRegressor
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
WOE_COL = ['qualidade']
OHE_COL = ['panela']

# IMPORTANTE: removemos qualquer referência ao legado nas features
COLS_TO_EXCLUDE_FROM_X = [
    'sugestaomodelolegado', 
    'temperaturamediareal', 
    'temperaturaobjetivada',
    'Desvio_Legado_Target' # Garantindo que não entra
] 

VAL_INICIO = 7000  
DELTA_NEG = 5   
DELTA_POS = 10  

# ==============================================================================
# 2. FUNÇÃO DE PERDA ASSIMÉTRICA
# ==============================================================================
def custom_asymmetric_loss(y_true, y_pred):
    resid = y_pred - y_true
    alpha = 1.3
    factor = np.where(resid < 0, alpha, 1.0)
    grad = 2.0 * resid * factor
    hess = 2.0 * factor
    return grad, hess

# ==============================================================================
# 3. CARREGAMENTO E FEATURE ENGINEERING (AUTÔNOMO)
# ==============================================================================
print("--- 1. Carregando Dados ---")
df = pd.read_csv('dados_fundo_do_amanha_evcomx.csv', delimiter=';')

# Remoção de Duplicatas
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

# FE (Apenas dados físicos/químicos, sem olhar pro legado)
df['C_Medio'] = (df['c_max'] + df['c_min']) / 2
df['T_Liquid_Desvio'] = df['temperaturaobjetivada'] - df['temperaturaliquidus']
# REMOVIDO: df['Desvio_Legado_Target'] (Para criar independência)
df['Tempo_Sequencia'] = df['tempociclo'] * df['sequencia']

cols_to_drop_early = ['acoatual','corrida','secao','c_min','c_max','s_min',
                      'temperaturaliquidus','velocidadeobjetivada','velocidadereal']
df_clean = df.drop(columns=[c for c in cols_to_drop_early if c in df.columns], errors='ignore')

# ==============================================================================
# 4. SPLIT GOLDEN BATCH (CRITÉRIO REAL)
# ==============================================================================
print(f"\n--- 2. Separando Conjuntos (Estratégia Elite Independente) ---")

# 1. Validação (Futuro)
df_valid = df_clean.iloc[VAL_INICIO:].copy()

# 2. Treino Total (Passado)
df_treino_total = df_clean.iloc[:VAL_INICIO].copy()

# 3. Filtro Elite (Golden Batch)
# Critério: O Resultado REAL da fábrica bateu com o OBJETIVO?
# Isso independe se o legado ajudou ou atrapalhou. Queremos aprender o sucesso.
erro_operacional_real = df_treino_total['temperaturamediareal'] - df_treino_total['temperaturaobjetivada']

# Máscara: Sucesso Real (-5 a +10)
mask_gold = (erro_operacional_real >= -DELTA_NEG) & (erro_operacional_real <= DELTA_POS)

df_treino_gold = df_treino_total[mask_gold].copy()    # ELITE
df_treino_dirty = df_treino_total[~mask_gold].copy()  # SUJOS

print(f"Treino ELITE (Sucessos Reais): {len(df_treino_gold)}")
print(f"Dados SUJOS (Erros Reais): {len(df_treino_dirty)}")
print(f"Validação (Futuro): {len(df_valid)}")

# Função auxiliar para separar X e Y
def split_XY(df_part):
    Y = df_part[TARGET].copy()
    X = df_part.drop(columns=[TARGET] + COLS_TO_EXCLUDE_FROM_X, errors='ignore')
    return X, Y

# Criação dos Datasets
X_elite, Y_elite = split_XY(df_treino_gold)
X_valid, Y_valid = split_XY(df_valid)
X_dirty, Y_dirty = split_XY(df_treino_dirty)

# ==============================================================================
# 5. ENCODING (DELTA STRATEGY)
# ==============================================================================
print("\n--- 3. Aplicando Encoders (Estratégia Delta) ---")

def encode_target_and_ohe(X_train, y_train, list_X_apply, df_train_full, woe_cols, ohe_cols):
    # 1. Target = Delta (Real - Objetivo)
    temp_obj_train = df_train_full.loc[y_train.index, 'temperaturaobjetivada']
    y_delta = y_train - temp_obj_train
    
    # 2. Fit no Delta
    te = ce.TargetEncoder(cols=woe_cols)
    X_train_te = te.fit_transform(X_train, y_delta) 
    X_train_final = pd.get_dummies(X_train_te, columns=ohe_cols, drop_first=True)
    
    # 3. Transform nos outros
    outputs = [X_train_final]
    colunas_mestras = X_train_final.columns
    
    for X_app in list_X_apply:
        X_app_te = te.transform(X_app.copy())
        X_app_final = pd.get_dummies(X_app_te, columns=ohe_cols, drop_first=True)
        X_app_final = X_app_final.reindex(columns=colunas_mestras, fill_value=0)
        outputs.append(X_app_final)
        
    return outputs

lista_transf = [X_valid, X_dirty]

outputs = encode_target_and_ohe(
    X_elite, Y_elite, 
    lista_transf, 
    df_treino_gold, # DF completo para pegar a temp objetivada
    WOE_COL, OHE_COL
)

X_elite_final = outputs[0]
X_valid_final = outputs[1]
X_dirty_final = outputs[2]

# ==============================================================================
# 6. TREINO (XGBOOST AUTÔNOMO)
# ==============================================================================
best_params = {
    'n_estimators': 1500, 
    'learning_rate': 0.10, 
    'max_depth': 8, 
    'subsample': 0.7, 
    'colsample_bytree': 0.7, 
    'min_child_weight': 15, 
    'reg_lambda': 0.01, 
    'alpha': 0.01, 
    "objective": custom_asymmetric_loss, 
    "booster": "gbtree",
    "tree_method": "hist", "device": "cuda", "random_state": 42,
    "verbosity": 0, "n_jobs": -1
}

print("\n--- 4. Treinando Modelo Único (XGBoost) ---")
model_unique = XGBRegressor(**best_params)
model_unique.fit(X_elite_final, Y_elite)

# ==============================================================================
# 8. FUNÇÃO DE AVALIAÇÃO COMPARATIVA (NOVO vs LEGADO)
# ==============================================================================
def avaliar_cenario_unico(X_input, df_orig, Y_orig, nome_dataset, usar_vies=False):
    print(f"\n>>> AVALIANDO: {nome_dataset} (Correção: {usar_vies})")
    
    # 1. Predição do Novo Modelo
    pred_raw = model_unique.predict(X_input)
    
    if usar_vies:
        pred_final = pred_raw -1
    else:
        pred_final = pred_raw
        
    # 2. Cálculo dos Erros
    # Legado (Fórmula: SugestãoLegado - ObjetivoReal)
    # Precisamos reconstruir o que aconteceu:
    # ErroLegado = (TempMediaReal + (SugestãoLegado - TempSaidaReal)) - Objetivo
    diff_legado = df_orig['sugestaomodelolegado'] - df_orig[TARGET]
    erro_legado = (df_orig['temperaturamediareal'] + diff_legado) - df_orig['temperaturaobjetivada']
    
    # Novo Modelo
    diff_novo = pred_final - df_orig[TARGET]
    erro_novo = (df_orig['temperaturamediareal'] + diff_novo) - df_orig['temperaturaobjetivada']
    
    # 3. Categorização
    df_plot = pd.DataFrame({'Legado': erro_legado, 'Novo': erro_novo})
    
    def categorizar(val):
        if val < -15:           return '1. Extremo Frio (< -15)'
        elif val >= -15 and val < -5: return '2. Frio (-15 a -5)'
        elif val >= -5 and val <= 10: return '3. Acerto (-5 a +10)'
        elif val > 10 and val <= 25:  return '4. Quente (+10 a +25)'
        else:                   return '5. Extremo Calor (> +25)'

    df_plot['Cat_Legado'] = df_plot['Legado'].apply(categorizar)
    df_plot['Cat_Novo'] = df_plot['Novo'].apply(categorizar)
    
    stats_legado = df_plot['Cat_Legado'].value_counts(normalize=True) * 100
    stats_novo = df_plot['Cat_Novo'].value_counts(normalize=True) * 100
    
    # 4. Plotagem
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    # Histograma
    sns.histplot(df_plot['Legado'], color='red', label='Legado', kde=True, ax=axes[0], alpha=0.3, element="step")
    sns.histplot(df_plot['Novo'], color='purple', label='XGBoost', kde=True, ax=axes[0], alpha=0.3, element="step")
    axes[0].axvline(-5, color='green', linestyle='--', label='Meta')
    axes[0].axvline(10, color='green', linestyle='--')
    axes[0].axvline(-10, color='black', linestyle=':')
    axes[0].axvline(20, color='black', linestyle=':')
    axes[0].set_title(f"Distribuição de Erros - {nome_dataset}")
    axes[0].set_xlabel("Erro em relação ao Objetivo (°C)")
    axes[0].legend()
    
    # Barras
    categorias = [
        '1. Extremo Frio (< -15)', '2. Frio (-15 a -5)', 
        '3. Acerto (-5 a +10)', 
        '4. Quente (+10 a +25)', '5. Extremo Calor (> +25)'
    ]
    vals_leg = [stats_legado.get(c, 0) for c in categorias]
    vals_nov = [stats_novo.get(c, 0) for c in categorias]
    x = np.arange(len(categorias))
    
    bars1 = axes[1].bar(x - 0.17, vals_leg, 0.35, label='Legado', color='red', alpha=0.7)
    bars2 = axes[1].bar(x + 0.17, vals_nov, 0.35, label='XGBoost', color='purple', alpha=0.7)
    
    axes[1].set_title(f"Comparação Detalhada - {nome_dataset}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['EXT. FRIO', 'Frio', 'ACERTO', 'Quente', 'EXT. CALOR'], fontsize=10)
    axes[1].bar_label(bars1, fmt='%.1f%%', padding=3, fontsize=9)
    axes[1].bar_label(bars2, fmt='%.1f%%', padding=3, fontsize=9)
    axes[1].legend()
    
    plt.tight_layout(); plt.show()
    
    resumo = pd.DataFrame({'% Legado': vals_leg, '% Novo': vals_nov}, index=categorias)
    print(resumo.round(2))
    print("="*60)

# ==============================================================================
# 9. EXECUÇÃO
# ==============================================================================

avaliar_cenario_unico(X_elite_final, df_treino_gold, Y_elite, 
                      "1. TREINO ELITE (Sem Viés)", usar_vies=False)

avaliar_cenario_unico(X_dirty_final, df_treino_dirty, Y_dirty, 
                      "2. DADOS SUJOS (Com Correção)", usar_vies=True)

avaliar_cenario_unico(X_valid_final, df_valid, Y_valid, 
                      "3. VALIDAÇÃO FUTURA (Com Correção)", usar_vies=True)

# ==============================================================================
# 10. DIAGNÓSTICO (VIF E IMPORTÂNCIA)
# ==============================================================================
print("\n--- 5. Diagnóstico de Features ---")

# VIF
df_vif = X_elite_final.copy().dropna().select_dtypes(include=[np.number])
X_vif = add_constant(df_vif)
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
vif_data = vif_data[vif_data['Feature'] != 'const'].sort_values(by='VIF', ascending=False)

print("\n=== VIF (Multicolinearidade) ===")
print(vif_data.head(10))

# Importância
importances = model_unique.feature_importances_
# Recuperar nomes do XGBoost se possível, senão usa do DF
try: feature_names = model_unique.get_booster().feature_names
except: feature_names = X_elite_final.columns

df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=df_imp.head(15), palette='magma')
plt.title("Top 15 Features mais Importantes (XGBoost Autônomo)")
plt.tight_layout(); plt.show()