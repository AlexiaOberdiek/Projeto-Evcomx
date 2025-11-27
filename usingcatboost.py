import numpy as np
import pandas as pd
import category_encoders as ce
from catboost import CatBoostRegressor
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
COLS_TO_EXCLUDE_FROM_X = ['sugestaomodelolegado', 'temperaturamediareal', 'temperaturaobjetivada', 'Desvio_Legado_Target'] 

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
df['Tempo_Sequencia'] = df['tempociclo'] * df['sequencia']

cols_to_drop_early = ['acoatual','corrida','secao','c_min','c_max','s_min',
                      'temperaturaliquidus','velocidadeobjetivada','velocidadereal']
df_clean = df.drop(columns=[c for c in cols_to_drop_early if c in df.columns], errors='ignore')

# ==============================================================================
# 3. SPLIT GOLDEN BATCH (CRITÉRIO REAL)
# ==============================================================================
print(f"\n--- 2. Separando Conjuntos (Estratégia Elite) ---")

# 1. Validação (Futuro)
df_valid = df_clean.iloc[VAL_INICIO:].copy()

# 2. Treino Total (Passado)
df_treino_total = df_clean.iloc[:VAL_INICIO].copy()

# 3. Filtro Elite (Golden Batch)
# Critério: Sucesso Operacional REAL (Real - Objetivo)
erro_operacional_real = df_treino_total['temperaturamediareal'] - df_treino_total['temperaturaobjetivada']
mask_gold = (erro_operacional_real >= -DELTA_NEG) & (erro_operacional_real <= DELTA_POS)

df_treino_gold = df_treino_total[mask_gold].copy()    # ELITE (Treino)
df_treino_dirty = df_treino_total[~mask_gold].copy()  # SUJOS (Teste)

print(f"Treino ELITE (Sucessos Reais): {len(df_treino_gold)}")
print(f"Dados SUJOS (Erros Reais): {len(df_treino_dirty)}")
print(f"Validação (Futuro): {len(df_valid)}")

def split_XY(df_part):
    Y = df_part[TARGET].copy()
    X = df_part.drop(columns=[TARGET] + COLS_TO_EXCLUDE_FROM_X, errors='ignore')
    return X, Y

X_elite, Y_elite = split_XY(df_treino_gold)
X_valid, Y_valid = split_XY(df_valid)
X_dirty, Y_dirty = split_XY(df_treino_dirty)

# ==============================================================================
# 4. ENCODING (ESTRATÉGIA DELTA MANUAL + CATBOOST)
# ==============================================================================
print("\n--- 3. Aplicando Encoders (Estratégia Delta) ---")

def encode_target_and_ohe(X_train, y_train, list_X_apply, df_train_full, woe_cols, ohe_cols):
    # 1. CRIAR O TARGET "DELTA" (Real - Objetivo) para o Encoder
    temp_obj_train = df_train_full.loc[y_train.index, 'temperaturaobjetivada']
    y_delta = y_train - temp_obj_train
    
    # 2. FIT NO DELTA (O encoder aprende o desvio médio)
    te = ce.TargetEncoder(cols=woe_cols)
    X_train_te = te.fit_transform(X_train, y_delta) 
    
    # One-Hot nas panelas
    X_train_final = pd.get_dummies(X_train_te, columns=ohe_cols, drop_first=True)
    
    # 3. APLICA NOS OUTROS
    outputs = [X_train_final]
    colunas_mestras = X_train_final.columns
    
    for X_app in list_X_apply:
        X_app_te = te.transform(X_app.copy())
        X_app_final = pd.get_dummies(X_app_te, columns=ohe_cols, drop_first=True)
        # Garante alinhamento de colunas
        X_app_final = X_app_final.reindex(columns=colunas_mestras, fill_value=0)
        outputs.append(X_app_final)
        
    return outputs

lista_transf = [X_valid, X_dirty]

outputs = encode_target_and_ohe(
    X_elite, Y_elite, 
    lista_transf, 
    df_treino_gold, 
    WOE_COL, OHE_COL
)

X_elite_final = outputs[0]
X_valid_final = outputs[1]
X_dirty_final = outputs[2]

# ==============================================================================
# 5. TREINO (CATBOOST ÚNICO - ESTÁVEL)
# ==============================================================================
# Mudamos para RMSE (padrão) para evitar a explosão matemática.
# O controle de segurança será feito na correção de viés.
params = {
    'iterations': 2000, 
    'learning_rate': 0.05, 
    'depth': 6,
    'loss_function': 'RMSE', # <--- MUDANÇA CRÍTICA: Volta para o estável
    'random_seed': 42,
    'verbose': 500,
    'allow_writing_files': False
    # 'task_type': 'GPU' # Se tiver GPU NVIDIA, descomente
}

print("\n--- 4. Treinando Modelo Único (CatBoost) ---")
model_unique = CatBoostRegressor(**params)
# cat_features=None pois já usamos o encoder manual
model_unique.fit(X_elite_final, Y_elite)

# ==============================================================================
# 7. FUNÇÃO DE AVALIAÇÃO DETALHADA
# ==============================================================================
def avaliar_cenario_unico(X_input, df_orig, Y_orig, nome_dataset, usar_vies=False):
    print(f"\n>>> AVALIANDO: {nome_dataset} (Correção: {usar_vies})")
    
    # 1. Predição
    pred_raw = model_unique.predict(X_input)
    
    # 2. Correção de Viés
    if usar_vies:
        # Se quiser forçar: pred_final = pred_raw - 1.5
        pred_final = pred_raw - 0.5
    else:
        pred_final = pred_raw
        
    # 3. Cálculo dos Erros
    # Legado
    diff_legado = df_orig['sugestaomodelolegado'] - df_orig[TARGET]
    erro_legado = (df_orig['temperaturamediareal'] + diff_legado) - df_orig['temperaturaobjetivada']
    
    # Novo Modelo
    diff_novo = pred_final - df_orig[TARGET]
    erro_novo = (df_orig['temperaturamediareal'] + diff_novo) - df_orig['temperaturaobjetivada']
    
    # 4. Categorização
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
    
    # Histograma
    sns.histplot(df_plot['Legado'], color='red', label='Legado', kde=True, ax=axes[0], alpha=0.3, element="step")
    sns.histplot(df_plot['Novo'], color='green', label='CatBoost', kde=True, ax=axes[0], alpha=0.3, element="step")
    axes[0].axvline(-5, color='green', linestyle='--', label='Meta')
    axes[0].axvline(10, color='green', linestyle='--')
    axes[0].axvline(-10, color='black', linestyle=':', label='Extremo')
    axes[0].axvline(20, color='black', linestyle=':')
    axes[0].set_title(f"Distribuição de Erros - {nome_dataset}")
    axes[0].set_xlabel("Erro vs Objetivo (°C)")
    axes[0].legend()
    
    # Barras
    categorias = [
        '1. Extremo Frio (< -15)', '2. Frio (-15 a -5)', 
        '3. Acerto (-5 a +10)', 
        '4. Quente (+10 a +20)', '5. Extremo Calor (> +20)'
    ]
    vals_leg = [stats_legado.get(c, 0) for c in categorias]
    vals_nov = [stats_novo.get(c, 0) for c in categorias]
    x = np.arange(len(categorias))
    
    bars1 = axes[1].bar(x - 0.17, vals_leg, 0.35, label='Legado', color='red', alpha=0.7)
    bars2 = axes[1].bar(x + 0.17, vals_nov, 0.35, label='CatBoost', color='green', alpha=0.7)
    
    axes[1].set_title(f"Comparação - {nome_dataset}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['EXT. FRIO', 'Frio', 'ACERTO', 'Quente', 'EXT. CALOR'], fontsize=10)
    axes[1].bar_label(bars1, fmt='%.1f%%', padding=3, fontsize=9)
    axes[1].bar_label(bars2, fmt='%.1f%%', padding=3, fontsize=9)
    axes[1].legend()
    
    plt.tight_layout(); plt.show()
    
    resumo = pd.DataFrame({'% Legado': vals_leg, '% Novo': vals_nov}, index=categorias)
    print(f"--- Resumo Numérico: {nome_dataset} ---")
    print(resumo.round(2))
    print("="*60)

# ==============================================================================
# 8. EXECUÇÃO
# ==============================================================================

avaliar_cenario_unico(X_elite_final, df_treino_gold, Y_elite, 
                      "1. TREINO ELITE (Sem Viés)", usar_vies=False)

avaliar_cenario_unico(X_dirty_final, df_treino_dirty, Y_dirty, 
                      "2. DADOS SUJOS (Com Correção)", usar_vies=True)

avaliar_cenario_unico(X_valid_final, df_valid, Y_valid, 
                      "3. VALIDAÇÃO FUTURA (Com Correção)", usar_vies=True)

# ==============================================================================
# 9. DIAGNÓSTICO
# ==============================================================================
print("\n--- 5. Diagnóstico Final ---")

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
importances = model_unique.get_feature_importance()
feature_names = model_unique.feature_names_
df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=df_imp.head(15), palette='magma')
plt.title("Feature Importance (CatBoost)")
plt.tight_layout(); plt.show()