import numpy as np
import pandas as pd
import category_encoders as ce
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================
TARGET = 'temperaturasaidafp'
WOE_COL = ['qualidade']
OHE_COL = ['panela']
COLS_TO_EXCLUDE_FROM_X = ['sugestaomodelolegado', 'temperaturamediareal', 'temperaturaobjetivada'] 
SWITCH_FEATURE = 'Desvio_Legado_Target'

VAL_INICIO = 7000  
LIMITE_USOU = 5
DELTA_NEG = 5   
DELTA_POS = 10  

# ==============================================================================
# 2. FUNÇÃO DE PERDA ASSIMÉTRICA (XGBOOST)
# ==============================================================================
def custom_asymmetric_loss(y_true, y_pred):
    resid = y_pred - y_true
    alpha = 1.3
    factor = np.where(resid < 0, alpha, 1.0)
    grad = 2.0 * resid * factor
    hess = 2.0 * factor
    return grad, hess

# ==============================================================================
# 3. CARREGAMENTO E PREPARAÇÃO
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
df['Desvio_Legado_Target'] = df['sugestaomodelolegado'] - df['temperaturaobjetivada']
df['Tempo_Sequencia'] = df['tempociclo'] * df['sequencia']

cols_to_drop_early = ['acoatual','corrida','secao','c_min','c_max',
                      'temperaturaliquidus','velocidadeobjetivada','velocidadereal']
df_clean = df.drop(columns=[c for c in cols_to_drop_early if c in df.columns], errors='ignore')

# ==============================================================================
# 4. SPLIT GOLDEN BATCH (MODELO ÚNICO)
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
    return X, Y

# Criação dos Datasets
X_elite, Y_elite = split_XY(df_treino_gold)
X_valid, Y_valid = split_XY(df_valid)
X_dirty, Y_dirty = split_XY(df_treino_dirty)

# ==============================================================================
# 5. ENCODING (ÚNICO)
# ==============================================================================
print("\n--- 3. Aplicando Encoders ---")

# Treina o encoder APENAS na Elite
te = ce.TargetEncoder(cols=WOE_COL)
X_elite_te = te.fit_transform(X_elite, Y_elite)
X_elite_final = pd.get_dummies(X_elite_te, columns=OHE_COL, drop_first=True)

# Aplica nos outros conjuntos
def aplicar_encoder(X_raw, encoder_treinado, colunas_modelo):
    X_te = encoder_treinado.transform(X_raw.copy())
    X_final = pd.get_dummies(X_te, columns=OHE_COL, drop_first=True)
    # Garante as mesmas colunas do treino
    X_final = X_final.reindex(columns=colunas_modelo, fill_value=0)
    return X_final

colunas_mestras = X_elite_final.columns
X_valid_final = aplicar_encoder(X_valid, te, colunas_mestras)
X_dirty_final = aplicar_encoder(X_dirty, te, colunas_mestras)

# ==============================================================================
# 6. TREINO (MODELO ÚNICO XGBOOST)
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
# 7. CÁLCULO DO VIÉS
# ==============================================================================
# Predição na Elite para calcular a calibração
p_tr_elite = model_unique.predict(X_elite_final)
vies_calculado = np.mean(p_tr_elite - Y_elite)

print(f"\n>>> VIÉS DETECTADO NA ELITE: {vies_calculado:.2f} °C")

# ==============================================================================
# 8. FUNÇÃO DE AVALIAÇÃO DETALHADA (5 FAIXAS)
# ==============================================================================
def avaliar_cenario_unico(X_input, df_orig, Y_orig, nome_dataset, usar_vies=False):
    print(f"\n>>> AVALIANDO: {nome_dataset} (Correção: {usar_vies})")
    
    # 1. Predição Única
    pred_raw = model_unique.predict(X_input)
    
    # 2. Aplicação do Viés
    if usar_vies:
        pred_final = pred_raw - 1.5
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
        if val < -10:           return '1. Extremo Frio (< -10)'
        elif val >= -10 and val < -5: return '2. Frio (-10 a -5)'
        elif val >= -5 and val <= 10: return '3. Acerto (-5 a +10)'
        elif val > 10 and val <= 25:  return '4. Quente (+10 a +25)'
        else:                   return '5. Extremo Calor (> +25)'

    df_plot['Cat_Legado'] = df_plot['Legado'].apply(categorizar)
    df_plot['Cat_Novo'] = df_plot['Novo'].apply(categorizar)
    
    stats_legado = df_plot['Cat_Legado'].value_counts(normalize=True) * 100
    stats_novo = df_plot['Cat_Novo'].value_counts(normalize=True) * 100
    
    # 5. Plotagem
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    # --- GRÁFICO 1: HISTOGRAMA ---
    sns.histplot(df_plot['Legado'], color='red', label='Legado', kde=True, ax=axes[0], alpha=0.3, element="step")
    sns.histplot(df_plot['Novo'], color='purple', label='XGBoost', kde=True, ax=axes[0], alpha=0.3, element="step")
    
    # Linhas de Limite
    axes[0].axvline(-5, color='green', linestyle='--', linewidth=2, label='Meta (-5)')
    axes[0].axvline(10, color='green', linestyle='--', linewidth=2, label='Meta (+10)')
    axes[0].axvline(-10, color='black', linestyle=':', linewidth=2, label='Extremo (-10)')
    axes[0].axvline(20, color='black', linestyle=':', linewidth=2, label='Extremo (+20)')
    
    axes[0].set_title(f"Distribuição de Erros - {nome_dataset}")
    axes[0].set_xlabel("Desvio do Objetivo (°C)")
    axes[0].legend()
    
    # --- GRÁFICO 2: BARRAS (5 CATEGORIAS) ---
    categorias = [
        '1. Extremo Frio (< -10)', 
        '2. Frio (-10 a -5)', 
        '3. Acerto (-5 a +10)', 
        '4. Quente (+10 a +25)', 
        '5. Extremo Calor (> +25)'
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
    
    plt.tight_layout()
    plt.show()
    
    # 6. Tabela Resumo
    resumo = pd.DataFrame({'% Legado': vals_leg, '% Novo': vals_nov}, index=categorias)
    print(f"--- Resumo Numérico: {nome_dataset} ---")
    print(resumo.round(2))
    print("="*60)

# ==============================================================================
# 9. EXECUÇÃO
# ==============================================================================

# 1. ELITE
avaliar_cenario_unico(X_elite_final, df_treino_gold, Y_elite, 
                      "1. TREINO ELITE (Sem Viés)", usar_vies=False)

# 2. DADOS SUJOS (Recuperação)
avaliar_cenario_unico(X_dirty_final, df_treino_dirty, Y_dirty, 
                      "2. DADOS SUJOS (Com Correção)", usar_vies=True)

# 3. VALIDAÇÃO FUTURA
avaliar_cenario_unico(X_valid_final, df_valid, Y_valid, 
                      "3. VALIDAÇÃO FUTURA (Com Correção)", usar_vies=True)