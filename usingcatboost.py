import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================
TARGET = 'temperaturasaidafp'
# Não precisamos mais definir WOE/OHE, o CatBoost resolve.
COLS_TO_EXCLUDE_FROM_X = ['sugestaomodelolegado', 'temperaturamediareal', 'temperaturaobjetivada'] 
SWITCH_FEATURE = 'Desvio_Legado_Target'

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

# CatBoost precisa de strings sem NaNs nas categóricas
cat_features = ['qualidade', 'panela'] 
for c in cat_features:
    if c in df.columns: df[c] = df[c].fillna('MISSING').astype(str)

# FE
df['C_Medio'] = (df['c_max'] + df['c_min']) / 2
df['T_Liquid_Desvio'] = df['temperaturaobjetivada'] - df['temperaturaliquidus']
df['Desvio_Legado_Target'] = df['sugestaomodelolegado'] - df['temperaturaobjetivada']
df['Tempo_Sequencia'] = df['tempociclo'] * df['sequencia']

cols_to_drop_early = ['acoatual','corrida','secao','c_min','c_max','s_min',
                      'temperaturaliquidus','velocidadeobjetivada','velocidadereal']
df_clean = df.drop(columns=[c for c in cols_to_drop_early if c in df.columns], errors='ignore')

# Atualiza lista de cat_features removendo as que foram dropadas
cat_features = [c for c in cat_features if c in df_clean.columns]

# ==============================================================================
# 3. SPLIT GOLDEN BATCH (MODELO ÚNICO)
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
# 4. TREINO (CATBOOST ÚNICO)
# ==============================================================================
# CatBoost não precisa de encoder externo! Passamos 'cat_features' no fit.
params = {
    'iterations': 2000, 
    'learning_rate': 0.05, 
    'depth':6,
    'loss_function': 'Expectile:alpha=0.7', # RMSE é mais rápido e estável. Corrigimos o viés depois.
    'eval_metric': 'Expectile',
    'task_type' : 'CPU',
    'random_seed': 42,
    'verbose': 500,
    'allow_writing_files': False
}

print("\n--- 3. Treinando Modelo Único (CatBoost) ---")
model_unique = CatBoostRegressor(**params)
model_unique.fit(X_elite, Y_elite, cat_features=cat_features)

# ==============================================================================
# 5. CÁLCULO DO VIÉS (CALIBRAÇÃO)
# ==============================================================================
# Predição na Elite para calcular a calibração
p_tr_elite = model_unique.predict(X_elite)
vies_calculado = np.mean(p_tr_elite - Y_elite)

print(f"\n>>> VIÉS DETECTADO NA ELITE: {vies_calculado:.2f} °C")

# ==============================================================================
# 6. FUNÇÃO DE AVALIAÇÃO SIMPLIFICADA
# ==============================================================================
def avaliar_cenario_unico(X_input, df_orig, Y_orig, nome_dataset, usar_vies=False):
    print(f"\n>>> AVALIANDO: {nome_dataset} (Correção: {usar_vies})")
    
    # 1. Predição Única
    pred_raw = model_unique.predict(X_input)
    
    # 2. Aplicação do Viés
    if usar_vies:
        # Se quiser testar manual: pred_final = pred_raw - 2.0
        pred_final = pred_raw - 2
    else:
        pred_final = pred_raw
        
    # 3. Cálculo dos Erros (Fórmula Complexa)
    # Legado
    diff_legado = df_orig['sugestaomodelolegado'] - df_orig[TARGET]
    erro_legado = (df_orig['temperaturamediareal'] + diff_legado) - df_orig['temperaturaobjetivada']
    
    # Novo Modelo
    diff_novo = pred_final - df_orig[TARGET]
    erro_novo = (df_orig['temperaturamediareal'] + diff_novo) - df_orig['temperaturaobjetivada']
    
    # 4. Categorização e Stats
    df_plot = pd.DataFrame({'Legado': erro_legado, 'Novo': erro_novo})
    
    def categorizar(val):
        if val < -DELTA_NEG: return '1. Frio'
        elif val > DELTA_POS: return '3. Quente'
        else: return '2. Acerto'

    df_plot['Cat_Legado'] = df_plot['Legado'].apply(categorizar)
    df_plot['Cat_Novo'] = df_plot['Novo'].apply(categorizar)
    
    stats_legado = df_plot['Cat_Legado'].value_counts(normalize=True) * 100
    stats_novo = df_plot['Cat_Novo'].value_counts(normalize=True) * 100
    
    # 5. Plotagem
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Histograma
    sns.histplot(df_plot['Legado'], color='red', label='Legado', kde=True, ax=axes[0], alpha=0.3, element="step")
    sns.histplot(df_plot['Novo'], color='green', label='CatBoost', kde=True, ax=axes[0], alpha=0.3, element="step")
    axes[0].axvline(-DELTA_NEG, color='k', linestyle='--')
    axes[0].axvline(DELTA_POS, color='k', linestyle='--')
    axes[0].set_title(f"Distribuição de Erros - {nome_dataset}")
    axes[0].set_xlabel("Desvio do Objetivo (°C)")
    axes[0].legend()
    
    # Barras
    categorias = ['1. Frio', '2. Acerto', '3. Quente']
    vals_leg = [stats_legado.get(c, 0) for c in categorias]
    vals_nov = [stats_novo.get(c, 0) for c in categorias]
    x = np.arange(len(categorias))
    
    bars1 = axes[1].bar(x - 0.17, vals_leg, 0.35, label='Legado', color='red', alpha=0.7)
    bars2 = axes[1].bar(x + 0.17, vals_nov, 0.35, label='CatBoost', color='green', alpha=0.7)
    
    axes[1].set_title(f"Comparação de Desempenho - {nome_dataset}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['FRIO (< -5)', 'ACERTO', 'QUENTE (> +10)'])
    axes[1].bar_label(bars1, fmt='%.1f%%', padding=3)
    axes[1].bar_label(bars2, fmt='%.1f%%', padding=3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    resumo = pd.DataFrame({'% Legado': vals_leg, '% Novo': vals_nov}, index=['Frio', 'Acerto', 'Quente'])
    print(resumo.round(2))
    print("="*60)

# ==============================================================================
# 7. EXECUÇÃO
# ==============================================================================

# 1. ELITE (Aprendizado Puro)
avaliar_cenario_unico(X_elite, df_treino_gold, Y_elite, 
                      "1. TREINO ELITE (Sem Viés)", usar_vies=False)

# 2. DADOS SUJOS (Recuperação)
avaliar_cenario_unico(X_dirty, df_treino_dirty, Y_dirty, 
                      "2. DADOS SUJOS (Com Correção)", usar_vies=True)

# 3. VALIDAÇÃO FUTURA
avaliar_cenario_unico(X_valid, df_valid, Y_valid, 
                      "3. VALIDAÇÃO FUTURA (Com Correção)", usar_vies=True)