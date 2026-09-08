import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import lightgbm as lgb
from lightgbm import LGBMRegressor as LightGBMRegressor
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================
TARGET = 'temperaturasaidafp'
COLS_TO_EXCLUDE_FROM_X = ['sugestaomodelolegado', 'temperaturamediareal', 'temperaturaobjetivada', 'Desvio_Legado_Target'] 
VAL_INICIO = 7000  
DELTA_NEG = 5   
DELTA_POS = 10  

# ==============================================================================
# 2. FUNÇÃO DE PERDA (Só pro XGBoost, CatBoost vai de nativa ou custom class)
# ==============================================================================
def custom_asymmetric_loss_xgb(y_true, y_pred):
    resid = y_pred - y_true
    alpha = 1.3 
    factor = np.where(resid < 0, alpha, 1.0)
    grad = 2.0 * resid * factor
    hess = 2.0 * factor
    return grad, hess

# ==============================================================================
# 3. CARREGAMENTO (O mesmo de sempre)
# ==============================================================================
print("--- 1. Carregando Dados ---")
df = pd.read_csv('dados_fundo_do_amanha_evcomx.csv', delimiter=';')
df = df.drop_duplicates(subset=['corrida'], keep='first')

# ... (Todo o bloco de limpeza e FE igual ao anterior) ...
# Vou resumir a parte de FE para não estourar o limite de texto, 
# mas assuma que as colunas numéricas e categóricas foram tratadas aqui.
num_cols = ['al_min','c_min','c_max','n_min','s_min','sequencia','sequenciatotal',
            'vidapanela','tempociclo','tempovacuototal','temperaturaliquidus',
            'velocidadeobjetivada','velocidadereal','sugestaomodelolegado',
            'temperaturasaidafp','temperaturaobjetivada','temperaturamediareal']
for c in num_cols:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

cat_features = ['qualidade', 'panela', 'acoatual', 'secao']
for c in cat_features:
    if c in df.columns: df[c] = df[c].fillna('MISSING').astype(str)

df['C_Medio'] = (df['c_max'] + df['c_min']) / 2
df['T_Liquid_Desvio'] = df['temperaturaobjetivada'] - df['temperaturaliquidus']

cols_to_drop_early = ['acoatual','corrida','secao','c_min','c_max','s_min',
                      'temperaturaliquidus','velocidadeobjetivada','velocidadereal']
df_clean = df.drop(columns=[c for c in cols_to_drop_early if c in df.columns], errors='ignore')

# Para o Stacking com Sklearn, precisamos transformar tudo em numérico (dummies)
# O Sklearn não lida bem com "cat_features" do CatBoost dentro do wrapper Stacking
# Então vamos fazer One-Hot Encoding geral.
import category_encoders as ce
te = ce.TargetEncoder(cols=['qualidade'])
df_encoded = te.fit_transform(df_clean, df_clean[TARGET])
df_encoded = pd.get_dummies(df_encoded, columns=['panela'], drop_first=True)

# ==============================================================================
# 4. SPLIT GOLDEN BATCH
# ==============================================================================
print(f"\n--- 2. Separando Conjuntos ---")

df_valid = df_encoded.iloc[VAL_INICIO:].copy()
df_treino_total = df_encoded.iloc[:VAL_INICIO].copy()

# Filtro Elite
erro_real = df_treino_total['temperaturamediareal'] - df_treino_total['temperaturaobjetivada']
mask_gold = (erro_real >= -DELTA_NEG) & (erro_real <= DELTA_POS)

df_treino_gold = df_treino_total[mask_gold].copy()    # ELITE
df_treino_dirty = df_treino_total[~mask_gold].copy()  # SUJOS

def split_XY(df_part):
    Y = df_part[TARGET].copy()
    X = df_part.drop(columns=[TARGET] + COLS_TO_EXCLUDE_FROM_X, errors='ignore')
    return X, Y

X_elite, Y_elite = split_XY(df_treino_gold)
X_valid, Y_valid = split_XY(df_valid)
X_dirty, Y_dirty = split_XY(df_treino_dirty)

# ==============================================================================
# 5. DEFININDO O STACKING
# ==============================================================================
print("\n--- 3. Configurando o Stacking ---")

# Modelos Base (Os Especialistas)
# Nota: Adicionei Random Forest para dar diversidade (ajuda o Stacking)
estimators = [
    ('xgb', XGBRegressor(
        n_estimators=840, learning_rate=0.05, max_depth=12,
        subsample=0.6, colsample_bytree=0.72,
        min_child_weight=8,alpha=1.4,
        objective=custom_asymmetric_loss_xgb, # XGB com sua perda
        n_jobs=-1, random_state=42
    )),
    ('cat', CatBoostRegressor(
        iterations=2000, learning_rate=0.05, depth=6,
        loss_function='RMSE', # Catboost vai de RMSE padrão
        verbose=0, random_seed=42, allow_writing_files=False
    )),
    ('rf', RandomForestRegressor(
        n_estimators=200, max_depth=10, n_jobs=-1, random_state=42
    ))
]

# O Meta-Modelo (O Chefe)
# RidgeCV é uma regressão linear robusta (ceguom rlarização)
meta_model = RidgeCV()

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=meta_model,
    cv=5, # Validação cruzada interna de 5 pastas
    n_jobs=1
)

# ==============================================================================
# 6. TREINO DO STACK
# ==============================================================================
print("Treinando o Stacking (Isso pode demorar um pouco)...")
stacking_model.fit(X_elite, Y_elite)

# ==============================================================================
# 7. FUNÇÃO DE AVALIAÇÃO DETALHADA
# ==============================================================================
def avaliar_cenario_unico(X_input, df_orig, Y_orig, nome_dataset, usar_vies=False):
    print(f"\n>>> AVALIANDO: {nome_dataset} (Correção: {usar_vies})")
    
    # 1. Predição
    pred_raw = stacking_model.predict(X_input)
    
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

avaliar_cenario_unico(X_elite, df_treino_gold, Y_elite, 
                      "1. TREINO ELITE (Sem Viés)", usar_vies=False)

avaliar_cenario_unico(X_dirty, df_treino_dirty, Y_dirty, 
                      "2. DADOS SUJOS (Com Correção)", usar_vies=True)

avaliar_cenario_unico(X_valid, df_valid, Y_valid, 
                      "3. VALIDAÇÃO FUTURA (Com Correção)", usar_vies=True)
