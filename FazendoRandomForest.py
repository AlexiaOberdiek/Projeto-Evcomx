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

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================
TARGET = 'temperaturasaidafp'
COLS_TO_EXCLUDE_FROM_X = ['sugestaomodelolegado', 'temperaturamediareal', 'temperaturaobjetivada'] 
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
df['Desvio_Legado_Target'] = df['sugestaomodelolegado'] - df['temperaturaobjetivada']
df['Tempo_Sequencia'] = df['tempociclo'] * df['sequencia']

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
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        objective=custom_asymmetric_loss_xgb, # XGB com sua perda
        n_jobs=-1, random_state=42
    )),
    ('cat', CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=6,
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
# 7. CÁLCULO DE VIÉS E AVALIAÇÃO
# ==============================================================================
# Mesmo sendo um Stack, calculamos o viés final na Elite
p_tr_elite = stacking_model.predict(X_elite)
vies_calculado = np.mean(p_tr_elite - Y_elite)
print(f"\n>>> VIÉS DO STACK NA ELITE: {vies_calculado:.2f} °C")

def avaliar_cenario(X_input, df_orig, nome_dataset, usar_vies=False):
    print(f"\n>>> AVALIANDO: {nome_dataset} ---")
    
    pred_raw = stacking_model.predict(X_input)
    
    if usar_vies:
        pred_final = pred_raw - vies_calculado
    else:
        pred_final = pred_raw
        
    # Erros
    diff_legado = df_orig['sugestaomodelolegado'] - df_orig[TARGET]
    erro_legado = (df_orig['temperaturamediareal'] + diff_legado) - df_orig['temperaturaobjetivada']
    
    diff_novo = pred_final - df_orig[TARGET]
    erro_novo = (df_orig['temperaturamediareal'] + diff_novo) - df_orig['temperaturaobjetivada']
    
    # Categorização (5 Faixas)
    df_plot = pd.DataFrame({'Legado': erro_legado, 'Stacking': erro_novo})
    
    def categorizar(val):
        if val < -10: return '1. Extremo Frio'
        elif val < -5: return '2. Frio'
        elif val <= 10: return '3. Acerto'
        elif val <= 20: return '4. Quente'
        else: return '5. Extremo Calor'

    df_plot['Cat_Legado'] = df_plot['Legado'].apply(categorizar)
    df_plot['Cat_Novo'] = df_plot['Stacking'].apply(categorizar)
    
    # Cálculo das porcentagens
    cols_order = ['1. Extremo Frio', '2. Frio', '3. Acerto', '4. Quente', '5. Extremo Calor']
    stats_legado = df_plot['Cat_Legado'].value_counts(normalize=True) * 100
    stats_novo = df_plot['Cat_Novo'].value_counts(normalize=True) * 100
    
    # Print Tabela
    resumo = pd.DataFrame({'% Legado': stats_legado, '% Stacking': stats_novo}).reindex(cols_order).fillna(0)
    print(resumo.round(2))
    
    # Plotagem Simplificada
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(cols_order))
    width = 0.35
    
    ax.bar(x - width/2, resumo['% Legado'], width, label='Legado', color='red', alpha=0.7)
    ax.bar(x + width/2, resumo['% Stacking'], width, label='Stacking', color='purple', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Ext. Frio', 'Frio', 'ACERTO', 'Quente', 'Ext. Calor'])
    ax.set_title(f"Performance: {nome_dataset}")
    ax.legend()
    plt.show()

# EXECUÇÃO
avaliar_cenario(X_elite, df_treino_gold, "1. ELITE (Sem Viés)", False)
avaliar_cenario(X_dirty, df_treino_dirty, "2. SUJOS (Com Viés)", True)
avaliar_cenario(X_valid, df_valid, "3. VALIDAÇÃO (Com Viés)", True)