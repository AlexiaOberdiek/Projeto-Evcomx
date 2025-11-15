import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from modeloxgb import XGBRegressor
# --- 0. DEFINIÇÕES E PARÂMETROS ---
TARGET = 'temperaturasaidafp'
WOE_COL = ['qualidade'] 
OHE_COL = ['panela']
SWITCH_FEATURE = 'Desvio_Legado_Target' 
LIMITE_VALIDACAO = 2000
LIMITE_USOU = 5 
DELTA_NEG = 5   
DELTA_POS = 10 
# Colunas a serem excluídas APENAS das FEATURES (X)
COLS_TO_EXCLUDE_FROM_X = [
    'sugestaomodelolegado', 'temperaturamediareal'
]


# --- 1. CARREGAMENTO E FEATURE ENGINEERING (FE) ---
df = pd.read_csv('dados_fundo_do_amanha_evcomx.csv', delimiter=';')

# Coerção de Tipos e Categorias
num_cols = ['al_min', 'c_min', 'c_max', 'n_min', 's_min', 'sequencia', 'sequenciatotal', 
            'vidapanela', 'tempociclo', 'tempovacuototal', 'temperaturaliquidus', 
            'velocidadeobjetivada', 'velocidadereal', 'sugestaomodelolegado', 
            'temperaturasaidafp', 'temperaturaobjetivada', 'temperaturamediareal']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce') 

id_cols = ['corrida', 'secao', 'acoatual', 'qualidade', 'panela']
for col in id_cols:
    df[col] = df[col].astype(str)

# Feature Engineering (Cria as colunas DESVIO e LIMITE)
df['C_Medio'] = (df['c_max'] + df['c_min']) / 2
df['T_Liquid_Desvio'] = df['temperaturaobjetivada'] - df['temperaturaliquidus']
df['Desvio_Legado_Target'] = df['sugestaomodelolegado'] - df['temperaturaobjetivada']
df['Tempo_Sequencia'] = df['tempociclo'] * df['sequencia']

# --- 2. DROP DE COLUNAS (APENAS IDs e VIF/MULTICOLINEARIDADE) ---
# MANTENDO 'sugestaomodelolegado' e 'temperaturamediareal' temporariamente
cols_to_drop_early = [
    'acoatual', 'corrida', 'secao', 'c_min', 'c_max', 
    'temperaturaliquidus', 'temperaturaobjetivada', 
    'velocidadeobjetivada', 'velocidadereal'
]
df_clean = df.drop(columns=cols_to_drop_early, errors='ignore')

# --- 3. DIVISÃO TEMPORAL E SEGMENTAÇÃO ---
df_validacao_final = df_clean.iloc[-LIMITE_VALIDACAO:].copy()
df_treino_bruto = df_clean.iloc[:-LIMITE_VALIDACAO].copy()

# Criação da CHAVE DE SEGMENTAÇÃO
df_treino_bruto['usage'] = np.where(
    (df_treino_bruto['sugestaomodelolegado'] - df_treino_bruto['temperaturasaidafp']).abs() <= LIMITE_USOU,
    'usou',
    'Naousou'
)

df_usou = df_treino_bruto[df_treino_bruto['usage'] == 'usou'].drop(columns=['usage'], errors='ignore').copy()
df_operadores = df_treino_bruto[df_treino_bruto['usage'] == 'Naousou'].drop(columns=['usage'], errors='ignore').copy()

# --- 4. SEPARAÇÃO X/Y E DROPPING FINAL (Onde o Legado é Removido das FEATURES) ---

# 4.1. Separação X e Y (USOU)
Y_usou = df_usou[TARGET]
X_usou = df_usou.drop(TARGET, axis=1).drop(COLS_TO_EXCLUDE_FROM_X, axis=1)

# 4.2. Separação X e Y (OPERADORES)
Y_operadores = df_operadores[TARGET]
X_operadores = df_operadores.drop(TARGET, axis=1).drop(COLS_TO_EXCLUDE_FROM_X, axis=1)

# 4.3. Separação X e Y (VALIDAÇÃO)
Y_validacao = df_validacao_final[TARGET]
X_validacao = df_validacao_final.drop(TARGET, axis=1).drop(COLS_TO_EXCLUDE_FROM_X, axis=1)

# --- 5. TARGET ENCODING (TARGET ENCODER) ---

# Target Encoder (USOU)
target_encoder_usou = ce.TargetEncoder(cols=WOE_COL) 
X_usou_te = target_encoder_usou.fit_transform(X_usou, Y_usou) 
X_usou_final = pd.get_dummies(X_usou_te, columns=OHE_COL, drop_first=True)

# Target Encoder (OPERADORES)
target_encoder_op = ce.TargetEncoder(cols=WOE_COL) 
X_operadores_te = target_encoder_op.fit_transform(X_operadores, Y_operadores)
X_operadores_final = pd.get_dummies(X_operadores_te, columns=OHE_COL, drop_first=True)

# Transformação da Validação
X_val_usou = target_encoder_usou.transform(X_validacao.copy())
X_val_usou_final = pd.get_dummies(X_val_usou, columns=OHE_COL, drop_first=True)

X_val_op = target_encoder_op.transform(X_validacao.copy())
X_val_op_final = pd.get_dummies(X_val_op, columns=OHE_COL, drop_first=True)

# Alinhamento de Colunas
colunas_mestras = X_usou_final.columns.union(X_operadores_final.columns)

X_usou_final = X_usou_final.reindex(columns=colunas_mestras, fill_value=0)
X_operadores_final = X_operadores_final.reindex(columns=colunas_mestras, fill_value=0)
X_val_usou_final = X_val_usou_final.reindex(columns=colunas_mestras, fill_value=0)
X_val_op_final = X_val_op_final.reindex(columns=colunas_mestras, fill_value=0)

# --- 6. TREINAMENTO DOS MODELOS XGBOOST ---
model_usou = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, tree_method='hist',booster = 'dart')
model_usou.fit(X_usou_final, Y_usou)

model_op = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, tree_method='hist',booster = 'dart')
model_op.fit(X_operadores_final, Y_operadores)

# --- 7. PREDIÇÃO E MECANISMO DE SWITCH ---
pred_usou = model_usou.predict(X_val_usou_final)
pred_op = model_op.predict(X_val_op_final)
# Mecanismo de SWITCH: Se o desvio Legado/Alvo for baixo, usa o Modelo USOU
desvio_val = df_validacao_final[SWITCH_FEATURE].abs()

pred_novo_modelo = np.where(
    desvio_val <= LIMITE_USOU,
    pred_usou, 
    pred_op
)

# Definição do Baseline
pred_legado = df_validacao_final['sugestaomodelolegado'] 

# --- 8. AVALIAÇÃO DO NOVO MODELO VS. LEGADO (MÉTRICA ASSIMÉTRICA) ---#
mse_novo = mean_squared_error(Y_validacao, pred_novo_modelo)
mse_legado = mean_squared_error(Y_validacao, pred_legado)
rmse_novo = np.sqrt(mse_novo)
rmse_legado = np.sqrt(mse_legado)
r2_score_novo = r2_score(Y_validacao, pred_novo_modelo)
r2_score_legado = r2_score(Y_validacao, pred_legado)    
# Métrica de Acerto: O erro deve estar entre -5°C e +10°C
erro_novo = pred_novo_modelo - Y_validacao
erro_legado = pred_legado - Y_validacao

is_hit_novo = (erro_novo >= -DELTA_NEG) & (erro_novo <= DELTA_POS)
acerto_novo = np.mean(is_hit_novo) * 100

is_hit_legado = (erro_legado >= -DELTA_NEG) & (erro_legado <= DELTA_POS)
acerto_legado = np.mean(is_hit_legado) * 100


# --- 9. EXIBIÇÃO DOS RESULTADOS ---
print("\n--- RESULTADOS FINAIS NO TESTE TEMPORAL (ÚLTIMAS 500 CORRIDAS) ---")
print("------------------------------------------------------------------")
print(f"RMSE (Legado): {rmse_legado:.2f} °C")
print(f"RMSE (Novo Modelo Segmentado - XGBoost): {rmse_novo:.2f} °C")
print("------------------------------------------------------------------")
print(f"R² (Legado): {r2_score_legado:.4f}")
print(f"R² (Novo Modelo Segmentado - XGBoost): {r2_score_novo:.4f}")
print("------------------------------------------------------------------")
print(f"Taxa de Acerto (-{DELTA_NEG}°C a +{DELTA_POS}°C) Legado: {acerto_legado:.2f}%")
print(f"Taxa de Acerto (-{DELTA_NEG}°C a +{DELTA_POS}°C) Novo Modelo: {acerto_novo:.2f}%")
print("------------------------------------------------------------------")
print(f"Erro pra baixo (-{DELTA_NEG}°C) Legado: {(100 - acerto_legado)/2:.2f}%")
print(f"Erro pra baixo (-{DELTA_NEG}°C) Novo Modelo: {(100 - acerto_novo)/2:.2f}%")
print("------------------------------------------------------------------") 
print(f"Erro pra cima (+{DELTA_POS}°C) Legado: {(100 - acerto_legado)/2:.2f}%")
print(f"Erro pra cima (+{DELTA_POS}°C) Novo Modelo: {(100 - acerto_novo)/2:.2f}%")
print("------------------------------------------------------------------")
print("Quantidade de corridas na Validação:", len(Y_validacao))
