import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Função de perda assimétrica (mantida)
def custom_asymmetric_loss(y_true, y_pred):
    resid = y_pred - y_true 
    alpha = 1.2
    grad = np.where(resid < 0, alpha * resid, 1.0 * resid) 
    hess = np.where(resid < 0, alpha, 1.0)
    return grad, hess

# --- 0. DEFINIÇÕES ---
TARGET = 'temperaturasaidafp'
WOE_COL = ['qualidade'] 
OHE_COL = ['panela']
SWITCH_FEATURE = 'Desvio_Legado_Target' 
LIMITE_VALIDACAO = 500
LIMITE_USOU = 5 
DELTA_NEG = 5   
DELTA_POS = 10 

# Colunas guardadas para cálculo da métrica complexa (não entram no X)
COLS_TO_EXCLUDE_FROM_X = [
    'sugestaomodelolegado', 
    'temperaturamediareal', 
    'temperaturaobjetivada'
]

# --- 1. CARREGAMENTO E FE ---
df = pd.read_csv('dados_fundo_do_amanha_evcomx.csv', delimiter=';')

num_cols = ['al_min', 'c_min', 'c_max', 'n_min', 's_min', 'sequencia', 'sequenciatotal', 
            'vidapanela', 'tempociclo', 'tempovacuototal', 'temperaturaliquidus', 
            'velocidadeobjetivada', 'velocidadereal', 'sugestaomodelolegado', 
            'temperaturasaidafp', 'temperaturaobjetivada', 'temperaturamediareal']

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce') 

id_cols = ['corrida', 'secao', 'acoatual', 'qualidade', 'panela']
for col in id_cols:
    df[col] = df[col].astype(str)

df['C_Medio'] = (df['c_max'] + df['c_min']) / 2
df['T_Liquid_Desvio'] = df['temperaturaobjetivada'] - df['temperaturaliquidus']
df['Desvio_Legado_Target'] = df['sugestaomodelolegado'] - df['temperaturaobjetivada']
df['Tempo_Sequencia'] = df['tempociclo'] * df['sequencia']

# --- 2. CLEANING ---
cols_to_drop_early = [
    'acoatual', 'corrida', 'secao', 'c_min', 'c_max', 
    'temperaturaliquidus', 
    'velocidadeobjetivada', 'velocidadereal'
]
df_clean = df.drop(columns=cols_to_drop_early, errors='ignore')

# --- 3. SPLIT TEMPORAL ---
df_validacao_final = df_clean.iloc[-LIMITE_VALIDACAO:].copy()
df_treino_bruto = df_clean.iloc[:-LIMITE_VALIDACAO].copy()

df_treino_bruto['usage'] = np.where(
    (df_treino_bruto['sugestaomodelolegado'] - df_treino_bruto['temperaturasaidafp']).abs() <= LIMITE_USOU,
    'usou', 'Naousou'
)

df_usou = df_treino_bruto[df_treino_bruto['usage'] == 'usou'].drop(columns=['usage'], errors='ignore').copy()
df_operadores = df_treino_bruto[df_treino_bruto['usage'] == 'Naousou'].drop(columns=['usage'], errors='ignore').copy()

# --- 4. SEPARAÇÃO X/Y ---
Y_usou = df_usou[TARGET]
X_usou = df_usou.drop(TARGET, axis=1).drop(COLS_TO_EXCLUDE_FROM_X, axis=1)

Y_operadores = df_operadores[TARGET]
X_operadores = df_operadores.drop(TARGET, axis=1).drop(COLS_TO_EXCLUDE_FROM_X, axis=1)

Y_validacao = df_validacao_final[TARGET]
X_validacao = df_validacao_final.drop(TARGET, axis=1).drop(COLS_TO_EXCLUDE_FROM_X, axis=1)

# --- 5. ENCODING ---
target_encoder_usou = ce.TargetEncoder(cols=WOE_COL) 
X_usou_te = target_encoder_usou.fit_transform(X_usou, Y_usou) 
X_usou_final = pd.get_dummies(X_usou_te, columns=OHE_COL, drop_first=True)

target_encoder_op = ce.TargetEncoder(cols=WOE_COL) 
X_operadores_te = target_encoder_op.fit_transform(X_operadores, Y_operadores)
X_operadores_final = pd.get_dummies(X_operadores_te, columns=OHE_COL, drop_first=True)

X_val_usou = target_encoder_usou.transform(X_validacao.copy())
X_val_usou_final = pd.get_dummies(X_val_usou, columns=OHE_COL, drop_first=True)

X_val_op = target_encoder_op.transform(X_validacao.copy())
X_val_op_final = pd.get_dummies(X_val_op, columns=OHE_COL, drop_first=True)

colunas_mestras = X_usou_final.columns.union(X_operadores_final.columns)
X_usou_final = X_usou_final.reindex(columns=colunas_mestras, fill_value=0)
X_operadores_final = X_operadores_final.reindex(columns=colunas_mestras, fill_value=0)
X_val_usou_final = X_val_usou_final.reindex(columns=colunas_mestras, fill_value=0)
X_val_op_final = X_val_op_final.reindex(columns=colunas_mestras, fill_value=0)

# --- 6. TREINO ---
params = {
    'n_estimators': 827, 'learning_rate': 0.056, 'random_state': 42,
    'objective': custom_asymmetric_loss, 'booster': 'dart', 'max_depth': 12,
    'alpha': 1.40375, 'colsample_bytree': 0.71256, 'min_child_weight': 8,
    'reg_lambda': 0.00643, 'subsample': 0.6017, "tree_method": "hist", 
    "device": "cuda",    
}
model_usou = XGBRegressor(**params)
model_usou.fit(X_usou_final, Y_usou)
model_op = XGBRegressor(**params)
model_op.fit(X_operadores_final, Y_operadores)

# --- 7. PREDIÇÃO VALIDAÇÃO ---
pred_usou = model_usou.predict(X_val_usou_final)
pred_op = model_op.predict(X_val_op_final)
desvio_val = df_validacao_final[SWITCH_FEATURE].abs()

# Previsão Bruta do Novo Modelo
pred_novo_modelo = np.where(desvio_val <= LIMITE_USOU, pred_usou, pred_op)

# ==============================================================================
# --- 8. CÁLCULO MÉTRICA COMPLEXA (VALIDAÇÃO) ---
# ==============================================================================

# --- CÁLCULO LEGADO ---
# 1. Diferença do que o Legado pediu vs O que aconteceu
diff_legado = df_validacao_final['sugestaomodelolegado'] - df_validacao_final['temperaturasaidafp']
# 2. Aplica na Média Real
temp_ajustada_legado = df_validacao_final['temperaturamediareal'] + diff_legado
# 3. Erro final vs Objetivo
erro_legado_complexo = temp_ajustada_legado - df_validacao_final['temperaturaobjetivada']

# --- CÁLCULO NOVO MODELO ---
# 1. Diferença do que o NOVO pediu vs O que aconteceu
diff_novo = pred_novo_modelo - df_validacao_final['temperaturasaidafp']
# 2. Aplica na Média Real
temp_ajustada_novo = df_validacao_final['temperaturamediareal'] + diff_novo
# 3. Erro final vs Objetivo
erro_novo_complexo = temp_ajustada_novo - df_validacao_final['temperaturaobjetivada']

# Métricas de Acerto
acerto_legado = np.mean((erro_legado_complexo >= -DELTA_NEG) & (erro_legado_complexo <= DELTA_POS)) * 100
acerto_novo = np.mean((erro_novo_complexo >= -DELTA_NEG) & (erro_novo_complexo <= DELTA_POS)) * 100

print("\n--- RESULTADOS VALIDAÇÃO (Lógica Complexa para AMBOS) ---")
print(f"Acerto Legado: {acerto_legado:.2f}%")
print(f"Acerto Novo Modelo: {acerto_novo:.2f}%")


# ==============================================================================
# --- 9. CÁLCULO MÉTRICA COMPLEXA (TREINO - HISTÓRICO TOTAL) ---
# ==============================================================================

# Preparação X Treino
Y_treino_total = df_treino_bruto[TARGET]
# CORREÇÃO AQUI: Adicionamos .drop(columns=['usage'], errors='ignore')
X_treino_total = df_treino_bruto.drop(columns=['usage'], errors='ignore').drop(TARGET, axis=1).drop(COLS_TO_EXCLUDE_FROM_X, axis=1)

X_treino_path_usou = target_encoder_usou.transform(X_treino_total.copy())
X_treino_path_usou_final = pd.get_dummies(X_treino_path_usou, columns=OHE_COL, drop_first=True)
X_treino_path_usou_final = X_treino_path_usou_final.reindex(columns=colunas_mestras, fill_value=0)

X_treino_path_op = target_encoder_op.transform(X_treino_total.copy())
X_treino_path_op_final = pd.get_dummies(X_treino_path_op, columns=OHE_COL, drop_first=True)
X_treino_path_op_final = X_treino_path_op_final.reindex(columns=colunas_mestras, fill_value=0)

# Predições Treino
pred_treino_usou = model_usou.predict(X_treino_path_usou_final)
pred_treino_op = model_op.predict(X_treino_path_op_final)
desvio_treino = df_treino_bruto[SWITCH_FEATURE].abs()

pred_novo_treino = np.where(desvio_treino <= LIMITE_USOU, pred_treino_usou, pred_treino_op)

# --- APLICAÇÃO DA LÓGICA COMPLEXA NO TREINO ---

# 1. LEGADO (Treino)
diff_legado_train = df_treino_bruto['sugestaomodelolegado'] - df_treino_bruto['temperaturasaidafp']
temp_ajustada_legado_train = df_treino_bruto['temperaturamediareal'] + diff_legado_train
erro_legado_complexo_train = temp_ajustada_legado_train - df_treino_bruto['temperaturaobjetivada']

# 2. NOVO MODELO (Treino)
diff_novo_train = pred_novo_treino - df_treino_bruto['temperaturasaidafp']
temp_ajustada_novo_train = df_treino_bruto['temperaturamediareal'] + diff_novo_train
erro_novo_complexo_train = temp_ajustada_novo_train - df_treino_bruto['temperaturaobjetivada']

# --- MÉTRICAS FINAIS (TREINO) ---
acerto_legado_train = np.mean((erro_legado_complexo_train >= -DELTA_NEG) & (erro_legado_complexo_train <= DELTA_POS)) * 100
acerto_novo_train = np.mean((erro_novo_complexo_train >= -DELTA_NEG) & (erro_novo_complexo_train <= DELTA_POS)) * 100

ErroBaixoLegado_train = np.mean(erro_legado_complexo_train < -DELTA_NEG) * 100
ErroCimaLegado_train = np.mean(erro_legado_complexo_train > DELTA_POS) * 100

ErroBaixoNovo_train = np.mean(erro_novo_complexo_train < -DELTA_NEG) * 100
ErroCimaNovo_train = np.mean(erro_novo_complexo_train > DELTA_POS) * 100

print("\n--- RESULTADOS NO TREINO (SIMULAÇÃO REAL VS OBJETIVO) ---")
print("-----------------------------------------------------------")
print(f"Taxa de Acerto Legado: {acerto_legado_train:.2f}%")
print(f"Taxa de Acerto Novo Modelo: {acerto_novo_train:.2f}%")
print("-----------------------------------------------------------")
print(f"Legado - Erro pra baixo (< -5): {ErroBaixoLegado_train:.2f}%")
print(f"Novo - Erro pra baixo (< -5): {ErroBaixoNovo_train:.2f}%")
print("-----------------------------------------------------------") 
print(f"Legado - Erro pra cima (> +10): {ErroCimaLegado_train:.2f}%")
print(f"Novo - Erro pra cima (> +10): {ErroCimaNovo_train:.2f}%")
print("-----------------------------------------------------------")

# Plotagem da Simulação
plt.figure(figsize=(12, 6))
sns.histplot(erro_legado_complexo_train, color='red', label='Legado (Simulado)', kde=True, alpha=0.3, binwidth=2)
sns.histplot(erro_novo_complexo_train, color='blue', label='Novo Modelo (Simulado)', kde=True, alpha=0.3, binwidth=2)
plt.axvline(-DELTA_NEG, color='k', linestyle='--', label='Limite Inferior (-5)')
plt.axvline(DELTA_POS, color='k', linestyle='--', label='Limite Superior (+10)')
plt.title("Comparação Justa: Desvio da Temperatura Objetivada (Simulação no Treino)")
plt.xlabel("Desvio em relação ao Objetivo (°C)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()