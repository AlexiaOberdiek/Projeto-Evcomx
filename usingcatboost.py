import numpy as np
import pandas as pd
import optuna
from xgboost import XGBRegressor
import category_encoders as ce
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ======================================
#   CUSTOM ASYMMETRIC LOSS
# ======================================
def custom_asymmetric_loss(y_true, y_pred):
    resid = y_pred - y_true
    alpha = 1.3
    factor = np.where(resid < 0, alpha, 1.0)
    grad = 2.0 * resid * factor
    hess = 2.0 * factor
    return grad, hess

# ======================================
#   CONSTANTES
# ======================================
TARGET = 'temperaturasaidafp'
WOE_COL = ['qualidade']
OHE_COL = ['panela']
COLS_TO_EXCLUDE_FROM_X = ['sugestaomodelolegado', 'temperaturamediareal', 'temperaturaobjetivada'] 
SWITCH_FEATURE = 'Desvio_Legado_Target'
LIMITE_VALIDACAO = 500
LIMITE_USOU = 5
DELTA_NEG = 5
DELTA_POS = 10

# ======================================
#   LOAD + FEATURE ENGINEERING
# ======================================
def load_and_fe(path):
    print("Carregando dados...")
    df = pd.read_csv(path, delimiter=';')

    # >>> AQUI ESTÁ A CORREÇÃO: REMOÇÃO DE DUPLICATAS <<<
    qtd_antes = len(df)
    df = df.drop_duplicates(subset=['corrida'], keep='first')
    print(f"Duplicadas removidas: {qtd_antes - len(df)}. Total atual: {len(df)}")

    num_cols = ['al_min','c_min','c_max','n_min','s_min','sequencia','sequenciatotal',
                'vidapanela','tempociclo','tempovacuototal','temperaturaliquidus',
                'velocidadeobjetivada','velocidadereal','sugestaomodelolegado',
                'temperaturasaidafp','temperaturaobjetivada','temperaturamediareal']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    id_cols = ['corrida','secao','acoatual','qualidade','panela']
    for c in id_cols:
        if c in df.columns:
            df[c] = df[c].fillna('MISSING').astype(str)

    # FE
    df['C_Medio'] = (df['c_max'] + df['c_min']) / 2
    df['T_Liquid_Desvio'] = df['temperaturaobjetivada'] - df['temperaturaliquidus']
    df['Desvio_Legado_Target'] = df['sugestaomodelolegado'] - df['temperaturaobjetivada']
    df['Tempo_Sequencia'] = df['tempociclo'] * df['sequencia']
    
    return df

# ======================================
#   PREPARAÇÃO E FILTRO GOLDEN BATCH
# ======================================
df = load_and_fe('dados_fundo_do_amanha_evcomx.csv')

# Drop inicial
cols_to_drop_early = ['acoatual','corrida','secao','c_min','c_max',
                      'temperaturaliquidus','velocidadeobjetivada','velocidadereal']
df_clean = df.drop(columns=[c for c in cols_to_drop_early if c in df.columns], errors='ignore')

# 1. Split Temporal (Treino vs Validação)
df_valid = df_clean.iloc[-LIMITE_VALIDACAO:].copy()
df_treino_total = df_clean.iloc[:-LIMITE_VALIDACAO].copy()

# ==============================================================================
# >>> FILTRO GOLDEN BATCH (ELITE) <<<
# ==============================================================================

# Calculamos o erro real histórico (Legado ou Operador vs Objetivo)
# Aqui usamos a métrica de sucesso: O resultado final ficou bom?
diff_legado_tr = df_treino_total['sugestaomodelolegado'] - df_treino_total['temperaturasaidafp']
erro_real_treino = (df_treino_total['temperaturamediareal'] + diff_legado_tr) - df_treino_total['temperaturaobjetivada']

# Máscara de Sucesso: Só queremos aprender com corridas que ficaram entre -5 e +10
mask_gold = (erro_real_treino >= -DELTA_NEG) & (erro_real_treino <= DELTA_POS)

# Separamos o Treino em "Elite" (Acertos) e "Sujos" (Erros Operacionais)
df_treino_gold = df_treino_total[mask_gold].copy()
df_treino_dirty = df_treino_total[~mask_gold].copy()

print(f"Total Treino Histórico: {len(df_treino_total)}")
print(f"Treino ELITE (Usado para aprender): {len(df_treino_gold)} ({(len(df_treino_gold)/len(df_treino_total))*100:.1f}%)")
print(f"Treino SUJO (Ignorado no treino, usado para teste): {len(df_treino_dirty)}")

# -----------------------------------------------------------
# Lógica de USOU / NÃO USOU (Aplicada APENAS na Elite)
# -----------------------------------------------------------
df_treino_gold['usage'] = np.where(
    (df_treino_gold['sugestaomodelolegado'] - df_treino_gold[TARGET]).abs() <= LIMITE_USOU,
    'usou', 'Naousou'
)

df_usou = df_treino_gold[df_treino_gold['usage']=='usou'].drop(columns=['usage'])
df_op = df_treino_gold[df_treino_gold['usage']=='Naousou'].drop(columns=['usage'])

# Função auxiliar de split
def split_XY(df_part):
    Y = df_part[TARGET].copy()
    X = df_part.drop(columns=[TARGET] + COLS_TO_EXCLUDE_FROM_X, errors='ignore')
    return X, Y

# Dados de Treino (Só Elite)
X_usou, Y_usou = split_XY(df_usou)
X_op, Y_op = split_XY(df_op)

# Dados de Teste (Validação e Sujos)
X_val, Y_val = split_XY(df_valid)
X_dirty, Y_dirty = split_XY(df_treino_dirty)

# ======================================
#   ENCODERS (Treinados SÓ na Elite)
# ======================================
def encode_target_and_ohe(X_train, y_train, list_X_apply, woe_cols=WOE_COL, ohe_cols=OHE_COL):
    # Fit apenas no X_train (Elite)
    te = ce.TargetEncoder(cols=woe_cols)
    X_train_te = te.fit_transform(X_train, y_train)
    X_train_final = pd.get_dummies(X_train_te, columns=ohe_cols, drop_first=True)
    
    # Lista de outputs transformados
    outputs = [X_train_final]
    colunas_mestras = X_train_final.columns
    
    # Transform nos outros (Validação, Dirty, etc)
    for X_app in list_X_apply:
        X_app_te = te.transform(X_app.copy())
        X_app_final = pd.get_dummies(X_app_te, columns=ohe_cols, drop_first=True)
        # Garante alinhamento de colunas
        X_app_final = X_app_final.reindex(columns=colunas_mestras, fill_value=0)
        outputs.append(X_app_final)

    return outputs # Retorna [Treino, Apply1, Apply2...]

# Encoders para o modelo USOU
out_usou = encode_target_and_ohe(X_usou, Y_usou, [X_val, X_dirty])
X_usou_final, X_val_usou_final, X_dirty_usou_final = out_usou[0], out_usou[1], out_usou[2]

# Encoders para o modelo OP
out_op = encode_target_and_ohe(X_op, Y_op, [X_val, X_dirty])
X_op_final, X_val_op_final, X_dirty_op_final = out_op[0], out_op[1], out_op[2]

# ======================================
#   TREINO (XGBOOST)
# ======================================
best_params = {
    'n_estimators': 1199,
    'learning_rate': 0.148993, 
    'max_depth': 9, 
    'subsample': 0.7104, 
    'colsample_bytree': 0.6085, 
    'min_child_weight': 20, 
    'reg_lambda': 0.0057, 
    'alpha': 0.0011, 
    "objective": custom_asymmetric_loss,
    "booster": "gbtree",
    "tree_method": "hist", 
    "device": "cuda",        
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1
}

print("\n>>> Treinando Modelos na Elite (Golden Batch)...")
model_usou = XGBRegressor(**best_params)
model_op   = XGBRegressor(**best_params)

model_usou.fit(X_usou_final, Y_usou)
model_op.fit(X_op_final, Y_op)

# ======================================
#   AVALIAÇÃO
# ======================================
def avaliar_dataset(X_u, X_o, df_orig, Y_orig, nome_set):
    # Predições
    p_usou = model_usou.predict(X_u)
    p_op = model_op.predict(X_o)
    
    # Switch
    desvio = df_orig[SWITCH_FEATURE].abs()
    pred_final = np.where(desvio <= LIMITE_USOU, p_usou, p_op)
    
    # Baseline Legado
    pred_legado = df_orig['sugestaomodelolegado'].values
    
    # Cálculo Métricas Complexas (Legado vs Novo)
    # Legado
    diff_legado = df_orig['sugestaomodelolegado'] - df_orig[TARGET]
    erro_legado = (df_orig['temperaturamediareal'] + diff_legado) - df_orig['temperaturaobjetivada']
    
    # Novo
    diff_novo = pred_final - df_orig[TARGET]
    erro_novo = (df_orig['temperaturamediareal'] + diff_novo) - df_orig['temperaturaobjetivada']
    
    # Hit Rate
    hit_legado = np.mean((erro_legado >= -DELTA_NEG) & (erro_legado <= DELTA_POS)) * 100
    hit_novo = np.mean((erro_novo >= -DELTA_NEG) & (erro_novo <= DELTA_POS)) * 100
    
    # RMSE (Padrão)
    rmse_novo = np.sqrt(mean_squared_error(Y_orig, pred_final))
    rmse_legado = np.sqrt(mean_squared_error(Y_orig, pred_legado))
    
    print(f"\n--- {nome_set} ---")
    print(f"RMSE Legado: {rmse_legado:.2f} | Novo: {rmse_novo:.2f}")
    print(f"Acerto Legado: {hit_legado:.2f}%")
    print(f"Acerto Novo:   {hit_novo:.2f}%")
    
    if nome_set == "DADOS SUJOS (Recuperação)":
        plt.figure(figsize=(10,5))
        sns.histplot(erro_legado, color='red', alpha=0.3, label='Erro Legado', kde=True)
        sns.histplot(erro_novo, color='blue', alpha=0.3, label='Erro Novo', kde=True)
        plt.axvline(-5, color='k', linestyle='--')
        plt.axvline(10, color='k', linestyle='--')
        plt.title(f"Recuperação de Corridas Ruins - {nome_set}")
        plt.xlabel("Erro em relação ao Objetivo (°C)")
        plt.legend()
        plt.show()

# Executa
avaliar_dataset(X_val_usou_final, X_val_op_final, df_valid, Y_val, "VALIDAÇÃO (Futuro)")
avaliar_dataset(X_dirty_usou_final, X_dirty_op_final, df_treino_dirty, Y_dirty, "DADOS SUJOS (Recuperação)")
# ======================================
#   DIAGNÓSTICO FINAL (Coloque isso no fim do arquivo)
# ======================================
print("\n--- DIAGNÓSTICO DE VIÉS NA VALIDAÇÃO ---")

# 1. Recriar as predições da Validação (para podermos analisar)
p_usou_val = model_usou.predict(X_val_usou_final)
p_op_val = model_op.predict(X_val_op_final)

# Aplicar o Switch
desvio_val = df_valid[SWITCH_FEATURE].abs()
pred_final_val = np.where(desvio_val <= LIMITE_USOU, p_usou_val, p_op_val)

# 2. Calcular o Erro Complexo (Previsão - Objetivo)
# Fórmula: (TempMediaReal + (Pred - Real)) - Objetivo
diff_novo = pred_final_val - df_valid[TARGET] 
erro_novo_val = (df_valid['temperaturamediareal'] + diff_novo) - df_valid['temperaturaobjetivada']

# 3. Calcular a Média desse erro (Viés)
vies_validacao = np.mean(erro_novo_val)
print(f">>> VIÉS MÉDIO NA VALIDAÇÃO: {vies_validacao:.2f}°C")
# Se for positivo (ex: +3.0), o modelo está chutando acima.
# Se for negativo (ex: -3.0), o modelo está chutando abaixo.

# 4. Simulação: "E se a gente apenas corrigisse esse viés?"
erro_corrigido = erro_novo_val - vies_validacao
acerto_simulado = np.mean((erro_corrigido >= -DELTA_NEG) & (erro_corrigido <= DELTA_POS)) * 100

print(f">>> Se aplicarmos essa correção, o acerto sobe para: {acerto_simulado:.2f}%")

# 5. Plotar histograma para ver o deslocamento
plt.figure(figsize=(10, 6))
sns.histplot(erro_novo_val, color='blue', kde=True, label='Erro Atual (Descalibrado)', alpha=0.4)
# Plota a versão corrigida simulada (fantasma)
sns.histplot(erro_corrigido, color='green', kde=True, label='Erro Simulado (Se corrigir viés)', alpha=0.2)

plt.axvline(-5, color='r', linestyle='--', label='Limites (-5, +10)')
plt.axvline(10, color='r', linestyle='--')
plt.axvline(0, color='k', linestyle='-', alpha=0.3)
plt.title(f"Diagnóstico: Viés de {vies_validacao:.2f}°C")
plt.xlabel("Erro em relação ao Objetivo (°C)")
plt.legend()
plt.show()