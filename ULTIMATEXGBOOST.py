import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# --- FUNÇÃO DE PERDA ASSIMÉTRICA
def custom_asymmetric_loss(y_true, y_pred):
    resid = y_pred - y_true 
    alpha = 1.3 # Penaliza erro pra baixo
    grad = np.where(resid < 0, alpha * resid, 1.0 * resid) 
    hess = np.where(resid < 0, alpha, 1.0)
    return grad, hess

# --- 0. DEFINIÇÕES ---
TARGET = 'temperaturasaidafp'
WOE_COL = ['qualidade'] 
OHE_COL = ['panela']
LIMITE_VALIDACAO = 500
# LIMITE_USOU = 5  <-- NÃO PRECISA MAIS DISSO (SEM SWITCH)
DELTA_NEG = 5   
DELTA_POS = 10 

# Colunas que não entram no X
COLS_TO_EXCLUDE_FROM_X = [
    'sugestaomodelolegado', 
    'temperaturamediareal', 
    'temperaturaobjetivada'
]

# --- 1. CARREGAMENTO E FE ---
print("Carregando dados...")
df = pd.read_csv('dados_fundo_do_amanha_evcomx.csv', delimiter=';')

# Remove duplicatas garantindo integridade
df = df.drop_duplicates(subset=['corrida'], keep='first')
print(f"Dados após remoção de duplicatas: {df.shape[0]} linhas.")
num_cols = ['al_min', 'c_min', 'c_max', 'n_min', 's_min', 'sequencia', 'sequenciatotal', 
            'vidapanela', 'tempociclo', 'tempovacuototal', 'temperaturaliquidus', 
            'velocidadeobjetivada', 'velocidadereal', 'sugestaomodelolegado', 
            'temperaturasaidafp', 'temperaturaobjetivada', 'temperaturamediareal']
for col in num_cols: df[col] = pd.to_numeric(df[col], errors='coerce') 
id_cols = ['corrida', 'secao', 'acoatual', 'qualidade', 'panela']
for col in id_cols: df[col] = df[col].astype(str)

df['C_Medio'] = (df['c_max'] + df['c_min']) / 2
df['T_Liquid_Desvio'] = df['temperaturaobjetivada'] - df['temperaturaliquidus']
df['Tempo_Sequencia'] = df['tempociclo'] * df['sequencia']

cols_to_drop_early = ['acoatual', 'corrida', 'secao', 'c_min', 'c_max', 'temperaturaliquidus', 'velocidadeobjetivada', 'velocidadereal','s_min']
df_clean = df.drop(columns=cols_to_drop_early, errors='ignore')

# --- 2. SPLIT TREINO / VALIDAÇÃO (Temporal simples) ---
df_validacao_final = df_clean.iloc[-LIMITE_VALIDACAO:].copy()
df_treino_bruto = df_clean.iloc[:-LIMITE_VALIDACAO].copy()

# --- 3. SEPARAÇÃO X/Y ---
Y_treino = df_treino_bruto[TARGET]
X_treino = df_treino_bruto.drop(TARGET, axis=1).drop(COLS_TO_EXCLUDE_FROM_X, axis=1)

Y_validacao = df_validacao_final[TARGET]
X_validacao = df_validacao_final.drop(TARGET, axis=1).drop(COLS_TO_EXCLUDE_FROM_X, axis=1)

# --- 4. ENCODING (Sem vazamento de dados) ---

# Treina o encoder APENAS no treino
target_encoder = ce.TargetEncoder(cols=WOE_COL)
X_treino_te = target_encoder.fit_transform(X_treino, Y_treino)
X_treino_final = pd.get_dummies(X_treino_te, columns=OHE_COL, drop_first=True)

# Aplica o encoder na validação (usando o aprendizado do treino)
X_val_te = target_encoder.transform(X_validacao)
X_val_final = pd.get_dummies(X_val_te, columns=OHE_COL, drop_first=True)

# Alinhamento de colunas (Garante que a validação tenha as mesmas colunas do treino)
colunas_mestras = X_treino_final.columns
X_treino_final = X_treino_final.reindex(columns=colunas_mestras, fill_value=0)
X_val_final = X_val_final.reindex(columns=colunas_mestras, fill_value=0)

# --- 5. TREINO (MODELO ÚNICO) ---
params = {
        'n_estimators': 1199,
        'learning_rate': 0.148993, 
        'max_depth': 9, 
        'subsample': 0.710427, 
        'colsample_bytree': 0.608515, 
        'min_child_weight': 20, 
        'reg_lambda': 0.005723, 
        'alpha': 0.001123, 
        "objective": custom_asymmetric_loss, # Sua função customizada
        "booster": "gbtree",
        "tree_method": "hist", 
        "device": "cuda",        
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1
}

print("Treinando modelo único XGBoost...")
model = XGBRegressor(**params)
model.fit(X_treino_final, Y_treino)

# --- 6. PREDIÇÕES ---

# Validação
pred_val_raw = model.predict(X_val_final)

# Treino (Para calcular viés)
pred_treino_raw = model.predict(X_treino_final)

# --- 7. CORREÇÃO DE VIÉS (Calculado no Treino) ---
vies_medio = np.mean(pred_treino_raw - Y_treino)
print(f"Viés detectado no treino: {vies_medio:.2f} °C")

# Aplica correção
pred_treino_corrigido = pred_treino_raw - vies_medio
pred_val_corrigido = pred_val_raw - vies_medio

# --- 8. MÉTRICAS E PLOTAGEM ---

def analisar_erros_detalhado(erro_legado, erro_novo, nome_dataset):
    # 1. Criar DataFrame para análise
    df_analise = pd.DataFrame({
        'Erro_Legado': erro_legado,
        'Erro_Novo': erro_novo
    })
    
    # 2. Categorizar
    def classificar(val):
        if val < -DELTA_NEG: return f'1. Frio (< -{DELTA_NEG}°C)'
        elif val > DELTA_POS: return f'3. Quente (> +{DELTA_POS}°C)'
        else: return f'2. Acerto (-{DELTA_NEG} a +{DELTA_POS})'

    df_analise['Cat_Legado'] = df_analise['Erro_Legado'].apply(classificar)
    df_analise['Cat_Novo'] = df_analise['Erro_Novo'].apply(classificar)

    # 3. Contagem
    contagem_legado = df_analise['Cat_Legado'].value_counts().sort_index()
    contagem_novo = df_analise['Cat_Novo'].value_counts().sort_index()
    
    total = len(df_analise)
    perc_legado = (contagem_legado / total) * 100
    perc_novo = (contagem_novo / total) * 100

    # --- PLOTAGEM ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histograma
    sns.histplot(df_analise['Erro_Legado'], color='red', label='Legado', kde=True, ax=axes[0], alpha=0.3)
    sns.histplot(df_analise['Erro_Novo'], color='blue', label='Novo Modelo', kde=True, ax=axes[0], alpha=0.3)
    axes[0].axvline(DELTA_POS, color='k', linestyle='--', label='Limite Quente')
    axes[0].axvline(-DELTA_NEG, color='k', linestyle='--', label='Limite Frio')
    axes[0].set_title(f"Distribuição de Erros - {nome_dataset}")
    axes[0].legend()

    # Barras
    categorias = [f'1. Frio (< -{DELTA_NEG}°C)', f'2. Acerto (-{DELTA_NEG} a +{DELTA_POS})', f'3. Quente (> +{DELTA_POS}°C)']
    vals_legado = [perc_legado.get(c, 0) for c in categorias]
    vals_novo = [perc_novo.get(c, 0) for c in categorias]
    x = np.arange(len(categorias))
    
    axes[1].bar(x - 0.17, vals_legado, 0.35, label='Legado', color='red', alpha=0.7)
    rects = axes[1].bar(x + 0.17, vals_novo, 0.35, label='Novo', color='blue', alpha=0.7)
    axes[1].bar_label(rects, fmt='%.1f%%')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Frio', 'No Alvo', 'QUENTE'])
    axes[1].legend()
    axes[1].set_title(f"Porcentagem por Categoria - {nome_dataset}")

    plt.tight_layout()
    plt.show()

    # Tabela
    print(f"\n=== RELATÓRIO: {nome_dataset} (Total: {total}) ===")
    df_qtd = pd.DataFrame({
        'Qtd Legado': contagem_legado,
        'Qtd Novo': contagem_novo,
        '% Legado': perc_legado.round(2),
        '% Novo': perc_novo.round(2)
    }).fillna(0)
    print(df_qtd)
    print("="*60)

# --- CÁLCULO DA MÉTRICA COMPLEXA ---
# Fórmula: (TempMediaReal + (Pred - TempSaidaReal)) - TempObjetivo

# 1. TREINO
diff_legado_tr = df_treino_bruto['sugestaomodelolegado'] - df_treino_bruto['temperaturasaidafp']
erro_legado_tr = (df_treino_bruto['temperaturamediareal'] + diff_legado_tr) - df_treino_bruto['temperaturaobjetivada']

diff_novo_tr = pred_treino_corrigido - df_treino_bruto['temperaturasaidafp']
erro_novo_tr = (df_treino_bruto['temperaturamediareal'] + diff_novo_tr) - df_treino_bruto['temperaturaobjetivada']

# 2. VALIDAÇÃO
diff_legado_val = df_validacao_final['sugestaomodelolegado'] - df_validacao_final['temperaturasaidafp']
erro_legado_val = (df_validacao_final['temperaturamediareal'] + diff_legado_val) - df_validacao_final['temperaturaobjetivada']

diff_novo_val = pred_val_corrigido - df_validacao_final['temperaturasaidafp']
erro_novo_val = (df_validacao_final['temperaturamediareal'] + diff_novo_val) - df_validacao_final['temperaturaobjetivada']

# EXECUTA
analisar_erros_detalhado(erro_legado_tr, erro_novo_tr, "TREINO (7000+ dados)")
analisar_erros_detalhado(erro_legado_val, erro_novo_val, "VALIDAÇÃO (500 dados)")

