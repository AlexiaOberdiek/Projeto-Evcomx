import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

# --- 0. CONFIGURAÇÕES ---
TARGET = 'temperaturasaidafp'
DELTA_NEG = 5   
DELTA_POS = 10 

# --- CONFIGURAÇÃO DA VALIDAÇÃO (SLICE) ---
# O dataset tem 7485 linhas.
# Vamos pegar a fatia entre 6500 e 7000 para validar.
VAL_INICIO = 7000
VAL_FIM    = 7485   

# Colunas ignoradas
COLS_TO_EXCLUDE_FROM_X = ['sugestaomodelolegado', 'temperaturamediareal', 'temperaturaobjetivada']

# --- 1. CARREGAMENTO ---
print("Carregando dados...")
df = pd.read_csv('dados_fundo_do_amanha_evcomx.csv', delimiter=';')
df = df.drop_duplicates(subset=['corrida'], keep='first')

print(f"Total de linhas carregadas: {len(df)}")

# Coerção
num_cols = ['al_min', 'c_min', 'c_max', 'n_min', 's_min', 'sequencia', 'sequenciatotal', 
            'vidapanela', 'tempociclo', 'tempovacuototal', 'temperaturaliquidus', 
            'velocidadeobjetivada', 'velocidadereal', 'sugestaomodelolegado', 
            'temperaturasaidafp', 'temperaturaobjetivada', 'temperaturamediareal']
for col in num_cols: df[col] = pd.to_numeric(df[col], errors='coerce') 

cat_features = ['qualidade', 'panela'] 
for col in cat_features: df[col] = df[col].fillna('MISSING').astype(str)

# Feature Engineering
df['C_Medio'] = (df['c_max'] + df['c_min']) / 2
df['T_Liquid_Desvio'] = df['temperaturaobjetivada'] - df['temperaturaliquidus']
df['Tempo_Sequencia'] = df['tempociclo'] * df['sequencia']

cols_to_drop = ['corrida', 'c_min', 'c_max', 'temperaturaliquidus','secao','acoatual', 'velocidadeobjetivada', 'velocidadereal', 's_min']
df_clean = df.drop(columns=cols_to_drop, errors='ignore')

# ==============================================================================
# --- 2. SPLIT POR ÍNDICE (HOLD-OUT) ---
# ==============================================================================
print(f"\n--- Configurando Validação: Índices {VAL_INICIO} a {VAL_FIM} ---")

# Validação: Fatie exata que você pediu
df_validacao_final = df_clean.iloc[VAL_INICIO:VAL_FIM].copy()

# Treino: Todo o resto (Dados anteriores a 6500 + Dados posteriores a 7000)
df_treino_bruto = df_clean.drop(df_validacao_final.index).copy()

print(f"Tamanho Validação: {len(df_validacao_final)}")
print(f"Tamanho Treino: {len(df_treino_bruto)}")

# --- 3. SEPARAÇÃO X/Y ---
Y_treino = df_treino_bruto[TARGET]
X_treino = df_treino_bruto.drop(TARGET, axis=1).drop(COLS_TO_EXCLUDE_FROM_X, axis=1)

Y_validacao = df_validacao_final[TARGET]
X_validacao = df_validacao_final.drop(TARGET, axis=1).drop(COLS_TO_EXCLUDE_FROM_X, axis=1)

# --- 4. TREINO CATBOOST ---
params = {
    'iterations': 2000,
    'learning_rate': 0.02,
    'depth': 6,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'task_type': 'GPU',
    'verbose': 500,
    'allow_writing_files': False
}

print("\nTreinando CatBoost...")
model = CatBoostRegressor(**params)
model.fit(X_treino, Y_treino, cat_features=cat_features)

# --- 5. CORREÇÃO DE VIÉS ---
pred_treino_raw = model.predict(X_treino)
vies_medio = np.mean(pred_treino_raw - Y_treino)

print(f"\n>>> Viés detectado no treino: {vies_medio:.2f} °C")

pred_treino_corrigido = pred_treino_raw - vies_medio
pred_val_raw = model.predict(X_validacao)
pred_val_corrigido = pred_val_raw - vies_medio

# --- 6. ANÁLISE ---
def analisar_erros_detalhado(erro_legado, erro_novo, nome_dataset):
    df_analise = pd.DataFrame({'Erro_Legado': erro_legado, 'Erro_Novo': erro_novo})
    
    def classificar(val):
        if val < -DELTA_NEG: return f'1. Frio (< -{DELTA_NEG}°C)'
        elif val > DELTA_POS: return f'3. Quente (> +{DELTA_POS}°C)'
        else: return f'2. Acerto (-{DELTA_NEG} a +{DELTA_POS})'

    df_analise['Cat_Legado'] = df_analise['Erro_Legado'].apply(classificar)
    df_analise['Cat_Novo'] = df_analise['Erro_Novo'].apply(classificar)

    contagem_legado = df_analise['Cat_Legado'].value_counts().sort_index()
    contagem_novo = df_analise['Cat_Novo'].value_counts().sort_index()
    perc_legado = (contagem_legado / len(df_analise)) * 100
    perc_novo = (contagem_novo / len(df_analise)) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(df_analise['Erro_Legado'], color='red', label='Legado', kde=True, ax=axes[0], alpha=0.3)
    sns.histplot(df_analise['Erro_Novo'], color='green', label='CatBoost', kde=True, ax=axes[0], alpha=0.3)
    axes[0].axvline(DELTA_POS, color='k', linestyle='--'); axes[0].axvline(-DELTA_NEG, color='k', linestyle='--')
    axes[0].set_title(f"Distribuição - {nome_dataset}"); axes[0].legend()

    categorias = [f'1. Frio (< -{DELTA_NEG}°C)', f'2. Acerto', f'3. Quente (> +{DELTA_POS}°C)']
    vals_legado = [perc_legado.get(c, 0) for c in categorias]
    vals_novo = [perc_novo.get(c, 0) for c in categorias]
    x = np.arange(len(categorias))
    
    axes[1].bar(x - 0.17, vals_legado, 0.35, label='Legado', color='red', alpha=0.7)
    rects = axes[1].bar(x + 0.17, vals_novo, 0.35, label='CatBoost', color='green', alpha=0.7)
    axes[1].bar_label(rects, fmt='%.1f%%')
    axes[1].set_xticks(x); axes[1].set_xticklabels(['Frio', 'No Alvo', 'QUENTE'])
    axes[1].legend(); axes[1].set_title(f"Performance - {nome_dataset}")
    plt.show()

    print(f"\n=== RELATÓRIO: {nome_dataset} ===")
    print(pd.DataFrame({'% Legado': perc_legado, '% Novo': perc_novo}).fillna(0).round(2))

# Cálculo Métricas
diff_legado_val = df_validacao_final['sugestaomodelolegado'] - df_validacao_final['temperaturasaidafp']
erro_legado_val = (df_validacao_final['temperaturamediareal'] + diff_legado_val) - df_validacao_final['temperaturaobjetivada']
diff_novo_val = pred_val_corrigido - df_validacao_final['temperaturasaidafp']
erro_novo_val = (df_validacao_final['temperaturamediareal'] + diff_novo_val) - df_validacao_final['temperaturaobjetivada']

analisar_erros_detalhado(erro_legado_val, erro_novo_val, f"VALIDAÇÃO (Linhas {VAL_INICIO}-{VAL_FIM})")
# ==============================================================================
# --- 8. VALIDAÇÃO TEMPORAL (TIMELINE DE PERFORMANCE) ---
# ==============================================================================
print("\n--- Gerando Análise Temporal por Lotes (Bins) ---")

# 1. Preparar os dados (Garantir ordem cronológica)
# Vamos usar o df_clean original que tem tudo, resetando o index para garantir ordem 0, 1, 2...
df_timeline = df_clean.copy().reset_index(drop=True)

# 2. Gerar Predições para TUDO (Histórico Completo)
X_timeline = df_timeline.drop(TARGET, axis=1).drop(COLS_TO_EXCLUDE_FROM_X, axis=1)

# Predição do CatBoost
preds_brutas = model.predict(X_timeline)

# IMPORTANTE: Aplicar a correção de viés que calculamos no treino
# (Assumindo que você já calculou 'vies_medio' no passo anterior)
preds_corrigidas = preds_brutas - vies_medio

# 3. Calcular os Erros (Complexos) linha a linha
# Legado
diff_legado = df_timeline['sugestaomodelolegado'] - df_timeline['temperaturasaidafp']
erro_legado = (df_timeline['temperaturamediareal'] + diff_legado) - df_timeline['temperaturaobjetivada']

# Novo Modelo
diff_novo = preds_corrigidas - df_timeline['temperaturasaidafp']
erro_novo = (df_timeline['temperaturamediareal'] + diff_novo) - df_timeline['temperaturaobjetivada']

# 4. Criar Colunas de "Placar" (1 = Aconteceu, 0 = Não aconteceu)
# Acerto: Entre -5 e +10
df_timeline['Legado_Acerto'] = ((erro_legado >= -DELTA_NEG) & (erro_legado <= DELTA_POS)).astype(int)
df_timeline['Novo_Acerto']   = ((erro_novo >= -DELTA_NEG) & (erro_novo <= DELTA_POS)).astype(int)

# Erro Crítico (Frio): < -5
df_timeline['Legado_Frio'] = (erro_legado < -DELTA_NEG).astype(int)
df_timeline['Novo_Frio']   = (erro_novo < -DELTA_NEG).astype(int)

# Erro Crítico (Quente): > +10
df_timeline['Legado_Quente'] = (erro_legado > DELTA_POS).astype(int)
df_timeline['Novo_Quente']   = (erro_novo > DELTA_POS).astype(int)

# 5. Criar os Bins (Fatias de 500 em 500 corridas)
# Cria bins: 0-500, 500-1000, 1000-1500... até o final do DF
bins = list(range(0, len(df_timeline) + 500, 500))
labels = [f"{b}-{b+500}" for b in bins[:-1]]

df_timeline['Faixa_Corridas'] = pd.cut(df_timeline.index, bins=bins, labels=labels)

# 6. Agrupar e Calcular Médias (Transformar em Porcentagem)
df_grouped = df_timeline.groupby('Faixa_Corridas')[
    ['Legado_Acerto', 'Novo_Acerto', 'Legado_Frio', 'Novo_Frio', 'Legado_Quente', 'Novo_Quente']
].mean() * 100

# Remover bins vazios (se houver)
df_grouped = df_grouped.dropna()

# --- 7. VISUALIZAÇÃO GRÁFICA (LINHA DO TEMPO) ---
plt.figure(figsize=(18, 6))

# Plot da Taxa de Acerto
plt.plot(df_grouped.index, df_grouped['Legado_Acerto'], marker='o', label='Legado', color='red', linestyle='--', linewidth=2)
plt.plot(df_grouped.index, df_grouped['Novo_Acerto'], marker='o', label='Novo Modelo', color='green', linewidth=3)

plt.title("Evolução da Taxa de Acerto ao Longo do Tempo (Bins de 500 corridas)")
plt.ylabel("Taxa de Acerto (%)")
plt.xlabel("Faixa de Corridas (Ordem Cronológica)")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# --- 8. EXIBIR TABELA RESUMIDA ---
print("\n=== DESEMPENHO POR FAIXA DE 500 CORRIDAS ===")
# Vamos mostrar só as colunas de Acerto para não poluir
print(df_grouped[['Legado_Acerto', 'Novo_Acerto']].round(2))

# Análise de Estabilidade
std_novo = df_grouped['Novo_Acerto'].std()
print(f"\nEstabilidade (Desvio Padrão dos Acertos do Novo Modelo): {std_novo:.2f}")
if std_novo < 5:
    print(">>> O modelo é MUITO ESTÁVEL ao longo do tempo.")
else:
    print(">>> O modelo apresenta oscilações de desempenho.")
    # Configuração de Estilo para ficar bonito
sns.set_style("whitegrid")

# Criar uma figura com 3 gráficos (subplots) um em cima do outro
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

# Cores
cor_legado = 'tab:red'
cor_novo = 'tab:green' # ou 'tab:blue' se preferir

# --- GRÁFICO 1: TAXA DE ACERTO (O MAIS IMPORTANTE) ---
axes[0].plot(df_grouped.index, df_grouped['Legado_Acerto'], marker='o', linestyle='--', color=cor_legado, label='Legado', linewidth=2)
axes[0].plot(df_grouped.index, df_grouped['Novo_Acerto'], marker='o', linestyle='-', color=cor_novo, label='Novo Modelo', linewidth=3)
axes[0].set_ylabel('Taxa de Acerto (%)')
axes[0].set_title('1. Evolução da Taxa de Acerto (-5°C a +10°C)', fontsize=14, fontweight='bold')
axes[0].legend(loc='lower right')
# Pinta a área onde o Novo Modelo ganha
axes[0].fill_between(df_grouped.index, df_grouped['Legado_Acerto'], df_grouped['Novo_Acerto'], 
                     where=(df_grouped['Novo_Acerto'] > df_grouped['Legado_Acerto']), 
                     color='green', alpha=0.1, interpolate=True)

# --- GRÁFICO 2: ERRO FRIO (RISCO CRÍTICO) ---
axes[1].plot(df_grouped.index, df_grouped['Legado_Frio'], marker='x', linestyle=':', color=cor_legado, label='Legado')
axes[1].plot(df_grouped.index, df_grouped['Novo_Frio'], marker='x', linestyle='-', color=cor_novo, label='Novo Modelo')
axes[1].set_ylabel('% Corridas Frias (< -5°C)')
axes[1].set_title('2. Risco de Aço Frio (Quanto menor, melhor)', fontsize=12)
axes[1].legend()

# --- GRÁFICO 3: ERRO QUENTE (CUSTO/ENERGIA) ---
axes[2].plot(df_grouped.index, df_grouped['Legado_Quente'], marker='s', linestyle=':', color=cor_legado, label='Legado')
axes[2].plot(df_grouped.index, df_grouped['Novo_Quente'], marker='s', linestyle='-', color=cor_novo, label='Novo Modelo')
axes[2].set_ylabel('% Corridas Quentes (> +10°C)')
axes[2].set_title('3. Tendência de Superaquecimento (Quanto menor, melhor)', fontsize=12)
axes[2].set_xlabel('Faixa de Corridas (Lotes de 500)', fontsize=12)
axes[2].legend()

# Ajustes Finais
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()