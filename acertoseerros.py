import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================
ARQUIVO_CSV = 'dados_fundo_do_amanha_evcomx.csv'
LIMITE_USOU = 5 
DELTA_NEG = 5   
DELTA_POS = 10  

# --- PALETA DE CORES SEMÂNTICA ---
# Aqui definimos a "Cara" do gráfico
CORES_FAIXAS = {
    '1. Extremo Frio (< -10)': '#003366', # Azul Marinho (Congelando)
    '2. Frio (-10 a -5)':      '#4287f5', # Azul Claro (Frio)
    '3. Acerto (-5 a +10)':    '#28a745', # Verde (Sucesso)
    '4. Quente (+10 a +20)':   '#fd7e14', # Laranja (Quente)
    '5. Extremo Calor (> +20)': '#dc3545' # Vermelho (Fogo)
}

# ==============================================================================
# 2. CARREGAMENTO E PREPARAÇÃO
# ==============================================================================
print("--- Carregando Dados Históricos ---")
df = pd.read_csv(ARQUIVO_CSV, delimiter=';')
df = df.drop_duplicates(subset=['corrida'], keep='first')

cols_num = ['sugestaomodelolegado', 'temperaturasaidafp', 'temperaturaobjetivada', 'temperaturamediareal']
for c in cols_num:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df['diff_uso'] = (df['sugestaomodelolegado'] - df['temperaturasaidafp']).abs()
df['quem_operou'] = np.where(df['diff_uso'] <= LIMITE_USOU, 'Legado (Seguiu)', 'Operador (Interviu)')

print(f"Total de Corridas: {len(df)}")
def plotar_pizza_adesao(df_input):
    # Contagem
    counts = df_input['quem_operou'].value_counts()
    
    # Definição de Cores (Para bater com os gráficos de histograma abaixo)
    # Azul = Legado, Laranja = Operador
    cores_map = {'Legado (Seguiu)': 'blue', 'Operador (Interviu)': 'orange'}
    lista_cores = [cores_map.get(x, 'gray') for x in counts.index]
    
    # Destacar a fatia menor (Explode)
    explode = [0.05] * len(counts)

    # Plotagem
    plt.figure(figsize=(10, 7))
    wedges, texts, autotexts = plt.pie(
        counts, 
        labels=counts.index, 
        autopct='%1.1f%%', 
        startangle=140,
        colors=lista_cores, 
        explode=explode, 
        shadow=True,
        textprops=dict(color="black")
    )
    
    # Estilização dos textos
    plt.setp(texts, size=12, weight="bold")
    plt.setp(autotexts, size=14, weight="bold", color="white")
    
    plt.title("Taxa de Adesão: Quantas vezes o Legado foi seguido?", fontsize=16, fontweight='bold')
    plt.axis('equal') # Garante círculo perfeito
    plt.tight_layout()
    plt.show()
    
    print("\n>>> ESTATÍSTICAS DE ADESÃO")
    print(counts)
    print("-" * 60)

# Chama a função da pizza agora
plotar_pizza_adesao(df)
# ==============================================================================
# 3. FUNÇÕES DE CÁLCULO
# ==============================================================================
def get_erro_real(df_input):
    return df_input['temperaturamediareal'] - df_input['temperaturaobjetivada']

def get_erro_legado_simulado(df_input):
    diff_legado = df_input['sugestaomodelolegado'] - df_input['temperaturasaidafp']
    return (df_input['temperaturamediareal'] + diff_legado) - df_input['temperaturaobjetivada']

# ==============================================================================
# 4. FUNÇÃO DE PLOTAGEM (VISUAL MELHORADO)
# ==============================================================================
def plotar_cenario(series_erro, titulo, cor_histograma):
    # 1. Categorizar
    def categorizar(val):
        if val < -10:           return '1. Extremo Frio (< -10)'
        elif val >= -10 and val < -5: return '2. Frio (-10 a -5)'
        elif val >= -5 and val <= 10: return '3. Acerto (-5 a +10)'
        elif val > 10 and val <= 20:  return '4. Quente (+10 a +20)'
        else:                   return '5. Extremo Calor (> +20)'

    cats = series_erro.apply(categorizar)
    stats = cats.value_counts(normalize=True) * 100
    
    # Ordem fixa
    ordem = ['1. Extremo Frio (< -10)', '2. Frio (-10 a -5)', '3. Acerto (-5 a +10)', '4. Quente (+10 a +20)', '5. Extremo Calor (> +20)']
    valores = [stats.get(c, 0) for c in ordem]
    
    # Prepara lista de cores para as barras
    lista_cores = [CORES_FAIXAS[cat] for cat in ordem]
    
    # Labels curtos para o eixo X
    labels_x = ['Ext. Frio\n(< -10)', 'Frio\n(-10 a -5)', 'ACERTO\n(-5 a +10)', 'Quente\n(+10 a +20)', 'Ext. Calor\n(> +20)']

    # --- PLOTAGEM ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    # GRÁFICO 1: DISTRIBUIÇÃO (Histograma)
    # Mantemos uma cor única aqui para representar o cenário (ex: Laranja para Operador)
    sns.histplot(series_erro, color=cor_histograma, kde=True, ax=axes[0], element="step", alpha=0.5)
    
    # Linhas de referência coloridas
    axes[0].axvline(-5, color=CORES_FAIXAS['3. Acerto (-5 a +10)'], linestyle='--', linewidth=2, label='Meta (-5)')
    axes[0].axvline(10, color=CORES_FAIXAS['3. Acerto (-5 a +10)'], linestyle='--', linewidth=2, label='Meta (+10)')
    axes[0].axvline(-10, color=CORES_FAIXAS['1. Extremo Frio (< -10)'], linestyle=':', linewidth=2, label='Crítico Frio')
    axes[0].axvline(20, color=CORES_FAIXAS['5. Extremo Calor (> +20)'], linestyle=':', linewidth=2, label='Crítico Calor')
    axes[0].set_title(f'Distribuição do Erro de Temperatura - {titulo}', fontsize=14)
    axes[0].set_xlabel("Erro em relação ao Objetivo (°C)", fontsize=12)
    axes[0].set_xlim(-50, 50)
    axes[0].grid(True, alpha=0.2)
    axes[0].legend()

    # GRÁFICO 2: BARRAS COLORIDAS
    bars = axes[1].bar(labels_x, valores, color=lista_cores, alpha=0.85, edgecolor='black')
    
    axes[1].set_title(f"Classificação: {titulo}", fontsize=14)
    axes[1].set_ylabel("% das Corridas", fontsize=12)
    
    # Adiciona os valores em cima das barras
    axes[1].bar_label(bars, fmt='%.1f%%', padding=3, fontsize=11, fontweight='bold')
    
    # --- LEGENDA PERSONALIZADA ---
    # Cria "manchas" de cor para a legenda explicar o que é cada cor
    legend_elements = [
        Patch(facecolor=CORES_FAIXAS['1. Extremo Frio (< -10)'], label='Perda Total (< -10°C)'),
        Patch(facecolor=CORES_FAIXAS['2. Frio (-10 a -5)'],      label='Risco Operacional (-10 a -5°C)'),
        Patch(facecolor=CORES_FAIXAS['3. Acerto (-5 a +10)'],    label='META Atingida'),
        Patch(facecolor=CORES_FAIXAS['4. Quente (+10 a +20)'],   label='Desperdício (+10 a +20°C)'),
        Patch(facecolor=CORES_FAIXAS['5. Extremo Calor (> +20)'], label='Dano ao Refratário (> +20°C)'),
    ]
    
    axes[1].legend(handles=legend_elements, loc='upper right', title="Legenda de Cores")
    axes[1].grid(axis='y', alpha=0.2)

    plt.tight_layout()
    plt.show()
    
    # Print Resumo
    print(f"\n>>> {titulo}")
    print(f"Média do Erro: {series_erro.mean():.2f}°C")
    print("-" * 60)

# ==============================================================================
# 5. EXECUÇÃO
# ==============================================================================

# 1. QUANDO O OPERADOR SEGUIU O LEGADO
df_seguiu = df[df['quem_operou'] == 'Legado (Seguiu)']
erro_cenario_1 = get_erro_real(df_seguiu)
plotar_cenario(erro_cenario_1, "Cenário 1: Operador SEGUIU o Legado", "blue")

# 2. QUANDO O OPERADOR INTERVIU (Realidade)
df_interviu = df[df['quem_operou'] == 'Operador (Interviu)']
erro_cenario_2 = get_erro_real(df_interviu)
plotar_cenario(erro_cenario_2, "Cenário 2: Operador INTERVIU (O que aconteceu)", "orange")

# 3. QUANDO O OPERADOR INTERVIU (Se ele tivesse usado o Legado)
erro_cenario_3 = get_erro_legado_simulado(df_interviu)
plotar_cenario(erro_cenario_3, "Cenário 3: E se o Operador NÃO tivesse intervindo?", "purple")

# 4. REALIDADE TOTAL (Histórico Completo)
erro_cenario_4 = get_erro_real(df)
plotar_cenario(erro_cenario_4, "Cenário 4: Histórico Total da Fábrica", "gray")

# 5. LEGADO PURO (Simulação Total)
erro_cenario_5 = get_erro_legado_simulado(df)
plotar_cenario(erro_cenario_5, "Cenário 5: Simulação (Se usasse 100% Legado)", "red")