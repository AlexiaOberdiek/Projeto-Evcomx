import numpy as np
import pandas as pd
import optuna
from xgboost import XGBRegressor
import category_encoders as ce
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ======================================
#   CUSTOM ASYMMETRIC LOSS
# ======================================
def custom_asymmetric_loss(y_true, y_pred):
    resid = y_pred - y_true
    alpha = 1.2
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
COLS_TO_EXCLUDE_FROM_X = ['sugestaomodelolegado', 'temperaturamediareal']
SWITCH_FEATURE = 'Desvio_Legado_Target'
LIMITE_VALIDACAO = 500
LIMITE_USOU = 5
DELTA_NEG = 5
DELTA_POS = 10


# ======================================
#   LOAD + FEATURE ENGINEERING
# ======================================
def load_and_fe(path):
    df = pd.read_csv(path, delimiter=';')

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
            df[c] = df[c].astype(str)

    df['C_Medio'] = (df['c_max'] + df['c_min']) / 2
    df['T_Liquid_Desvio'] = df['temperaturaobjetivada'] - df['temperaturaliquidus']
    df['Desvio_Legado_Target'] = df['sugestaomodelolegado'] - df['temperaturaobjetivada']
    df['Tempo_Sequencia'] = df['tempociclo'] * df['sequencia']

    return df


# ======================================
#   PREPARAÇÃO E SEGMENTAÇÃO
# ======================================
df = load_and_fe('dados_fundo_do_amanha_evcomx.csv')

cols_to_drop_early = ['acoatual','corrida','secao','c_min','c_max',
                      'temperaturaliquidus','temperaturaobjetivada',
                      'velocidadeobjetivada','velocidadereal']
df_clean = df.drop(columns=[c for c in cols_to_drop_early if c in df.columns], errors='ignore')

df_valid = df_clean.iloc[-LIMITE_VALIDACAO:].copy()
df_treino = df_clean.iloc[:-LIMITE_VALIDACAO].copy()

df_treino['usage'] = np.where(
    (df_treino['sugestaomodelolegado'] - df_treino[TARGET]).abs() <= LIMITE_USOU,
    'usou', 'Naousou'
)

df_usou = df_treino[df_treino['usage']=='usou'].drop(columns=['usage'])
df_op = df_treino[df_treino['usage']=='Naousou'].drop(columns=['usage'])


def split_XY(df_part):
    Y = df_part[TARGET].copy()
    X = df_part.drop(columns=[TARGET] + COLS_TO_EXCLUDE_FROM_X, errors='ignore')
    return X, Y

X_usou, Y_usou = split_XY(df_usou)
X_op, Y_op = split_XY(df_op)
X_val, Y_val = split_XY(df_valid)


# ======================================
#   ENCODERS
# ======================================
def encode_target_and_ohe(X_train, y_train, X_apply, woe_cols=WOE_COL, ohe_cols=OHE_COL):
    te = ce.TargetEncoder(cols=woe_cols)
    X_train_te = te.fit_transform(X_train, y_train)
    X_train_final = pd.get_dummies(X_train_te, columns=ohe_cols, drop_first=True)

    X_apply_te = te.transform(X_apply.copy())
    X_apply_final = pd.get_dummies(X_apply_te, columns=ohe_cols, drop_first=True)

    return te, X_train_final, X_apply_final

te_usou, X_usou_te, X_val_usou_te = encode_target_and_ohe(X_usou, Y_usou, X_val)
te_op, X_op_te, X_val_op_te = encode_target_and_ohe(X_op, Y_op, X_val)

colunas_mestras = X_usou_te.columns.union(X_op_te.columns)
X_usou_final = X_usou_te.reindex(columns=colunas_mestras, fill_value=0)
X_op_final   = X_op_te.reindex(columns=colunas_mestras, fill_value=0)
X_val_usou_final = X_val_usou_te.reindex(columns=colunas_mestras, fill_value=0)
X_val_op_final   = X_val_op_te.reindex(columns=colunas_mestras, fill_value=0)


# ======================================
#   FUNÇÃO OPTUNA
# ======================================
def treinar_com_optuna(X_train, Y_train, trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),

        "objective": custom_asymmetric_loss,
        "booster": "dart",
        "tree_method": "hist", 
        "device": "cuda",        
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1
    }

    model = XGBRegressor(**params)

    model.fit(
        X_train, Y_train,
        eval_set=[(X_val_usou_final, Y_val)],
        verbose=False
    )

    preds = model.predict(X_val_usou_final)
    return mean_squared_error(Y_val, preds)


# ======================================
#  OPTUNA EXECUÇÃO (para o modelo USOU)
# ======================================
study = optuna.create_study(direction="minimize")
study.optimize(lambda trial: treinar_com_optuna(X_usou_final, Y_usou, trial), n_trials=20)

best_params = study.best_params
best_params.update({
    "objective": custom_asymmetric_loss,
    "booster": "gbrtree",
    "tree_method": "gpu_hist",
    "device": "cuda",
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1
})


# ======================================
#   TREINO FINAL DOS 2 MODELOS
# ======================================
model_usou = XGBRegressor(**best_params)
model_op   = XGBRegressor(**best_params)

model_usou.fit(X_usou_final, Y_usou)
model_op.fit(X_op_final, Y_op)


# ======================================
#   PREDIÇÃO + SWITCH
# ======================================
pred_usou = model_usou.predict(X_val_usou_final)
pred_op   = model_op.predict(X_val_op_final)
desvio_val = df_valid[SWITCH_FEATURE].abs()

pred_final = np.where(desvio_val <= LIMITE_USOU, pred_usou, pred_op)
pred_legado = df_valid['sugestaomodelolegado'].values
Yv = Y_val.values


# ======================================
#   MÉTRICAS
# ======================================
mse_novo = mean_squared_error(Yv, pred_final)
mse_legado = mean_squared_error(Yv, pred_legado)
rmse_novo = np.sqrt(mse_novo)
rmse_legado = np.sqrt(mse_legado)

r2_novo = r2_score(Yv, pred_final)
r2_legado = r2_score(Yv, pred_legado)

erro_novo = pred_final - Yv
erro_legado = pred_legado - Yv

hit_novo = np.mean((erro_novo >= -DELTA_NEG) & (erro_novo <= DELTA_POS)) * 100
hit_legado = np.mean((erro_legado >= -DELTA_NEG) & (erro_legado <= DELTA_POS)) * 100


# ======================================
#   RESULTADOS
# ======================================
print("\n--- RESULTADOS FINAIS ---")
print(f"RMSE Legado: {rmse_legado:.2f} °C")
print(f"RMSE Novo Modelo: {rmse_novo:.2f} °C")
print(f"R² Legado: {r2_legado:.4f}")
print(f"R² Novo: {r2_novo:.4f}")
print(f"Acerto Legado: {hit_legado:.2f}%")
print(f"Acerto Novo: {hit_novo:.2f}%")
