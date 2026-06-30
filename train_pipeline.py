# -*- coding: utf-8 -*-
"""
Replica FIEL del pipeline del notebook mortality_prediction.ipynb.
Entrena los 5 modelos, selecciona XGBoost, hace selección de variables
(importancia 95% + correlación 0.65 + SHAP top 25) y guarda los artefactos
que consumirá la app web. NO cambia ningún modelo ni hiperparámetro.
"""
import json, pickle, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

import os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "training_v2.csv")   # coloca training_v2.csv junto a este script
OUT = os.path.join(HERE, "artifacts")
os.makedirs(OUT, exist_ok=True)

print(">> Leyendo dataset…")
df = pd.read_csv(SRC)
print("   shape inicial:", df.shape)

# ----------------------------------------------------------------------
# CORRECCIÓN DE TIPOS  (idéntico al notebook)
# ----------------------------------------------------------------------
df['apache_2_diagnosis'] = df['apache_2_diagnosis'].astype(str)
df['apache_3j_diagnosis'] = df['apache_3j_diagnosis'].astype(str)
df['apache_2_diagnosis'] = df['apache_2_diagnosis'].astype(str).str.split('.').str[0]

for c in ['age','arf_apache','gcs_eyes_apache','gcs_motor_apache','gcs_unable_apache',
          'gcs_verbal_apache','intubated_apache','ventilated_apache','aids','cirrhosis',
          'diabetes_mellitus','hepatic_failure','immunosuppression','leukemia','lymphoma',
          'solid_tumor_with_metastasis']:
    df[c] = df[c].astype('Int64')

for c in ['encounter_id','patient_id','hospital_id','icu_id']:
    df[c] = df[c].astype(str)

# ----------------------------------------------------------------------
# OUTLIERS  (sólo pre_icu_los_days negativos)
# ----------------------------------------------------------------------
df = df[df['pre_icu_los_days'] >= 0]

# ----------------------------------------------------------------------
# NULOS  (idéntico al notebook)
# ----------------------------------------------------------------------
def porcentaje_nulos(d):
    n = d.isnull().sum()
    return pd.DataFrame({'Columna': n.index, 'Valores Nulos': n.values,
                         '% Nulos': (n/len(d)*100).values})

resumen = porcentaje_nulos(df)
df.drop(columns=resumen[resumen['% Nulos'] > 8]['Columna'].tolist(), inplace=True)
print("   shape tras drop >8% nulos:", df.shape)

df.dropna(subset=['cirrhosis'], inplace=True)
df.dropna(subset=['apache_2_diagnosis'], inplace=True)
df.dropna(subset=['gcs_eyes_apache'], inplace=True)

keep_impute = ['d1_glucose_max','d1_glucose_min','h1_sysbp_noninvasive_min',
               'h1_sysbp_noninvasive_max','h1_diasbp_noninvasive_min','h1_diasbp_noninvasive_max']
df.dropna(subset=df.columns.difference(keep_impute), inplace=True)
for c in keep_impute:
    df[c] = df[c].fillna(-99999)
print("   shape tras limpieza nulos:", df.shape)

# Guardamos una versión "limpia legible" para el Panel A ANTES de dummies
df_panelA = df.copy()

# ----------------------------------------------------------------------
# DUMMIES  (idéntico al notebook)
# ----------------------------------------------------------------------
categorias = {
    'ethnicity':['Caucasian','African American','Hispanic','Other/Unknown'],
    'gender':['M','F'],
    'icu_admit_source':['Accident & Emergency','Operating Room / Recovery','Floor','Other Hospital'],
    'icu_stay_type':['admit','transfer','readmit'],
    'icu_type':['Med-Surg ICU','MICU','Neuro ICU','CCU-CTICU','SICU'],
    'apache_2_diagnosis':['113','301','302','112','308'],
    'apache_3j_diagnosis':['501.05','107.01','403.01','106.01','703.03'],
    'apache_2_bodysystem':['Cardiovascular','Neurologic','Respiratory','Gastrointestinal','Metabolic'],
    'apache_3j_bodysystem':['Cardiovascular','Neurological','Sepsis','Respiratory','Gastrointestinal'],
}
df_final = df.copy()

def reemplazar(v, cats): return v if v in cats else 'OTROS'

for col, cats in categorias.items():
    df[col] = df[col].apply(reemplazar, cats=cats)
    cats2 = cats + ['OTROS']
    dummy = pd.get_dummies(df_final[col][df_final[col].isin(cats2)], prefix=col, dtype='Int64')
    df_final = pd.concat([df_final, dummy], axis=1)
    df_final = df_final.drop(col, axis=1)

df_final.fillna(0, inplace=True)
df_final.drop(columns=['gender_M'], inplace=True)
df_final.rename(columns={'gender_F':'gender'}, inplace=True)

vars_predict = list(df_final.columns)
for c in ['hospital_death','encounter_id','patient_id','hospital_id','icu_id']:
    vars_predict.remove(c)
print("   nº variables predictoras:", len(vars_predict))

# ----------------------------------------------------------------------
# MODELOS
# ----------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import randint
import xgboost as xgb
from catboost import CatBoostClassifier

y = df_final['hospital_death'].astype(int)
metrics_rows = []

def evaluar(nombre, y_test, y_pred, y_prob):
    return {
        "Modelo": nombre,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
    }

# --- Regresión logística (con estandarización, igual que el notebook) ---
print(">> Regresión Logística…")
X = df_final[vars_predict]
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
Xtr_r, ytr_r = RandomOverSampler(random_state=42).fit_resample(Xtr, ytr)
m_log = LogisticRegression(max_iter=1000, solver='liblinear').fit(Xtr_r, ytr_r)
metrics_rows.append(evaluar("Regresión Logística", yte, m_log.predict(Xte), m_log.predict_proba(Xte)[:,1]))

# --- Árbol de decisión ---
print(">> Árbol de Decisión…")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
Xtr_r, ytr_r = RandomOverSampler(random_state=42).fit_resample(Xtr, ytr)
rs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42),
        {'max_depth':randint(3,10),'min_samples_split':randint(2,20),'min_samples_leaf':randint(1,20)},
        n_iter=2, cv=3, scoring='roc_auc', random_state=42).fit(Xtr_r, ytr_r)
m_dt = rs.best_estimator_
metrics_rows.append(evaluar("Árbol de Decisión", yte, m_dt.predict(Xte), m_dt.predict_proba(Xte)[:,1]))

# --- Random Forest ---
print(">> Random Forest…")
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42),
        {'n_estimators':randint(50,200),'max_depth':randint(3,10),
         'min_samples_split':randint(2,20),'min_samples_leaf':randint(1,20)},
        n_iter=2, cv=3, scoring='roc_auc', random_state=42).fit(Xtr_r, ytr_r)
m_rf = rs.best_estimator_
metrics_rows.append(evaluar("Random Forest", yte, m_rf.predict(Xte), m_rf.predict_proba(Xte)[:,1]))

# --- XGBoost ---
print(">> XGBoost…")
rs = RandomizedSearchCV(xgb.XGBClassifier(random_state=42),
        {'n_estimators':randint(50,200),'max_depth':randint(3,10),
         'learning_rate':[0.01,0.1,0.2],'subsample':[0.8,0.9,1.0],
         'min_child_weight':randint(1,10)},
        n_iter=2, cv=3, scoring='roc_auc', random_state=42).fit(Xtr_r, ytr_r)
m_xgb = rs.best_estimator_
metrics_rows.append(evaluar("XGBoost", yte, m_xgb.predict(Xte), m_xgb.predict_proba(Xte)[:,1]))

# --- CatBoost ---
print(">> CatBoost…")
rs = RandomizedSearchCV(CatBoostClassifier(random_state=42, verbose=0),
        {'iterations':randint(50,200),'depth':randint(3,10),
         'learning_rate':[0.01,0.1,0.2],'subsample':[0.8,0.9,1.0],
         'min_child_samples':randint(1,10)},
        n_iter=2, cv=3, scoring='roc_auc', random_state=42).fit(Xtr_r, ytr_r)
m_cat = rs.best_estimator_
metrics_rows.append(evaluar("CatBoost", yte, m_cat.predict(Xte), m_cat.predict_proba(Xte)[:,1]))

comp = pd.DataFrame(metrics_rows).sort_values("AUC", ascending=False).reset_index(drop=True)
print(comp)

# ----------------------------------------------------------------------
# SELECCIÓN DE VARIABLES (importancia 95% + correlación + SHAP top25)
# ----------------------------------------------------------------------
print(">> Selección de variables…")
imp = pd.DataFrame({'Variable':X.columns, 'Importancia':m_xgb.feature_importances_}) \
        .sort_values('Importancia', ascending=False)
imp['Porcentaje Acumulado'] = imp['Importancia'].cumsum()*100
vars_95 = list(imp[imp['Porcentaje Acumulado'] <= 95.1]['Variable'])

def quitar_correladas(d, umbral=0.65, seed=42):
    mc = d.corr().abs()
    pares = [(mc.columns[i], mc.columns[j]) for i in range(len(mc.columns))
             for j in range(i) if mc.iloc[i,j] > umbral]
    elim = set(); np.random.seed(seed)
    for p in pares: elim.add(np.random.choice(p))
    return list(set(d.columns) - elim)

sin_corr = quitar_correladas(df_final[vars_predict], seed=42)
vars_finales = list(set(vars_95).intersection(sin_corr))
print("   vars tras importancia+correlación:", len(vars_finales))

# XGBoost sobre vars_finales (igual que el notebook)
Xf = df_final[vars_finales]
Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2, random_state=42, stratify=y)
Xtr_r, ytr_r = RandomOverSampler(random_state=42).fit_resample(Xtr, ytr)
rs = RandomizedSearchCV(xgb.XGBClassifier(random_state=42),
        {'n_estimators':randint(50,200),'max_depth':randint(3,10),
         'learning_rate':[0.01,0.1,0.2],'subsample':[0.8,0.9,1.0],
         'min_child_weight':randint(1,10)},
        n_iter=2, cv=3, scoring='roc_auc', random_state=42).fit(Xtr_r, ytr_r)
m_tmp = rs.best_estimator_

# SHAP top 25 (muestra para velocidad; no cambia el modelo, sólo la selección)
print(">> SHAP…")
import shap
sample = Xf.sample(n=min(4000, len(Xf)), random_state=42)
expl = shap.TreeExplainer(m_tmp)
sv = expl.shap_values(sample)
shap_imp = pd.DataFrame({'Variable':Xf.columns, 'Importancia':np.abs(sv).mean(axis=0)}) \
             .sort_values('Importancia', ascending=False)
variables_finales = shap_imp.head(25)['Variable'].tolist()
print("   variables_finales (25):", variables_finales)

# Modelo final sobre las 25 variables SHAP
print(">> Modelo final XGBoost…")
Xff = df_final[variables_finales]
Xtr, Xte, ytr, yte = train_test_split(Xff, y, test_size=0.2, random_state=42, stratify=y)
Xtr_r, ytr_r = RandomOverSampler(random_state=42).fit_resample(Xtr, ytr)
rs = RandomizedSearchCV(xgb.XGBClassifier(random_state=42),
        {'n_estimators':randint(50,200),'max_depth':randint(3,10),
         'learning_rate':[0.01,0.1,0.2],'subsample':[0.8,0.9,1.0],
         'min_child_weight':randint(1,10)},
        n_iter=2, cv=3, scoring='roc_auc', random_state=42).fit(Xtr_r, ytr_r)
modelo_final = rs.best_estimator_
yp = modelo_final.predict(Xte); ypr = modelo_final.predict_proba(Xte)[:,1]
final_metrics = {
    "AUC": round(roc_auc_score(yte, ypr),4),
    "Accuracy": round(accuracy_score(yte, yp),4),
    "Precision": round(precision_score(yte, yp),4),
    "Recall": round(recall_score(yte, yp),4),
    "F1": round(f1_score(yte, yp),4),
    "confusion": confusion_matrix(yte, yp).tolist(),
}
print("   métricas finales:", final_metrics)

# ----------------------------------------------------------------------
# METADATA DE VARIABLES PARA EL FORMULARIO (Panel B)
# ----------------------------------------------------------------------
# Para cada variable final, decidir si es numérica continua o binaria/dummy
meta = {}
for v in variables_finales:
    serie = df_final[v]
    vals = pd.to_numeric(serie, errors='coerce')
    uniq = sorted([float(u) for u in pd.unique(vals.dropna())])
    if v == 'gender' or v.split('_')[0] in ('ethnicity','icu','apache','gender') and set(uniq).issubset({0.0,1.0}):
        tipo = 'binary'
    elif set(uniq).issubset({0.0,1.0}):
        tipo = 'binary'
    else:
        tipo = 'numeric'
    meta[v] = {
        "tipo": tipo,
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
        "median": float(np.nanmedian(vals)),
        "mean": float(np.nanmean(vals)),
    }

# ----------------------------------------------------------------------
# DATASET PARA PANEL A  (versión legible y compacta)
# ----------------------------------------------------------------------
panelA_cols = ['hospital_death','age','bmi','gender','ethnicity','icu_type',
               'height','weight','pre_icu_los_days','heart_rate_apache','map_apache',
               'temp_apache','resprate_apache','sodium_apache','creatinine_apache',
               'bun_apache','bilirubin_apache','hematocrit_apache','fio2_apache',
               'gcs_eyes_apache','gcs_motor_apache','gcs_verbal_apache',
               'd1_heartrate_max','d1_heartrate_min','apache_3j_bodysystem']
panelA_cols = [c for c in panelA_cols if c in df_panelA.columns]
dfA = df_panelA[panelA_cols].copy()
for c in dfA.columns:
    if str(dfA[c].dtype) == 'Int64':
        dfA[c] = dfA[c].astype('float').astype('Int64')
dfA.to_parquet(f"{OUT}/panelA.parquet", index=False)

# ----------------------------------------------------------------------
# GUARDAR ARTEFACTOS
# ----------------------------------------------------------------------
with open(f"{OUT}/modelo_final.pkl","wb") as f: pickle.dump(modelo_final, f)
with open(f"{OUT}/metadata.json","w") as f:
    json.dump({
        "variables_finales": variables_finales,
        "meta": meta,
        "comparacion": comp.to_dict(orient="records"),
        "modelo_ganador": "XGBoost",
        "final_metrics": final_metrics,
        "tasa_mortalidad": float(y.mean()*100),
        "n_pacientes": int(df_final['patient_id'].nunique()),
        "n_registros": int(len(df_final)),
    }, f, ensure_ascii=False, indent=2)

print("\n>> LISTO. Artefactos en", OUT)
print("   panelA.parquet:", dfA.shape)
