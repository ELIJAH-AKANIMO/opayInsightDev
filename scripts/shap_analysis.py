"""Compute SHAP explanations for the most recent structural graph model.
Saves: models/shap_summary.json, models/shap_top_features.png, models/shap_values_sample.npy
"""
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split

print('Loading data and model...')
df = pd.read_csv('data/enriched_full_graph_struct.csv')
clf = joblib.load('models/lgb_rolling_graph_struct_full.joblib')

# time-split validation set
df_sorted = df.sort_values('unix_time').reset_index(drop=True)
idx = int(len(df_sorted) * 0.7)
val = df_sorted[df_sorted['unix_time'] > df_sorted.loc[idx,'unix_time']].copy()

# prepare X
X_val = val.select_dtypes(include=[np.number]).drop(columns=['is_fraud']).fillna(0)
y_val = val['is_fraud'].astype(int)

# sanitize column names as pipeline does
import re

def sanitize_columns(cols):
    new = []
    seen = {}
    for c in cols:
        nc = re.sub(r'[^0-9a-zA-Z_]', '_', str(c))
        if nc in seen:
            seen[nc] += 1
            nc = f"{nc}_{seen[nc]}"
        else:
            seen[nc] = 0
        new.append(nc)
    return new

X_val.columns = sanitize_columns(X_val.columns)

# sample for SHAP (to keep compute reasonable)
if len(X_val) > 50000:
    X_shap = X_val.sample(n=50000, random_state=42)
else:
    X_shap = X_val

print('Computing SHAP values (TreeExplainer)...')
# Align X_shap to the features used by the model
model_feat = None
try:
    model_feat = clf.booster_.feature_name()
except Exception:
    try:
        model_feat = clf.feature_name_
    except Exception:
        model_feat = None

if model_feat is None:
    print('Warning: Could not retrieve model feature names; proceeding with X_shap as-is')
else:
    # Ensure same columns and order: add missing with zeros, drop extras
    missing = [f for f in model_feat if f not in X_shap.columns]
    extra = [f for f in X_shap.columns if f not in model_feat]
    if missing:
        print('Adding missing features with zeros:', missing[:10])
        for m in missing:
            X_shap[m] = 0.0
    if extra:
        print('Dropping extra features:', extra[:10])
        X_shap = X_shap[model_feat]
    else:
        X_shap = X_shap[model_feat]

explainer = shap.TreeExplainer(clf)
sv = explainer.shap_values(X_shap)
# Handle LightGBM binary output format (list of arrays) and scalar possibilities
import numpy as np
if isinstance(sv, (list, tuple)):
    if len(sv) > 1:
        shap_arr = np.array(sv[1])
    else:
        shap_arr = np.array(sv[0])
else:
    shap_arr = np.array(sv)

# get mean absolute shap per feature
mean_abs = np.abs(shap_arr).mean(axis=0)
feat_importance = sorted(zip(X_shap.columns.tolist(), mean_abs.tolist()), key=lambda x: x[1], reverse=True)

# save top features JSON
Path('models').mkdir(exist_ok=True)
with open('models/shap_summary.json','w') as f:
    json.dump({'top_features': feat_importance[:50]}, f)

# Save shap values sample and create a summary plot
np.save('models/shap_values_sample.npy', shap_arr)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,10))
shap.summary_plot(shap_arr, X_shap, show=False)
plt.tight_layout()
plt.savefig('models/shap_top_features.png', dpi=150)
print('Saved models/shap_summary.json and models/shap_top_features.png')

print('Top 10 features:')
for f, v in feat_importance[:10]:
    print(f, v)
