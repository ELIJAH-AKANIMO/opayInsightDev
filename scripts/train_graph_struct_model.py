import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier
import joblib
import json

print('Loading enriched full graph-struct data...')
df = pd.read_csv('data/enriched_full_graph_struct.csv')
print('Rows:', len(df))

# time split
print('Splitting by time...')
df_sorted = df.sort_values('unix_time').reset_index(drop=True)
idx = int(len(df_sorted) * 0.7)
split_time = df_sorted.loc[idx, 'unix_time']
train_time = df_sorted[df_sorted['unix_time'] <= split_time].copy()
val_time = df_sorted[df_sorted['unix_time'] > split_time].copy()

# drop identifiers
for c in ['first','last','street','city','state','zip','dob','trans_num','trans_date_trans_time']:
    if c in train_time.columns: train_time.drop(columns=[c], inplace=True)
    if c in val_time.columns: val_time.drop(columns=[c], inplace=True)

X_train = train_time.select_dtypes(include=[np.number]).drop(columns=['is_fraud']).fillna(0)
y_train = train_time['is_fraud'].astype(int)
X_val = val_time.select_dtypes(include=[np.number]).drop(columns=['is_fraud']).fillna(0)
y_val = val_time['is_fraud'].astype(int)

# sanitize columns
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

X_train.columns = sanitize_columns(X_train.columns)
X_val.columns = sanitize_columns(X_val.columns)

# scale amount-ish
scaler = StandardScaler()
amt_cols = [c for c in X_train.columns if 'amt' in c or c=='amt']
if amt_cols:
    X_train[amt_cols] = scaler.fit_transform(X_train[amt_cols])
    X_val[amt_cols] = scaler.transform(X_val[amt_cols])

# train
print('Training LGBM...')
clf = LGBMClassifier(n_estimators=500, learning_rate=0.05, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# eval
y_val_proba = clf.predict_proba(X_val)[:,1]
y_val_pred = (y_val_proba >= 0.5).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='binary', zero_division=0)
roc_auc = roc_auc_score(y_val, y_val_proba)
pr_auc = average_precision_score(y_val, y_val_proba)
metrics = {'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'roc_auc': float(roc_auc), 'pr_auc': float(pr_auc)}
print('Graph-struct model metrics:', metrics)

# save
Path('models').mkdir(exist_ok=True)
joblib.dump(clf, 'models/lgb_rolling_graph_struct_full.joblib')
joblib.dump(scaler, 'models/scaler_rolling_graph_struct_full.joblib')
with open('models/full_metrics_graph_struct.json','w') as f:
    json.dump(metrics, f)
print('Saved model and metrics')
print(metrics)
