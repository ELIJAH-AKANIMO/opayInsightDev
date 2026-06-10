from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib
import io
import json
import numpy as np
import traceback

# Ensure multipart support is available for form uploads
try:
    import multipart  # provided by package `python-multipart`
except Exception:
    raise RuntimeError("Form uploads require the 'python-multipart' package. Install it with: pip install python-multipart")

from pipeline.run_pipeline import build_features, sanitize_columns

app = FastAPI(title="Fraud Detection API")

# load model & scaler at startup
MODEL_PATH = 'models/lgb_rolling_graph_struct_full.joblib'
SCALER_PATH = 'models/scaler_rolling_graph_struct_full.joblib'
MODEL = None
SCALER = None
MODEL_FEATURES = None
# in-memory graph structural dicts (pr, evc, clust, core, neighbor_fraud) loaded at startup
GRAPH_DICTS = {}
# curated minimal features for single-transaction fast path (candidates)
MINIMAL_FEATURE_CANDIDATES = [
    'amt',
    'node_pr_card', 'node_pr_merch',
    'node_evc_card', 'node_evc_merch',
    'node_clust_card', 'node_clust_merch',
    'node_core_card', 'node_core_merch',
    'node_neighbor_fraud_card', 'node_neighbor_fraud_merch'
]
# resolved minimal set (intersection with model features when available)
MIN_FEATURE_SET = None
# whether trimming to minimal set is enabled; can be controlled via env var
import os as _os
MIN_FEATURE_TRIM_ENABLED = True if _os.environ.get('MIN_FEATURE_TRIM_ENABLED', '1') not in ('0', 'false', 'False') else False


@app.on_event('startup')
def load_models():
    global MODEL, SCALER, MODEL_FEATURES
    try:
        MODEL = joblib.load(MODEL_PATH)
    except Exception as e:
        MODEL = None
    try:
        SCALER = joblib.load(SCALER_PATH)
    except Exception:
        SCALER = None
    # try to fetch feature names
    try:
        MODEL_FEATURES = MODEL.booster_.feature_name()
    except Exception:
        try:
            MODEL_FEATURES = MODEL.feature_name_
        except Exception:
            MODEL_FEATURES = None

    # resolve minimal feature set for single-transaction fast path
    try:
        global MIN_FEATURE_SET
        if MODEL_FEATURES is not None:
            MIN_FEATURE_SET = [f for f in MINIMAL_FEATURE_CANDIDATES if f in MODEL_FEATURES]
        else:
            MIN_FEATURE_SET = MINIMAL_FEATURE_CANDIDATES.copy()
    except Exception:
        MIN_FEATURE_SET = MINIMAL_FEATURE_CANDIDATES.copy()

    # attempt to preload graph structural dicts for fast feature lookups
    try:
        import os as _os
        gd_path = 'models/graph_struct_dicts.pkl'
        if _os.path.exists(gd_path):
            GRAPH_DICTS.update(joblib.load(gd_path))
    except Exception:
        pass


# Actions API endpoints (small audit API for actions log)
@app.get('/actions')
async def list_actions():
    from webapp import actions_db
    actions = actions_db.get_actions()
    return JSONResponse(content={'n': len(actions), 'actions': actions})


@app.get('/actions/csv')
async def actions_csv():
    from webapp import actions_db
    data = actions_db.export_csv_bytes()
    return JSONResponse(content={'csv_base64_len': len(data)})


@app.post('/actions')
async def post_action(payload: dict):
    """Accepts JSON payload: {orig_index, action, reviewer, reason, ticket_id} and logs it to the local DB."""
    try:
        orig_index = payload.get('orig_index')
        action = payload.get('action')
        if orig_index is None or action is None:
            raise HTTPException(status_code=400, detail='orig_index and action are required')
        reviewer = payload.get('reviewer')
        reason = payload.get('reason')
        ticket_id = payload.get('ticket_id')
        from webapp import actions_db
        actions_db.log_action(orig_index, action, reviewer=reviewer, reason=reason, ticket_id=ticket_id)
        return JSONResponse(content={'status':'ok'})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ScoreResult(BaseModel):
    index: int
    proba: float
    pred: int


def align_and_prepare(df: pd.DataFrame):
    # Ensure features are numeric and enriched
    df_enr = build_features(df) if 'card_7d_tx_count' not in df.columns else df.copy()

    # select numeric features
    X = df_enr.select_dtypes(include=[np.number]).drop(columns=['is_fraud'], errors='ignore').fillna(0)

    if MODEL_FEATURES is not None:
        # sanitize columns similarly to training
        X.columns = sanitize_columns(X.columns)
        # add missing features
        for f in MODEL_FEATURES:
            if f not in X.columns:
                X[f] = 0.0
        # drop extras and reorder
        X = X[[f for f in MODEL_FEATURES if f in X.columns]]
    else:
        X = X

    # simple scale if scaler has amt-related columns
    if SCALER is not None:
        amt_cols = [c for c in X.columns if 'amt' in c or c == 'amt']
        if amt_cols:
            try:
                X[amt_cols] = SCALER.transform(X[amt_cols])
            except Exception:
                pass

    return X, df_enr


def _map_graph_features_single_row(row: dict, graph_dicts: dict):
    """Helper to map node-level graph dicts to features for a single transaction row."""
    out = {}
    # card node keys
    card_key = row.get('cc_num') or row.get('card_id')
    merch_key = row.get('merchant')
    if graph_dicts:
        for k, mapping in graph_dicts.items():
            # mapping keys are like 'pr', 'evc', 'clust', 'core', 'neighbor_fraud'
            # we map for card and merchant where possible
            try:
                if card_key is not None:
                    out[f'node_{k}_card'] = mapping.get(card_key, 0)
                if merch_key is not None:
                    out[f'node_{k}_merch'] = mapping.get(merch_key, 0)
            except Exception:
                out[f'node_{k}_card'] = out.get(f'node_{k}_card', 0)
                out[f'node_{k}_merch'] = out.get(f'node_{k}_merch', 0)
    return out


def align_and_prepare_single(df: pd.DataFrame, trim: bool | None = None):
    """Lightweight alignment for single transaction scoring.
    Adds quick node-level features using preloaded GRAPH_DICTS and applies sanitization & scaling.

    trim: Optional boolean. If None, use global MIN_FEATURE_TRIM_ENABLED. If True, restrict to MIN_FEATURE_SET.
    """
    df_enr = df.copy()

    # quick graph mappings for each row
    if GRAPH_DICTS:
        mapped = df_enr.apply(lambda r: _map_graph_features_single_row(r.to_dict(), GRAPH_DICTS), axis=1)
        # mapped is a series of dicts; expand into dataframe
        mapped_df = pd.DataFrame(list(mapped)) if len(mapped) else pd.DataFrame()
        if not mapped_df.empty:
            df_enr = pd.concat([df_enr.reset_index(drop=True), mapped_df.reset_index(drop=True)], axis=1)

    # select numeric features (only what's present in the tx)
    X = df_enr.select_dtypes(include=[np.number]).drop(columns=['is_fraud'], errors='ignore').fillna(0)

    if MODEL_FEATURES is not None:
        # sanitize and add missing features as zeros
        X.columns = sanitize_columns(X.columns)
        for f in MODEL_FEATURES:
            if f not in X.columns:
                X[f] = 0.0
        X = X[[f for f in MODEL_FEATURES if f in X.columns]]

    # decide whether to trim to minimal set
    effective_trim = MIN_FEATURE_TRIM_ENABLED if trim is None else bool(trim)

    if effective_trim and MIN_FEATURE_SET is not None and len(MIN_FEATURE_SET) > 0:
        # ensure columns sanitized to match model naming
        X.columns = sanitize_columns(X.columns)
        for f in MIN_FEATURE_SET:
            if f not in X.columns:
                X[f] = 0.0
        # preserve order and only keep minimal features
        X = X[[f for f in MIN_FEATURE_SET if f in X.columns]]

    # scale amt-related columns if scaler is available
    if SCALER is not None:
        amt_cols = [c for c in X.columns if 'amt' in c or c == 'amt']
        if amt_cols:
            try:
                X[amt_cols] = SCALER.transform(X[amt_cols])
            except Exception:
                pass

    return X, df_enr


@app.post('/score')
async def score_csv(file: Optional[UploadFile] = File(None), explain: bool = Query(False), top_k: int = Query(10)):
    """Score uploaded CSV transactions (or JSON array) and return probabilities. Set explain=true to get SHAP for top_k rows."""
    try:
        if file is None:
            raise HTTPException(status_code=400, detail='No file uploaded; send CSV file.')

        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        X, df_enr = align_and_prepare(df)
        if MODEL is None:
            raise HTTPException(status_code=500, detail='Model not loaded on server')

        probs = MODEL.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

        results = []
        for i, p in enumerate(probs.tolist()):
            results.append({'index': int(i), 'proba': float(p), 'pred': int(preds[i])})

        # prepare response
        resp = {'n': len(probs), 'results': results}

        if explain:
            # compute SHAP for top_k highest prob rows
            try:
                import shap
                idxs = np.argsort(-probs)[:top_k]
                X_explain = X.iloc[idxs]
                explainer = shap.TreeExplainer(MODEL)
                sv = explainer.shap_values(X_explain)
                # standardize output format
                if isinstance(sv, (list, tuple)):
                    shap_arr = np.array(sv[1]) if len(sv) > 1 else np.array(sv[0])
                else:
                    shap_arr = np.array(sv)
                shap_list = shap_arr.tolist()
                resp['shap'] = {'idxs': idxs.tolist(), 'values': shap_list, 'columns': X_explain.columns.tolist()}
            except Exception as e:
                resp['shap_error'] = str(e)

        return JSONResponse(content=resp)

    except Exception as exc:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=str(exc) + "\n" + tb)


@app.post('/score/tx')
async def score_single_tx(payload: dict, explain: bool = Query(False), trim: bool | None = Query(None)):
    """Score a single transaction JSON object and return a single result. Set explain=true to get SHAP values for the transaction.

    Optional query param `trim` overrides the configured minimal-feature trimming behavior (trim=true|false).
    """
    try:
        if not isinstance(payload, dict) or not payload:
            raise HTTPException(status_code=400, detail='Payload must be a non-empty JSON object representing one transaction')

        df = pd.DataFrame([payload])
        # use lightweight aligner for single tx to keep latency low; trim can override config
        X, df_enr = align_and_prepare_single(df, trim=trim)
        if MODEL is None:
            raise HTTPException(status_code=500, detail='Model not loaded on server')

        if len(X) == 0:
            raise HTTPException(status_code=400, detail='No numeric features available after alignment')

        probs = MODEL.predict_proba(X)[:, 1]
        p = float(probs[0])
        pred = int(p >= 0.5)
        result = {'index': 0, 'proba': p, 'pred': pred}

        resp = {'n': 1, 'result': result}

        if explain:
            try:
                import shap
                explainer = shap.TreeExplainer(MODEL)
                sv = explainer.shap_values(X.iloc[[0]])
                if isinstance(sv, (list, tuple)):
                    shap_arr = np.array(sv[1]) if len(sv) > 1 else np.array(sv[0])
                else:
                    shap_arr = np.array(sv)
                resp['shap'] = {'values': shap_arr.tolist(), 'columns': X.columns.tolist()}
            except Exception as e:
                resp['shap_error'] = str(e)

        return JSONResponse(content=resp)

    except Exception as exc:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=str(exc) + "\n" + tb)


@app.get('/')
async def root():
    """Root endpoint: returns basic info and links to API docs."""
    return JSONResponse(content={
        'status': 'ok',
        'message': 'Fraud Detection API running',
        'docs': '/docs',
        'redoc': '/redoc',
        'score_endpoint': '/score'
    })


@app.get('/health')
async def health():
    """Health check endpoint."""
    return JSONResponse(content={'status': 'ok', 'model_loaded': MODEL is not None})


@app.get('/config')
async def get_config():
    """Return debug configuration values for inspection."""
    return JSONResponse(content={
        'min_feature_trim_enabled': MIN_FEATURE_TRIM_ENABLED,
        'min_feature_set': MIN_FEATURE_SET,
        'graph_dicts_keys': list(GRAPH_DICTS.keys())
    })


@app.post('/sar')
async def generate_sar(payload: dict):
    """Generate a SAR CSV from provided transactions and optional metadata.
    Payload format: {"transactions": [ {..tx..}, ... ], "actions": [ {orig_index, action, reviewer, reason, ticket_id}, ... ] }
    Returns: CSV string in response (utf-8).
    """
    try:
        txs = payload.get('transactions') if isinstance(payload, dict) else None
        if not txs or not isinstance(txs, list):
            raise HTTPException(status_code=400, detail='Provide a non-empty "transactions" list in the payload')

        import csv
        import io as _io
        # optional actions map
        actions = {str(a['orig_index']): a for a in payload.get('actions', []) if 'orig_index' in a}

        out = _io.StringIO()
        writer = None
        for i, t in enumerate(txs):
            # flatten transaction and attach any action metadata
            t_copy = dict(t)
            idx = str(t_copy.get('orig_index', i))
            act = actions.get(idx, {})
            t_copy['action'] = act.get('action')
            t_copy['reviewer'] = act.get('reviewer')
            t_copy['reason'] = act.get('reason')
            t_copy['ticket_id'] = act.get('ticket_id')
            if writer is None:
                writer = csv.DictWriter(out, fieldnames=list(t_copy.keys()))
                writer.writeheader()
            writer.writerow(t_copy)
        csv_str = out.getvalue()
        return JSONResponse(content={'csv_len': len(csv_str), 'csv_str': csv_str})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))