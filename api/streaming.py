"""
Real-time Fraud Detection Streaming API

FastAPI endpoint for live transaction processing.
Supports single transactions and micro-batches.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import sys
import os
import io

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoder import AnomalyDetector
import joblib

app = FastAPI(title="Fraud Detection Streaming API", version="1.0.0")

# Global model instances
anomaly_detector = None
lightgbm_model = None
# Global model instances
anomaly_detector = None
lightgbm_model = None
scaler = None
explainer = None

def sanitize_columns(cols):
    import re
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

class Transaction(BaseModel):
    """Single transaction schema"""
    cc_num: str
    merchant: str
    category: str
    amt: float
    lat: float
    long: float
    city_pop: int
    unix_time: int
    trans_date_trans_time: Optional[str] = None
    is_fraud: Optional[int] = 0  # For evaluation purposes

class BatchTransactions(BaseModel):
    """Batch of transactions"""
    transactions: List[Transaction]

class FraudScore(BaseModel):
    """Fraud detection result"""
    transaction_id: str
    ml_probability: float
    anomaly_score: float
    hybrid_risk: float
    risk_level: str  # LOW, MEDIUM, HIGH

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global anomaly_detector, lightgbm_model, scaler, explainer
    
    print("Loading models...")
    import os
    print(f"DEBUG: CWD = {os.getcwd()}")
    if os.path.exists('models'):
        print(f"DEBUG: models/ contents = {os.listdir('models')}")
    else:
        print("DEBUG: models/ directory NOT FOUND at CWD")
    
    try:
        # Load Autoencoder
        anomaly_detector = AnomalyDetector()
        
        # Check for available models
        model_files = [
            'models/saved_models/autoencoder_sample.pth',
            'models/saved_models/autoencoder_full.pth',
            'models/saved_models/autoencoder.pth'
        ]
        
        loaded = False
        for model_path in model_files:
            if os.path.exists(model_path):
                anomaly_detector.load_model(model_path)
                print(f"Loaded Autoencoder from {model_path}")
                loaded = True
                break
        
        if not loaded:
            print("No saved Autoencoder found, will train on first request")
        
        # Load LightGBM
        # Try sample then full
        lgbm_files = ['models/lgb_rolling_graph_sample.joblib', 'models/lgb_rolling_graph_full.joblib', 'models/lightgbm_model.pkl']
        for lgbm_path in lgbm_files:
            if os.path.exists(lgbm_path):
                lightgbm_model = joblib.load(lgbm_path)
                print(f"Loaded LightGBM from {lgbm_path}")
                break
        
        if lightgbm_model is None:
            print("No LightGBM model found")
        else:
            # Initialize SHAP explainer
            try:
                import shap
                # TreeExplainer is fast for trees
                explainer = shap.TreeExplainer(lightgbm_model)
                print("Initialized SHAP Explainer")
            except Exception as e:
                print(f"Failed to init SHAP: {e}")

        # Load Scaler
        scaler_files = ['models/scaler_rolling_sample.joblib', 'models/scaler_rolling_full.joblib']
        for sc_path in scaler_files:
            if os.path.exists(sc_path):
                scaler = joblib.load(sc_path)
                print(f"Loaded Scaler from {sc_path}")
                break
            
    except Exception as e:
        print(f"Error loading models: {e}")

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "operational",
        "anomaly_detector_loaded": anomaly_detector is not None,
        "lightgbm_loaded": lightgbm_model is not None
    }

@app.post("/score", response_model=FraudScore)
async def score_transaction(txn: Transaction):
    """
    Score a single transaction in real-time.
    """
    if anomaly_detector is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert to DataFrame
        txn_dict = txn.dict()
        df = pd.DataFrame([txn_dict])
        
        # Get ML probability (if model available)
        ml_prob = 0.5  # Default
        if lightgbm_model is not None:
            try:
                # Simplified: assume features are prepared
                ml_prob = 0.5  # Placeholder - would need feature engineering
            except:
                pass
        
        # Get anomaly score
        anomaly_scores = anomaly_detector.score(df)
        anomaly_score = anomaly_scores.iloc[0] if len(anomaly_scores) > 0 else 0.0
        
        # Calculate hybrid risk
        hybrid_risk = (ml_prob * 0.6) + (anomaly_score * 0.4)
        
        # Determine risk level
        if hybrid_risk >= 0.7:
            risk_level = "HIGH"
        elif hybrid_risk >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return FraudScore(
            transaction_id=txn.cc_num,
            ml_probability=ml_prob,
            anomaly_score=float(anomaly_score),
            hybrid_risk=hybrid_risk,
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

@app.post("/score_batch", response_model=List[FraudScore])
async def score_batch(batch: BatchTransactions):
    """
    Score a batch of transactions.
    More efficient than individual calls.
    """
    if anomaly_detector is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert to DataFrame
        txns_data = [txn.dict() for txn in batch.transactions]
        df = pd.DataFrame(txns_data)
        
        # Get anomaly scores for batch
        anomaly_scores = anomaly_detector.score(df)
        
        results = []
        for idx, txn in enumerate(batch.transactions):
            ml_prob = 0.5  # Placeholder
            anomaly_score = anomaly_scores.iloc[idx] if idx < len(anomaly_scores) else 0.0
            hybrid_risk = (ml_prob * 0.6) + (float(anomaly_score) * 0.4)
            
            risk_level = "HIGH" if hybrid_risk >= 0.7 else "MEDIUM" if hybrid_risk >= 0.4 else "LOW"
            
            results.append(FraudScore(
                transaction_id=txn.cc_num,
                ml_probability=ml_prob,
                anomaly_score=float(anomaly_score),
                hybrid_risk=hybrid_risk,
                risk_level=risk_level
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scoring error: {str(e)}")



# Redefining properly with imports
from fastapi import UploadFile, File

@app.post("/score_file")
async def score_file_endpoint(file: UploadFile = File(...), explain: bool = False, top_k: int = 50):
    if anomaly_detector is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Get anomaly scores
        anomaly_scores = anomaly_detector.score(df)
        
        # Align anomaly scores to dataframe index (fill missing with 0 for first few steps)
        # Align anomaly scores to dataframe index (fill missing with 0 for first few steps)
        full_anomaly_scores = pd.Series(0.0, index=df.index)
        full_anomaly_scores.update(anomaly_scores)
        
        # Prepare for ML and Hybrid Score
        ml_probs = np.full(len(df), 0.5) # Default 0.5
        
        if lightgbm_model and 'card_7d_tx_count' in df.columns:
            try:
                # Prepare features for LightGBM - MUST MATCH PIPELINE EXACTLY
                # run_pipeline.py drops these: ['first','last','street','city','state','zip','dob','trans_num','trans_date_trans_time']
                drop_cols = ['first','last','street','city','state','zip','dob','trans_num','trans_date_trans_time']
                
                # Check what we have
                X_base = df.drop(columns=drop_cols, errors='ignore')
                
                # 1. Select numeric
                X = X_base.select_dtypes(include=[np.number]).drop(columns=['is_fraud'], errors='ignore').fillna(0)
                
                # 2. Sanitize
                X.columns = sanitize_columns(X.columns)
                
                # 3. Scale if needed
                if scaler:
                    # Scaler expects specific columns. 
                    # We rely on name matching if scaler supports it, or valid pipeline inputs.
                    # Since we dropped 'zip' etc, we should match X_train columns better now.
                    # Assuming X columns are a subset or match.
                    # Try to transform 'amt' columns
                    amt_cols = [c for c in X.columns if 'amt' in c or c=='amt']
                    if amt_cols:
                        try:
                             X[amt_cols] = scaler.transform(X[amt_cols])
                        except:
                             pass
                
                ml_probs = lightgbm_model.predict_proba(X)[:, 1]
                
                # Calculate SHAP if requested (only if ML success)
                shap_output = None
                print(f"DEBUG: explain={explain}, explainer={explainer is not None}")
                
                if explain and explainer:
                    try:
                        print("DEBUG: Starting SHAP calculation...")
                        # SHAP on all rows can be slow. 
                        # We only return SHAP for user, typically top_k is what matters in UI?
                        # But UI asks for 'shap' in response. 
                        # We'll compute for top_k riskiest rows to save time.
                        
                        # Calculate provisional hybrid risk to find top_k
                        # Vectorized calculation
                        anom_aligned = full_anomaly_scores.loc[df.index].values
                        risks = (ml_probs * 0.6) + (anom_aligned * 0.4)
                        
                        # Get indices of top_k risks
                        # argsort is ascending, take last k
                        k = min(top_k, len(df))
                        top_indices_pos = np.argsort(risks)[-k:] 
                        # Use iloc for X subset
                        X_subset = X.iloc[top_indices_pos]
                        
                        print(f"DEBUG: Computing SHAP for {len(X_subset)} rows")
                        shap_vals = explainer.shap_values(X_subset)
                        # LightGBM binary: shap_vals is list [array_class0, array_class1] or just array?
                        if isinstance(shap_vals, list):
                            sv = shap_vals[1]
                        else:
                            sv = shap_vals
                            
                        # Map back to original indices
                        original_indices = df.index[top_indices_pos].tolist()
                        
                        shap_output = {
                            'idxs': original_indices,
                            'values': sv.tolist(),
                            'columns': X.columns.tolist()
                        }
                        print("DEBUG: SHAP calculation successful")
                    except Exception as ex:
                        print(f"SHAP calc failed: {ex}")
                        import traceback
                        traceback.print_exc()

            except Exception as e:
                print(f"ML Prediction failed: {e}")
                import traceback
                traceback.print_exc()

        else:
             shap_output = None

        # Build results
        results = []
        # Vectorized assembly is faster but loop is fine for <1 sec latency on 10k rows
        # Actually loop is slow for 200k rows. 200k rows takes seconds in Python loop.
        # Vectorize result creation if possible? 
        # For now keep loop but maybe limit size? No, full report needed.
        # We optimize by using list comprehension zip
        
        anom_vals = full_anomaly_scores.loc[df.index].values
        hybrid_risks = (ml_probs * 0.6) + (anom_vals * 0.4)
        preds = (hybrid_risks > 0.5).astype(int)
        
        # Create records
        # Use DataFrame for speed
        res_df = pd.DataFrame({
            'index': df.index,
            'proba': ml_probs,
            'pred': preds,
            'anomaly_score': anom_vals,
            'hybrid_risk': hybrid_risks
        })
        
        results = res_df.to_dict(orient='records')
            
        resp = {"results": results}
        if shap_output:
            resp['shap'] = shap_output
            
        return resp
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
