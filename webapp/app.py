import streamlit as st
import pandas as pd
import requests
import io
import joblib
import numpy as np
import os
import time
import altair as alt
import streamlit.components.v1 as components

# Attempt import from webapp package; fallback for local run
try:
    from webapp import utils as ui
    from webapp import actions_db
except ImportError:
    import utils as ui
    import actions_db


def clear_custom_sidebar_toggle():
    """Remove stale custom sidebar state from earlier app versions."""
    components.html(
        """
        <script>
        const doc = window.parent.document;
        doc.getElementById('graphite-sidebar-toggle')?.remove();
        doc.getElementById('graphite-sidebar-toggle-style')?.remove();
        doc.getElementById('graphite-sidebar-opener')?.remove();
        doc.getElementById('graphite-sidebar-opener-style')?.remove();
        doc.body.classList.remove('graphite-sidebar-closed');
        window.parent.localStorage.removeItem('graphite-sidebar-closed');

        const sidebarIsOpen = () => {
            const sidebar = doc.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) {
                return false;
            }

            const rect = sidebar.getBoundingClientRect();
            const styles = window.parent.getComputedStyle(sidebar);
            return rect.width > 80 &&
                rect.right > 80 &&
                styles.display !== 'none' &&
                styles.visibility !== 'hidden' &&
                styles.opacity !== '0';
        };

        const restoreNativeSidebar = () => {
            if (sidebarIsOpen()) {
                return;
            }

            const controls = Array.from(doc.querySelectorAll('button, [role="button"]'));
            const opener = controls.find((control) => {
                const label = [
                    control.getAttribute('aria-label'),
                    control.getAttribute('title'),
                    control.innerText
                ].filter(Boolean).join(' ').toLowerCase();

                return label.includes('open') && label.includes('sidebar');
            }) || doc.querySelector('[data-testid="collapsedControl"]');

            if (opener) {
                opener.click();
            }
        };

        window.setTimeout(restoreNativeSidebar, 100);
        window.setTimeout(restoreNativeSidebar, 500);
        </script>
        """,
        height=0,
    )


# Page Config
st.set_page_config(
    page_title='Graph-Powered Fraud Intelligence System: AI-Powered Transaction Monitoring & Financial Crime Detection System',
    page_icon='🛡️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# --- ADD THIS BLOCK ---
st.markdown(
    """
    <style>
    /* We use !important to make sure your 20% override wins */
    .st-emotion-cache-1l0wbpp svg {
        width: 40% !important;
        max-width: 40% !important;
        height: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if 'dark_theme_enabled' not in st.session_state:
    st.session_state.dark_theme_enabled = True

# Load custom styling
selected_theme = 'dark' if st.session_state.dark_theme_enabled else 'light'
try:
    ui.load_css(theme=selected_theme)
except TypeError as exc:
    if "unexpected keyword argument 'theme'" not in str(exc):
        raise
    ui.load_css()
clear_custom_sidebar_toggle()

# --- Sidebar Configuration ---
with st.sidebar:
    st.image('webapp/logo.svg', width=60)
    st.title("Graphite AI")
    st.toggle('Dark Theme', key='dark_theme_enabled')
    
    # Main Navigation
    app_mode = st.radio("System Mode", ["Dashboard", "Reports Explorer"], index=0)
    
    if app_mode == "Dashboard":
        st.markdown("### Settings")
        mode = st.radio('Select Detection Mode', ['API Server', 'Local Model'], index=0, help="Choose between calling an external API or running the model locally in-memory.")
        
        st.markdown("### Parameters")
        threshold = st.slider('Risk Threshold', 0.0, 1.0, 0.5, 0.01, help="Transactions with a score above this value are flagged.")
        top_n = st.number_input('Top Alerts to Show', 1, 500, 50)
        
        st.markdown("### Explainability")
        explain = st.toggle('Enable SHAP Explanations', True)
        if explain:
            num_explain = st.number_input('Max rows to explain', 1, 20, 5)
        
        st.markdown("### Data Source")
        use_sample = st.checkbox('Use Sample Data', True)
        uploaded_file = None
        if not use_sample:
            uploaded_file = st.file_uploader("Upload Transaction CSV", type=['csv'])

# --- Main Content ---
ui.render_header()

if app_mode == "Reports Explorer":
    ui.render_report_explorer()
    st.stop()

# Data Loading
df = None
if use_sample:
    sample_path = os.path.join('data', 'enriched_sample2.csv')
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
    else:
        st.error(f"Sample file not found at `{sample_path}`.")
elif uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.toast('CSV uploaded successfully.', icon='✅')
    except Exception as e:
        st.error(f"Error reading file: {e}")

if df is None:
    st.info("👈 Please upload a CSV file or select the sample dataset to begin.")
    st.stop()

# --- Preview Section ---
with st.expander("Data Preview", expanded=False):
    st.dataframe(df.head(), width="stretch")

# --- Scoring Logic ---
def run_analysis(df_input):
    """Main analysis engine with real-time UI updates."""
    # Initialize Loading UI
    status_container = st.empty()
    progress_bar = st.progress(0)
    log_container = st.empty()
    
    logs = []
    def add_log(message, type="info"):
        ts = time.strftime("%H:%M:%S")
        icon = "●"
        style_class = f"scan-log-{type}"
        logs.insert(0, f'<div class="scan-log-entry"><span class="scan-log-ts">[{ts}]</span> <span class="{style_class}">{icon}</span> {message}</div>')
        log_html = f'<div class="scan-log-container">{"".join(logs)}</div>'
        log_container.markdown(log_html, unsafe_allow_html=True)

    # Stage 0: SQLite Sync
    status_container.markdown("### 🗄️ Synchronizing with Forensic Ledger (SQLite)...")
    add_log("Committing batch to GraphiteDB...", "info")
    progress_bar.progress(5)
    try:
        actions_db.ingest_transactions(df_input)
        add_log("Integrity check passed. DB Sync complete.", "success")
    except Exception as e:
        add_log(f"DB Sync Warning: {e}", "warning")
    
    # Reload from DB to be purely stateful
    df = actions_db.get_transactions()
    if df.empty: df = df_input.copy() # Fallback

    # Stage 1: Initializing
    status_container.markdown("### 🛰️ Initializing Graphite Engine...")
    add_log("System boot sequence initiated...", "info")
    progress_bar.progress(15)
    time.sleep(0.4)
    add_log("Loading enriched features from disk...", "info")
    
    start_time = time.time()
    results_df = None
    error_msg = None
    shap_data = None

    # Stage 2: Data Preprocessing
    status_container.markdown("### 🧪 Preprocessing Transaction Batch...")
    add_log(f"Processing {len(df)} transactions...", "info")
    progress_bar.progress(30)
    time.sleep(0.5)

    if mode == 'API Server':
        status_container.markdown("### 🔌 Connecting to Remote Analysis API...")
        add_log("Establishing secure handshake with API gateway...", "info")
        progress_bar.progress(45)
        
        api_url = 'http://localhost:8001/score_file'
        try:
            files = {'file': ('upload.csv', df.to_csv(index=False).encode())}
            params = {'explain': str(explain).lower(), 'top_k': str(top_n)}
            
            add_log("Streaming data to remote neural engine...", "info")
            resp = requests.post(api_url, files=files, params=params, timeout=60)
            
            if resp.status_code == 200:
                add_log("API cluster response received. Decoding packet...", "success")
                progress_bar.progress(70)
                data = resp.json()
                res_rows = data.get('results', [])
                preds_df = pd.DataFrame(res_rows).set_index('index')
                
                results_df = df.copy()
                results_df['proba'] = preds_df['proba']
                results_df['pred'] = preds_df['pred']
                
                if 'anomaly_score' in preds_df.columns:
                    results_df['anomaly_score'] = preds_df['anomaly_score']
                if 'hybrid_risk' in preds_df.columns:
                    results_df['hybrid_risk'] = preds_df['hybrid_risk']

                if 'shap' in data:
                    shap_data = data['shap']
            else:
                error_msg = f"API Error {resp.status_code}: {resp.text}"
                add_log(f"API rejection: {error_msg}", "warning")
        except Exception as e:
            error_msg = f"Connection Failed: {e}"
            add_log(f"Fatal link error: {error_msg}", "warning")

    else: # Local Mode
        status_container.markdown("### 🧬 Executing Local ML Ensemble Analysis...")
        add_log("Loading LightGBM & Graph models into VRAM...", "info")
        progress_bar.progress(45)
        
        try:
            # Imports inside generic try/catch
            from pipeline.run_pipeline import build_features, sanitize_columns
            from train_graph_struct_model import load_model as _load_model
            
            model, scaler, model_features = _load_model()
            
            add_log("Building node neighborhoods (ego-graphs)...", "info")
            # Check for enriched features
            if 'card_7d_tx_count' not in df.columns:
                X, df_enr = build_features(df)
                results_df = df.copy()
            else:
                results_df = df.copy()
                X = results_df.select_dtypes(include=[np.number]).drop(columns=['is_fraud'], errors='ignore').fillna(0)
            
            add_log("Normalizing feature space...", "info")
            progress_bar.progress(65)
            
            # Align columns
            X.columns = sanitize_columns(X.columns)
            if model_features:
                for f in model_features:
                    if f not in X.columns: X[f] = 0.0
                X = X[[f for f in model_features if f in X.columns]]
            
            # Scale
            if scaler:
                amt_cols = [c for c in X.columns if 'amt' in c]
                if amt_cols:
                        X[amt_cols] = scaler.transform(X[amt_cols])

            add_log("Scoring batch via Local Ensemble...", "success")
            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= threshold).astype(int)
            
            results_df['proba'] = probs
            results_df['pred'] = preds
            
            if explain:
                status_container.markdown("### 📊 Calculating SHAP Explanations...")
                add_log("Computing feature attributions (TreeExplainer)...", "info")
                try:
                    import shap
                    explainer = shap.TreeExplainer(model)
                    idxs = np.argsort(-probs)[:int(num_explain)]
                    vals = explainer.shap_values(X.iloc[idxs])
                    if isinstance(vals, (list, tuple)): 
                        sv = np.array(vals[1]) if len(vals) > 1 else np.array(vals[0])
                    else: 
                        sv = np.array(vals)
                    shap_data = {'idxs': idxs.tolist(), 'values': sv, 'columns': X.columns.tolist()}
                    add_log("SHAP values synchronized.", "success")
                except Exception as e:
                    st.warning(f"SHAP explanation failed: {e}")
                    add_log("SHAP subsystem failed - skipping.", "warning")

        except Exception as e:
            error_msg = f"Local Prediction Failed: {e}"
            add_log(f"Model failure: {error_msg}", "warning")

        # --- Deep Learning Layer (Anomaly Detection) ---
        status_container.markdown("### 🕸️ Running Deep Learning Anomaly Detection...")
        add_log("Initializing LSTM Autoencoder...", "info")
        progress_bar.progress(85)
        
        try:
            from models import AnomalyDetector
            dl_model = AnomalyDetector()
            
            model_path_sample = 'models/saved_models/autoencoder_sample.pth'
            model_path_full = 'models/saved_models/autoencoder_full.pth'
            
            if os.path.exists(model_path_sample):
                dl_model.load_model(model_path_sample)
                add_log("Found saved DL model (sample).", "info")
            elif os.path.exists(model_path_full):
                dl_model.load_model(model_path_full)
                add_log("Found saved DL model (full production).", "info")
            else:
                add_log("No saved DL model. Training demo batch on-the-fly...", "warning")
                dl_model.train(df, epochs=5, save_best=False)

            dl_scores = dl_model.score(df)
            add_log("Behavioral anomaly scores calculated.", "success")
            
            results_df['anomaly_score'] = dl_scores
            results_df['hybrid_risk'] = (results_df['proba'] * 0.6) + (results_df['anomaly_score'] * 0.4)
            add_log("Hybrid Intelligence scoring complete.", "success")
            
       	except Exception as e:
            st.warning(f"Deep Learning Layer inactive: {e}")
            add_log("DL Layer offline - using fallback scoring.", "warning")
            results_df['anomaly_score'] = 0.0
            results_df['hybrid_risk'] = results_df['proba']

    # Finalize
    progress_bar.progress(100)
    status_container.markdown("### ✅ Analysis Complete!")
    add_log("Finalizing security report and dashboard...", "success")
    time.sleep(0.5)
    
    # Hide loading elements
    status_container.empty()
    progress_bar.empty()
    log_container.empty()

    elapsed = time.time() - start_time
    
    if error_msg:
        st.error(error_msg)
    elif results_df is not None:
         # Success! Store in session state to persist after interactions
        st.session_state['last_results'] = results_df
        st.session_state['shap_data'] = shap_data
        st.session_state['last_run_time'] = elapsed
        st.session_state['threshold'] = threshold # store threshold used at runtime
        st.session_state['auto_scanned'] = True # Mark as scanned
        st.rerun()

# --- Scoring Logic ---
if st.button('🔍 Run System Scan', type='primary', width="stretch") or \
   ('auto_scanned' not in st.session_state and df is not None):
    run_analysis(df)

# --- Display Results ---
if 'last_results' in st.session_state:
    results_df = st.session_state['last_results']
    run_threshold = threshold # Use sidebar value directly for dynamic feedback
    
    # Sort by Hybrid Risk Score
    if 'hybrid_risk' in results_df.columns:
        results_df = results_df.sort_values('hybrid_risk', ascending=False)
        risk_metric = results_df['hybrid_risk']
    else:
        results_df = results_df.sort_values('proba', ascending=False)
        risk_metric = results_df['proba']


    # --- Review Table & Actions ---
    # Ensure orig_index exists before using it
    if 'orig_index' not in results_df.columns:
        results_df['orig_index'] = results_df.index
        

    
    


    # Show main data table with styling
    # --- Interactive Risk Dashboard ---
    # Replaces the static table with the new interactive one
    ui.render_interactive_risk_dashboard(results_df, top_n=top_n)

    # Export
    csv_data = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Full Report (CSV)", data=csv_data, file_name="fraud_predictions.csv", mime="text/csv")
    
# Bottom Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>Fraud Detection System v2.0 | Secured by Team Gyara</div>",
    unsafe_allow_html=True
)
