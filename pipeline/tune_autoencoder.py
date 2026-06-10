"""
Hyperparameter Tuning Script for Autoencoder

Performs grid search over:
- Sequence length
- Hidden dimensions
- Learning rate
- Number of LSTM layers

Saves the best configuration to a JSON file.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from models.autoencoder import AnomalyDetector
from sklearn.model_selection import train_test_split
import json
import itertools
from pathlib import Path

def load_data():
    """Load the fraud dataset"""
    data_path = 'data/enriched_sample.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} transactions")
    return df

def evaluate_model(detector, test_df):
    """
    Evaluate the model on test data.
    Returns mean reconstruction error (lower is better for normal transactions).
    """
    scores = detector.score(test_df)
    
    # Get actual fraud labels if available
    if 'is_fraud' in test_df.columns:
        # Calculate separation between fraud and non-fraud
        fraud_scores = scores[test_df.loc[scores.index, 'is_fraud'] == 1]
        normal_scores = scores[test_df.loc[scores.index, 'is_fraud'] == 0]
        
        if len(fraud_scores) > 0 and len(normal_scores) > 0:
            # Good model: high scores for fraud, low for normal
            separation = fraud_scores.mean() - normal_scores.mean()
            return separation
        else:
            return scores.std()  # Fallback: higher variance = better
    else:
        # Fallback: return score variability
        return scores.std()

def grid_search(train_df, test_df):
    """
    Perform grid search over hyperparameters.
    """
    # Define hyperparameter grid
    param_grid = {
        'sequence_length': [5, 10, 15],
        'hidden_dim': [32, 64, 128],
        'num_layers': [1, 2, 3],
        'learning_rate': [0.0001, 0.001, 0.01]
    }
    
    print("🔍 Starting Grid Search...")
    print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    best_score = -float('inf')
    best_params = None
    results = []
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    
    for i, combination in enumerate(itertools.product(*values)):
        params = dict(zip(keys, combination))
        
        print(f"\n[{i+1}] Testing: {params}")
        
        try:
            # Create detector with current params
            detector = AnomalyDetector(
                sequence_length=params['sequence_length'],
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                learning_rate=params['learning_rate']
            )
            
            # Train
            detector.train(train_df, epochs=10, save_best=False)
            
            # Evaluate
            score = evaluate_model(detector, test_df)
            
            results.append({
                **params,
                'score': score
            })
            
            print(f"   Score: {score:.6f}")
            
            # Track best
            if score > best_score:
                best_score = score
                best_params = params
                print(f"   ⭐ New best!")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    return best_params, best_score, results

def main():
    """Main tuning pipeline"""
    print("="*60)
    print("AUTOENCODER HYPERPARAMETER TUNING")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Split train/test
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    print(f"📊 Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Run grid search
    best_params, best_score, results = grid_search(train_df, test_df)
    
    # Save results
    output_dir = Path('models/tuning_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best config
    config_path = output_dir / 'best_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_score': float(best_score)
        }, f, indent=2)
    
    # Save all results
    results_path = output_dir / 'all_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("TUNING COMPLETE!")
    print("="*60)
    print(f"✅ Best Parameters: {best_params}")
    print(f"✅ Best Score: {best_score:.6f}")
    print(f"✅ Config saved to: {config_path}")
    print(f"✅ All results saved to: {results_path}")

if __name__ == "__main__":
    main()
