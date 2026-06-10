import streamlit as st
import pandas as pd

def show_loading_skeleton(num_rows=3):
    """Display a skeleton loading screen"""
    st.markdown("""
        <div style="padding: 20px;">
            <div class="skeleton" style="height: 40px; margin-bottom: 20px;"></div>
            <div class="skeleton" style="height: 100px; margin-bottom: 15px;"></div>
            <div class="skeleton" style="height: 100px; margin-bottom: 15px;"></div>
            <div class="skeleton" style="height: 100px;"></div>
        </div>
    """, unsafe_allow_html=True)

def show_spinner_loading(message="Processing..."):
    """Display centered spinner with message"""
    st.markdown(f"""
        <div style="text-align: center; padding: 40px;">
            <div class="spinner"></div>
            <p style="color: var(--text-secondary); margin-top: 20px;">{message}</p>
        </div>
    """, unsafe_allow_html=True)

def show_empty_state(icon="📭", title="No Data Available", description="", action_text=""):
    """Display a beautiful empty state"""
    st.markdown(f"""
        <div class="empty-state">
            <div class="empty-state-icon">{icon}</div>
            <h3 style="color: var(--text-primary); margin-bottom: 0.5rem;">{title}</h3>
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">{description}</p>
            {f'<p style="color: var(--accent-primary); cursor: pointer;">{action_text}</p>' if action_text else ''}
        </div>
    """, unsafe_allow_html=True)

def render_risk_badge(risk_level):
    """Render a styled risk badge"""
    badge_map = {
        'High': '<span class="risk-badge-high">🔴 HIGH RISK</span>',
        'Medium': '<span class="risk-badge-medium">🟠 MEDIUM RISK</span>',
        'Low': '<span class="risk-badge-low">🟢 LOW RISK</span>'
    }
    return badge_map.get(risk_level, '')

def render_notification_badge(count):
    """Render notification badge"""
    if count > 0:
        return f'<span class="notification-badge">{count if count < 100 else "99+"}</span>'
    return ''

def clean_merchant_name(merchant_name):
    """Clean merchant name for human-readable display"""
    if not merchant_name or pd.isna(merchant_name):
        return "Unknown"
    
    # Convert to string
    name = str(merchant_name)
    
    # Remove 'fraud_' prefix (case-insensitive)
    import re
    name = re.sub(r'^fraud[_\s-]*', '', name, flags=re.IGNORECASE)
    
    # Remove underscores and replace with spaces
    name = name.replace('_', ' ')
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    # Title case for readability
    name = name.title()
    
    # Truncate if too long (for graph display)
    if len(name) > 30:
        name = name[:27] + '...'
    
    return name if name else "Unknown"

def clean_feature_name(feature_name):
    """Clean feature name for human-readable display in charts"""
    if not feature_name or pd.isna(feature_name):
        return "Unknown Feature"
    
    name = str(feature_name)
    
    # Feature name mappings for better readability
    replacements = {
        'amt': 'Transaction Amount',
        'card_7d_amt_sum': 'Card 7-Day Amount Sum',
        'card_30d_amt_sum': 'Card 30-Day Amount Sum',
        'merch_amt_sum': 'Merchant Amount Sum',
        'merch_amt_sum_': 'Merchant Total Amount',
        'city_pop': 'City Population',
        'mode_sig_card': 'Card Mode Signature',
        'safe_pagrank': 'Safe PageRank Score',
        'card_30d_tx_count': 'Card 30-Day Transaction Count',
        'card_7d_tx_count': 'Card 7-Day Transaction Count',
        'card_1d_tx_count': 'Card 1-Day Transaction Count',
        'merch_tx_count': 'Merchant Transaction Count',
        'lat': 'Latitude',
        'long': 'Longitude',
        'unix_time': 'Unix Timestamp',
        'category': 'Merchant Category',
        'proba': 'Fraud Probability',
        'is_fraud': 'Fraud Label'
    }
    
    # Check for exact match first
    if name in replacements:
        return replacements[name]
    
    # Otherwise, clean up the name
    # Remove underscores and split
    parts = name.replace('_', ' ').split()
    
    # Capitalize each word
    cleaned = ' '.join(word.capitalize() for word in parts)
    
    # Common abbreviations to uppercase
    cleaned = cleaned.replace('Tx ', 'TX ').replace('Amt ', 'Amount ')
    cleaned = cleaned.replace('7d', '7-Day').replace('30d', '30-Day').replace('1d', '1-Day')
    cleaned = cleaned.replace('Merch ', 'Merchant ')
    
    return cleaned
