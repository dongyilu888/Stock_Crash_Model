import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Config
st.set_page_config(page_title="S&P 500 Crash Predictor", layout="wide")

# Paths
RESULTS_DIR = 'results'
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, 'predictions.csv')
METRICS_FILE = os.path.join(RESULTS_DIR, 'metrics.csv')
SHAP_FILE = os.path.join(RESULTS_DIR, 'shap_values.pkl')

@st.cache_data
def load_data(model_type):
    if model_type == 'Gradient Boosting':
        preds_file = os.path.join(RESULTS_DIR, 'predictions_gbm.csv')
        metrics_file = os.path.join(RESULTS_DIR, 'metrics_gbm.csv')
    else:
        preds_file = os.path.join(RESULTS_DIR, 'predictions.csv')
        metrics_file = os.path.join(RESULTS_DIR, 'metrics.csv')
        
    if os.path.exists(preds_file):
        preds = pd.read_csv(preds_file, index_col=0, parse_dates=[0])
    else:
        st.warning(f" predictions file not found: {preds_file}")
        return pd.DataFrame(), pd.DataFrame()

    if os.path.exists(metrics_file):
        metrics = pd.read_csv(metrics_file, index_col=0, parse_dates=[0])
    else:
        metrics = pd.DataFrame()
        
    return preds, metrics

@st.cache_resource
def load_interpretation(model_type):
    if model_type == 'Gradient Boosting':
        shap_file = os.path.join(RESULTS_DIR, 'importance_gbm.pkl')
    else:
        shap_file = os.path.join(RESULTS_DIR, 'shap_values.pkl')
        
    if os.path.exists(shap_file):
        with open(shap_file, 'rb') as f:
            data = pickle.load(f)
        return data
    return None

def main():
    st.title("S&P 500 Crash Risk Model")
    
    # Sidebar
    st.sidebar.header("Model Configuration")
    model_type = st.sidebar.radio("Select Model", ["Random Forest", "Gradient Boosting"])
    
    st.sidebar.markdown("""
    **Objective**: Estimate likelihood of a >20% market crash in the next 12 months.
    **Models**:
    - *Random Forest*: Ensemble of 100 trees; robust to outliers and non-linearities.
    - *Gradient Boosting*: Histogram-based GBT; handles missing values natively and excels at finding complex patterns.
    """)

    with st.expander("Explore Architecture & Methodology"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("### 🛠 Model Structure")
            st.markdown("""
            - **Algorithm Selection**:
                - **Random Forest**: A bagging ensemble that reduces variance. Configured with 100 estimators and a depth of 5 to prevent overfitting.
                - **HistGradientBoosting**: A boosting ensemble that builds trees sequentially to reduce bias. Uses binning (histograms) for efficiency.
            - **Target Definition**: Binary classification where `1` represents an S&P 500 drawdown > 20% at any point in the following 12 months.
            - **Validation Strategy**: **Expanding Window Backtest**. We train on all data up to year $T$, then evaluate on year $T+1$, moving forward annually from 1960.
            """)
        with col_m2:
            st.markdown("### 📊 Feature Engineering")
            st.markdown("""
            - **Valuation**: CAPE (Cyclically Adjusted PE) and TTM Earnings Yield.
            - **Macro & Labor**: YoY Unemployment changes and CPI Inflation.
            - **Rates & Credit**: Term Spread (10Y-3M) and Credit Spreads (Baa-Aaa).
            - **Market Internals**: 12-month Momentum and Price Volatility.
            - **Commodities**: Broad PPI Commodity returns as a proxy for input costs.
            - **Leakage Prevention**: All macro/fundamental features are lagged by 1 month to respect reporting delays.
            """)
    
    
    predictions, metrics = load_data(model_type)
    
    if predictions.empty:
         st.error(f"No data available for {model_type}. Please run training script.")
         return

    interp_data = load_interpretation(model_type)

    latest_date = predictions.index.max()
    st.sidebar.info(f"Latest Prediction Date: {latest_date.date()}")
    
    # Today's Risk
    latest_row = predictions.iloc[-1]
    latest_prob = latest_row['Crash_Prob']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"Current Risk ({model_type})")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = latest_prob * 100,
            title = {'text': "Probability (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if latest_prob > 0.5 else "orange" if latest_prob > 0.3 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 50], 'color': "lightyellow"},
                    {'range': [50, 100], 'color': "salmon"}],
            }
        ))
        st.plotly_chart(fig_gauge, width='stretch')
        
        status = "CRITICAL" if latest_prob > 0.5 else "ELEVATED" if latest_prob > 0.3 else "LOW"
        st.metric("Risk Level", status)

    with col2:
        st.subheader("Top Risk Drivers")
        if interp_data:
            method = interp_data.get('method', 'unknown')
            
            if method == 'shap':
                # Plot SHAP values for the LAST observation
                shap_vals = interp_data['shap_values']
                # If shap_vals is (N, F), get last row
                if len(shap_vals.shape) == 2:
                    last_shap = shap_vals[-1, :]
                else:
                    last_shap = shap_vals # Should capture the shape
                    
                feature_names = interp_data['X'].columns
                
                # Create DataFrame
                shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP': last_shap})
                shap_df['AbsSHAP'] = shap_df['SHAP'].abs()
                shap_df = shap_df.sort_values('AbsSHAP', ascending=True).tail(10)
                
                fig_shap = px.bar(shap_df, x='SHAP', y='Feature', orientation='h', 
                                  title="Feature Contribution (SHAP)",
                                  color='SHAP', color_continuous_scale=['blue', 'red'])
                st.plotly_chart(fig_shap, width='stretch')
                
            elif method == 'importance' or method == 'permutation':
                # Feature Importance
                key = 'importances_mean' if 'importances_mean' in interp_data else 'feature_importances'
                importances = interp_data[key]
                feats = interp_data['features']
                fi_df = pd.DataFrame({'Feature': feats, 'Importance': importances}).sort_values('Importance', ascending=True)
                fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                                title=f"Feature Importance ({method.title()})")
                st.plotly_chart(fig_fi, width='stretch')
        else:
            st.warning("Interpretation data not found.")

    # Timeline
    st.subheader(f"Historical Crash Probability ({model_type})")
    fig_time = px.area(predictions, y='Crash_Prob', title="Crash Probability Over Time")
    crashes = predictions[predictions['Target'] == 1]
    fig_time.add_trace(go.Scatter(x=crashes.index, y=crashes['Crash_Prob'], mode='markers', name='Actual Crash Window', marker=dict(color='red', size=2)))
    st.plotly_chart(fig_time, width='stretch')
    
    # Backtest Performance
    st.subheader("Model Performance (Rolling 5-Year Somers' D)")
    if not metrics.empty:
        fig_perf = px.line(metrics, y='Rolling_SomersD', title="Rolling Somers' D Score")
        fig_perf.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_perf, width='stretch')
    else:
        st.write("Not enough data for rolling metrics.")

if __name__ == "__main__":
    main()
