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
    metrics = pd.DataFrame()
    if model_type == 'Gradient Boosting':
        preds_file = os.path.join(RESULTS_DIR, 'predictions_gbm.csv')
        metrics_file = os.path.join(RESULTS_DIR, 'metrics_gbm.csv')
    elif model_type == 'Survival (CoxPH)':
        preds_file = os.path.join(RESULTS_DIR, 'predictions_survival.csv')
        metrics_file = os.path.join(RESULTS_DIR, 'metrics_survival.csv')
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
        
    return preds, metrics

@st.cache_resource
def load_interpretation(model_type):
    if model_type == 'Gradient Boosting':
        path = os.path.join(RESULTS_DIR, 'importance_gbm.pkl')
    elif model_type == 'Survival (CoxPH)':
        path = os.path.join(RESULTS_DIR, 'coefficients_survival.csv')
    else:
        path = os.path.join(RESULTS_DIR, 'shap_values.pkl')
        
    if os.path.exists(path):
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            return pd.read_csv(path, index_col=0)
    return None

@st.cache_data
def load_survival_curve():
    path = os.path.join(RESULTS_DIR, 'current_survival_curve.csv')
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


def main():
    st.title("S&P 500 Crash Risk Model")
    
    # Sidebar
    st.sidebar.header("Model Configuration")
    model_type = st.sidebar.radio("Select Model", ["Gradient Boosting", "Survival (CoxPH)"])
    
    st.sidebar.markdown("""
    **Objective**: Estimate likelihood of a >20% market crash in the next 12 months.
    **Models**:
    - *Gradient Boosting*: Fast, handles missing data naturally.
    - *Survival Model*: Estimates "Time-to-Failure" and conditional survival probability.
    """)

    with st.expander("Explore Architecture & Methodology"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("### 🛠 Model Structure")
            st.markdown("""
            - **Algorithm Selection**:
                - **HistGradientBoosting**: A boosting ensemble that builds trees sequentially.
                - **Cox Proportional Hazards**: A semi-parametric survival model that relates the "Hazard Rate" (risk of imminent crash) to covariates.
            - **Target Definition**: 
                - *Classification*: Binary (Crash in next 12m).
                - *Survival*: Time-to-Event (Months until next crash onset).
            - **Validation Strategy**: **Expanding Window Backtest**. Annual steps from 1960. Evaluated via Somers' D (Classification) or C-Index (Survival).
            """)
        with col_m2:
            st.markdown("### 📊 Feature Engineering")
            st.markdown("""
            - **Valuation**: CAPE and TTM Earnings Yield.
            - **Macro & Labor**: YoY Unemployment and CPI Inflation.
            - **Rates & Credit**: Term Spread and Credit Spreads.
            - **Market Internals**: 12m Momentum and Price Volatility.
            - **Leakage Prevention**: All macro features are lagged by 1 month.
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
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"Current Risk ({model_type})")
        
        if model_type == 'Survival (CoxPH)':
             latest_hazard = latest_row['Hazard_Score']
             st.metric("Hazard Score", f"{latest_hazard:.2f}")
             st.info("The Hazard Score multiplier indicates the current risk level relative to the historical baseline. >1.0 implies higher than average risk.")
             
             # Show Survival Curve
             curve = load_survival_curve()
             if curve is not None:
                 st.write("Survival Function (Prob. of NO crash over time)")
                 fig_curve = px.line(curve, title="Survival Probability Distribution", labels={'index': 'Months', 'value': 'Prob(Survival)'})
                 st.plotly_chart(fig_curve, width='stretch')
        else:
            latest_prob = latest_row['Crash_Prob']
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
        if interp_data is not None:
            if isinstance(interp_data, dict):
                method = interp_data.get('method', 'unknown')
                if method == 'shap':
                    shap_vals = interp_data['shap_values']
                    last_shap = shap_vals[-1, :] if len(shap_vals.shape) == 2 else shap_vals
                    feature_names = interp_data['X'].columns
                    shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP': last_shap})
                    shap_df['AbsSHAP'] = shap_df['SHAP'].abs()
                    shap_df = shap_df.sort_values('AbsSHAP', ascending=True).tail(10)
                    fig_shap = px.bar(shap_df, x='SHAP', y='Feature', orientation='h', 
                                      title="Feature Contribution (SHAP)",
                                      color='SHAP', color_continuous_scale=['blue', 'red'])
                    st.plotly_chart(fig_shap, width='stretch')
                elif method == 'permutation':
                    fig_fi = px.bar(pd.DataFrame({'Feature': interp_data['features'], 'Importance': interp_data['importances_mean']}).sort_values('Importance'), 
                                   x='Importance', y='Feature', orientation='h', title="Permutation Importance")
                    st.plotly_chart(fig_fi, width='stretch')
            else:
                # Survival Coefficients (DataFrame)
                coef_df = interp_data.copy()
                coef_df['Exp(Coef)'] = np.exp(coef_df['coef'])
                coef_df = coef_df.sort_values('coef', ascending=True)
                fig_coef = px.bar(coef_df, x='coef', y=coef_df.index, orientation='h', 
                                  title="Log Hazard Coefficients",
                                  labels={'index': 'Feature', 'coef': 'Log Hazard Ratio'})
                st.plotly_chart(fig_coef, width='stretch')
                st.caption("Positive coefficients increase the crash hazard.")
        else:
            st.warning("Interpretation data not found.")

    # Timeline
    st.subheader(f"Historical Risk Timeline ({model_type})")
    y_col = 'Hazard_Score' if model_type == 'Survival (CoxPH)' else 'Crash_Prob'
    fig_time = px.area(predictions, y=y_col, title=f"{y_col} Over Time")
    
    target_col = 'Event' if model_type == 'Survival (CoxPH)' else 'Target'
    crashes = predictions[predictions[target_col] == 1]
    fig_time.add_trace(go.Scatter(x=crashes.index, y=crashes[y_col], mode='markers', name='Actual Crash Window', marker=dict(color='red', size=2)))
    st.plotly_chart(fig_time, width='stretch')
    
    # Backtest Performance
    perf_label = "C-Index" if model_type == 'Survival (CoxPH)' else "Somers' D"
    st.subheader(f"Model Performance (Rolling {perf_label})")
    
    with st.expander("🎓 How is this calculated?"):
        if model_type == 'Survival (CoxPH)':
            st.markdown("""
            **Rolling C-Index (Concordance Index)**:
            - **Window**: 5-Year (60-month) out-of-sample period.
            - **Calculation**: Measures the proportion of pairs where the model correctly predicted which event would happen first.
            - **Interpretation**: A score of 0.5 is random; 1.0 is a perfect ranking of "Time-to-Crash."
            """)
        else:
            st.markdown("""
            **Rolling Somers' D**:
            - **Window**: 5-Year (60-month) out-of-sample period.
            - **Calculation**: $D = 2 \times AUC - 1$. It scales the Area Under the Curve (AUC) from [-1, 1].
            - **Interpretation**: 
                - **> 0**: Model rank-orders risk better than random.
                - **0.60**: Represents a "strong" predictive signal (equivalent to 0.80 AUC).
            """)

    if not metrics.empty:
        m_col = 'C_Index' if model_type == 'Survival (CoxPH)' else 'Rolling_SomersD'
        fig_perf = px.line(metrics, y=m_col, title=f"Rolling {perf_label}")
        if model_type != 'Survival (CoxPH)':
            fig_perf.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_perf, width='stretch')
    else:
        st.write("Not enough data for rolling metrics.")

if __name__ == "__main__":
    main()
