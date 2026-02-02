import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import brier_score_loss, average_precision_score, precision_recall_curve, auc

# Config
st.set_page_config(page_title="S&P 500 Crash Predictor", layout="wide")

# Paths
RESULTS_DIR = 'results'
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, 'predictions.csv')
METRICS_FILE = os.path.join(RESULTS_DIR, 'metrics.csv')
SHAP_FILE = os.path.join(RESULTS_DIR, 'shap_values.pkl')
DATA_DIR = 'data'
FEATURES_FILE = os.path.join(DATA_DIR, 'features.csv')

FEATURES = [
    'CAPE', 'Earnings_Yield', 
    'Inflation_12m', 'Unemployment_Rate', 'Unemployment_Change_12m', 
    'Term_Spread', 'Credit_Spread', 'Real_Rate', 
    'Commodity_Ret_12m', 'S&P_Ret_12m', 'S&P_Vol_12m',
    'Housing_Starts_12m', 'Building_Permits_12m', 'Sentiment_Change_12m', 'Mfg_Hours_Change_12m',
    'CAPE_Unemp_Interaction', 'Term_Inversion'
]

@st.cache_data
def load_data(model_type):
    metrics = pd.DataFrame()
    if model_type == 'Random Forest':
        preds_file = os.path.join(RESULTS_DIR, 'predictions.csv')
        metrics_file = os.path.join(RESULTS_DIR, 'metrics.csv')
    elif model_type == 'Gradient Boosting':
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
        
    # Merge Price Data for Visualization
    if os.path.exists(FEATURES_FILE):
        features = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=[0])
        if 'S&P500_Price' in features.columns:
            # Shift features to match point-in-time if necessary? No, prices are contemporaneous for visualization
            cols_to_join = ['S&P500_Price']
            for extra in ['Current_Drawdown', 'Contraction_Phase']:
                if extra in features.columns:
                    cols_to_join.append(extra)
            preds = preds.join(features[cols_to_join], how='left')
            
    return preds, metrics

@st.cache_resource
def load_interpretation(model_type):
    if model_type == 'Random Forest':
        path = os.path.join(RESULTS_DIR, 'shap_values.pkl')
    elif model_type == 'Gradient Boosting':
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
    model_type = st.sidebar.radio("Select Model", ["Random Forest", "Gradient Boosting", "Survival (CoxPH)"])
    
    if model_type == 'Random Forest':
        st.sidebar.markdown("""
        **Objective**: Binary Classification (Crash in next 12m).
        **Model**: Random Forest
        - *Perf*: Somers' D: 0.546
        - *Strength*: Interaction detection.
        """)
        arch_text = """
        - **Algorithm**: **Random Forest Classifier** (scikit-learn). An ensemble of decision trees trained to differentiate "Crash" vs "Normal" regimes.
        - **Target**: **Binary (0/1)**. *Is the Maximum Drawdown in the next 12 months > 20%?*
        - **Output**: Probability of Crash (0-100%).
        """
    elif model_type == 'Gradient Boosting':
        st.sidebar.markdown("""
        **Objective**: Binary Classification (Crash in next 12m).
        **Model**: Gradient Boosting
        - *Perf*: Somers' D: 0.444
        - *Strength*: Precision & Calibration.
        """)
        arch_text = """
        - **Algorithm**: **Histogram-based Gradient Boosting**. Sequentially builds trees to correct errors of previous estimators. Using Monotonic Constraints.
        - **Target**: **Binary (0/1)**. *Is the Maximum Drawdown in the next 12 months > 20%?*
        - **Output**: Calibrated Probability (0-100%).
        """
    else:
        st.sidebar.markdown("""
        **Objective**: Time-to-Event Analysis.
        **Model**: Cox Proportional Hazards
        - *Perf*: C-Index: 0.593
        - *Strength*: Timing & Hazard Rate.
        """)
        arch_text = """
        - **Algorithm**: **Cox Proportional Hazards**. Estimates the baseline risk of crash and how factors (rates, valuation) multiply that risk.
        - **Target**: **Time-to-Event**. *Months until the next crash starts.*
        - **Output**: Hazard Score (Relative Risk Multiplier).
        """

    with st.expander("📖 Model Narrative & Strategy", expanded=True):
        if model_type == 'Random Forest':
            st.markdown("""
            ### 📉 Random Forest Classification Narrative
            
            **1. Crash Definition**
            We define a "crash" as a **>20% drawdown from a local peak** (the traditional definition of a Bear Market). The model's target is binary: *Will a crash begin at any point within the next 12 months?* This allows us to capture the "building risk" before the actual price drop is fully realized.
            
            **2. Dataset Choices**
            - **Historical Depth**: Our backtest and training focus start in **1929** to capture the full trajectory of the Great Depression and all subsequent modern cycles.
            - **Leading Indicators**: We prioritize **Housing Starts** and **Term Spreads** as they historically lead market top conditions by several months.
            - **Real-World Lag**: All macro data is lagged by 1 month to ensure we are only using data that was actually available to an investor on the prediction date.
            
            **3. Modeling Decisions**
            A Random Forest was chosen because market crashes are rarely caused by a single factor. They result from **non-linear interactions** (e.g., high CAPE *is* dangerous, but *extremely* dangerous if unemployment is also rising). RF naturally identifies these intersectional risks. We use an **Expanding Window** training strategy to simulate a true "live" walk-forward backtest.
            """)
        elif model_type == 'Gradient Boosting':
            st.markdown("""
            ### 🚀 Gradient Boosting (GBM) Narrative
            
            **1. Crash Definition**
            We define a "crash" as a **>20% drawdown from a local peak** (the traditional definition of a Bear Market). The model's target is binary: *Will a crash begin at any point within the next 12 months?* This allows us to capture the "building risk" before the actual price drop is fully realized.
            
            **2. Dataset Choices**
            Similar to other models, we use the full history from 1929. GBM is particularly effective at handling standard tabular financial data without extensive preprocessing, though we maintain the strict lag structure to prevent leakage.
            
            **3. Modeling Decisions**
            - **Histogram-Based**: We use a modern Histogram-based implementation (similar to LightGBM) for efficiency.
            - **Monotonic Constraints**: We enforce logical rules (e.g., Higher Unemployment *must* increase risk, not decrease it) to ensure the model behaves economically rationally even in unseen data regimes.
            - **Calibration**: The raw scores are calibrated using a Sigmoid function to produce true probabilities (0-100%).
            """)
        else:
            st.markdown("""
            ### ⏳ Survival Analysis (CoxPH) Narrative
            
            **1. Crash Definition**
            In this model, a crash is treated as an **event observed in time**. Unlike classification (which looks for a "yes/no" in a fixed 12m window), the Survival Model estimates the **Hazard Rate**—the instantaneous risk that a crash will begin today, given that it hasn't started yet.
            
            **2. Dataset Choices**
            We use the same multidisciplinary feature set (Valuation, Macro, Rates) but focus on **Hazard Ratios**. For example, the model quantifies exactly how much a 1% increase in Inflation multiplies the hazard of an imminent market correction.
            
            **3. Modeling Decisions**
            We implemented a **Cox Proportional Hazards** model because market cycles exhibit "fat tails" and varying durations. This model is superior for answering the question: *"Regardless of the 12-month window, how much 'time' is realistically left in this bull market regime?"* It provides a full **Survival Curve**, showing the probability of the market surviving (not crashing) over the next 1, 3, 6, and 12+ months.
            """)

    with st.expander("Explore Architecture & Methodology"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"### 🛠 Model Structure ({model_type})")
            st.markdown(arch_text)
        with col_m2:
            st.markdown("### 📊 Feature Engineering")
            st.markdown("""
            - **Valuation**: CAPE and TTM Earnings Yield.
            - **Macro & Labor**: YoY Unemployment, CPI Inflation, and Manufacturing Hours.
            - **Leading Indicators**: Housing Starts, Building Permits, and Consumer Sentiment (YoY/Change).
            - **Rates & Credit**: Term Spread and Credit Spreads.
            - **Market Internals**: 12m Momentum and Price Volatility.
            - **Leakage Prevention**: 
                - **Feature-Specific Lags**: Implemented precise **0-month** (Price/Rates), **1-month** (Macro), and **2-month** (Housing) lags to eliminate lookahead bias.
                - **Forward Horizon Gap**: Training windows include a 12-month "dead zone" before the test date to prevent target labels from leaking into the training features.
            """)
    
    with st.expander("🔬 Validation Strategy & Data Periods"):
        if model_type == 'Survival (CoxPH)':
            st.markdown("""
            ### Validation Plan: Survival Model
            To rigorously test the "Time-to-Event" predictions, we employ a custom **3-Year Expanding Window** strategy.
            
            ### Protocol
            - **Initial Training**: 1929 – 1959 (30 Years).
            - **Walk-Forward**: Starting in 1960, the model predicts the risk for the next **3 years** (OOS). The window then expands by 3 years, retrains on all history, and repeats.
            - **Metric**: **Rolling 60-Month C-Index**. We use a 5-year rolling window to evaluate ranking performance, smoothing out short-term noise.
            
            ### Summary Table
            | Phase | Date Range | Description |
            | :--- | :--- | :--- |
            | **Available History** | 1929 – Present | Core dataset capturing the Great Depression onwards. |
            | **Backtest Start** | 1960 | First out-of-sample prediction block. |
            | **Step Size** | **3 Years** | Retraining frequency to capture major regime shifts. |
            | **Key Metrics** | C-Index, IBS | Evaluates ranking accuracy & probability calibration. |
            """)
        else:
            st.markdown(f"""
            ### Validation Plan: {model_type}
            To ensure robust classification performance, we use a granular **1-Year Expanding Window** strategy.
            
            ### Protocol
            - **Initial Training**: 1929 – 1959 (30 Years).
            - **Walk-Forward**: Starting in 1960, the model predicts crash probability for the **next 1 year** (OOS). The window expands by 1 year, retrains, and repeats.
            - **Metric**: **Rolling 30-Month Somers' D**. We evaluate the rank-ordering of risk scores over a 2.5-year rolling period.
            
            ### Summary Table
            | Phase | Date Range | Description |
            | :--- | :--- | :--- |
            | **Available History** | 1929 – Present | Core dataset capturing the Great Depression onwards. |
            | **Backtest Start** | 1960 | First out-of-sample prediction block. |
            | **Step Size** | **1 Year** | Annual retraining for maximum responsiveness to new data. |
            | **Key Metrics** | Somers' D, Brier, PR-AUC | Assessing discrimination and calibration. |
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
        # Select Date for "Current" View (Time Travel)
        st.subheader("Risk Analysis")
        
        # Default to latest, but allow user to slide back
        valid_dates = predictions.index.sort_values()
        selected_date = st.select_slider(
            "Select Date to Analyze",
            options=valid_dates,
            value=valid_dates[-1],
            format_func=lambda x: x.date()
        )
        
        # Get Data for Selected Date
        selected_row = predictions.loc[selected_date]
        
        if model_type == 'Survival (CoxPH)':
             latest_hazard = selected_row['Hazard_Score']
             st.metric(f"Hazard Score ({selected_date.date()})", f"{latest_hazard:.2f}")
             
             # Dynamic Survival Curve Calculation
             # 1. Load the "Base" curve (which is the curve for the LATEST available date in the CSV)
             latest_curve_df = load_survival_curve()
             
             if latest_curve_df is not None:
                 # The CSV has the curve for the LAST DATE in the training set.
                 # We need to reverse-engineer the Baseline Hazard S0(t)
                 # Formula: S_t(t) = S_0(t) ^ exp(beta*x_t) = S_0(t) ^ Hazard_Score_t
                 # Thus: S_0(t) = S_t(t) ^ (1 / Hazard_Score_t)
                 
                 # Get the hazard score corresponding to the curve's date (latest prediction date)
                 # We assume current_survival_curve.csv corresponds to valid_dates[-1]
                 ref_hazard = predictions.iloc[-1]['Hazard_Score']
                 
                 # Take the first column (should be the probability values)
                 curve_vals = latest_curve_df.iloc[:, 0]
                 
                 # Calculate Baseline
                 baseline_curve = curve_vals ** (1.0 / ref_hazard)
                 
                 # Calculate Selected Date Curve: S_new(t) = S_0(t) ^ new_hazard
                 selected_curve = baseline_curve ** latest_hazard
                 
                 fig_curve = px.line(x=latest_curve_df.index, y=selected_curve, 
                                     title=f"Survival Probability (from {selected_date.date()})", 
                                     labels={'x': 'Months Forward', 'y': 'Prob(No Crash)'})
                 
                 # Fix y-axis to 0-1
                 fig_curve.update_yaxes(range=[0, 1.05])
                 st.plotly_chart(fig_curve, width='stretch')
        else:
            latest_prob = selected_row['Crash_Prob']
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
        with st.expander("🤔 How to read this chart?"):
            if model_type == 'Survival (CoxPH)':
                st.markdown("""
                - **Log Hazard Ratio**: Shows how each feature impacts the 'hazard' (risk) of a crash.
                - **Positive (Red)**: Increases crash risk (e.g., high CAPE).
                - **Negative (Blue)**: Decreases crash risk (e.g., high earnings yield).
                - **Scale**: A coefficient of 0.7 roughly doubles the hazard for a 1-unit increase in the feature.
                """)
            else:
                st.markdown("""
                - **SHAP values**: Quantify the 'push' each feature gives to the final probability for the **latest available month**.
                - **Red (Positive)**: This feature is currently **increasing** the predicted probability of a crash.
                - **Blue (Negative)**: This feature is currently **decreasing** the predicted risk.
                - **Magnitude**: The longer the bar, the more powerful the driver is for today's specific score.
                """)
        if interp_data is not None:
            if isinstance(interp_data, dict):
                method = interp_data.get('method', 'unknown')
                if method == 'shap':
                    shap_vals = interp_data['shap_values']
                    # Ensure we have a 1D vector for the last observation
                    if len(shap_vals.shape) == 2:
                        last_shap = shap_vals[-1, :]
                    elif len(shap_vals.shape) == 3:
                        last_shap = shap_vals[-1, :, 1]
                    else:
                        last_shap = shap_vals
                        
                    feature_names = interp_data['X'].columns
                    shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP': last_shap})
                    shap_df['AbsSHAP'] = shap_df['SHAP'].abs()
                    shap_df = shap_df.sort_values('AbsSHAP', ascending=True).tail(10)
                    fig_shap = px.bar(shap_df, x='SHAP', y='Feature', orientation='h', 
                                      title="Feature Contribution (SHAP)",
                                      color='SHAP', color_continuous_scale=['blue', 'red'])
                    st.plotly_chart(fig_shap, width='stretch')
                elif method == 'importance' or method == 'permutation':
                    feats = interp_data.get('features', FEATURES)
                    imp = interp_data.get('feature_importances', interp_data.get('importances_mean'))
                    fig_fi = px.bar(pd.DataFrame({'Feature': feats, 'Importance': imp}).sort_values('Importance'), 
                                   x='Importance', y='Feature', orientation='h', title="Feature Importance")
                    st.plotly_chart(fig_fi, width='stretch')
            else:
                # Survival Coefficients (DataFrame)
                coef_df = interp_data.copy()
                coef_df['Feature'] = coef_df.index
                coef_df = coef_df.sort_values('coef', ascending=True)
                fig_coef = px.bar(coef_df, x='coef', y='Feature', orientation='h', 
                                  title="Log Hazard Coefficients (Survival Contribution)",
                                  color='coef', color_continuous_scale=['blue', 'red'],
                                  labels={'coef': 'Log Hazard Ratio'})
                st.plotly_chart(fig_coef, width='stretch')
                st.caption("Positive coefficients (red) increase the crash hazard; negative (blue) decrease it.")
        else:
            st.warning("Interpretation data not found.")

    # Timeline
    st.subheader(f"Historical Risk Timeline ({model_type})")
    y_col = 'Hazard_Score' if model_type == 'Survival (CoxPH)' else 'Crash_Prob'
    
    from plotly.subplots import make_subplots
    fig_time = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Prob/Hazard Area
    fig_time.add_trace(
        go.Scatter(x=predictions.index, y=predictions[y_col], fill='tozeroy', name=y_col, fillcolor='rgba(255, 0, 0, 0.2)', line=dict(color='red')),
        secondary_y=False,
    )
    
    # S&P 500 Price Line
    if 'S&P500_Price' in predictions.columns:
        fig_time.add_trace(
            go.Scatter(x=predictions.index, y=predictions['S&P500_Price'], name="S&P 500 Index", line=dict(color='blue', width=2)),
            secondary_y=True,
        )
    
    # Shaded Crash Regions (where 20% Contraction occurred)
    if 'Contraction_Phase' in predictions.columns:
        target_col = 'Contraction_Phase'
        # Find continuous blocks of Crash
        crash_blocks = predictions[predictions[target_col] == 1]
        if not crash_blocks.empty:
            # We add rectangles. To avoid too many traces, we can use shapes.
            diff = predictions[target_col].diff()
            starts = predictions.index[diff == 1]
            ends = predictions.index[diff == -1]
            
            # Handle edge cases (starts with 1 or ends with 1)
            if predictions[target_col].iloc[0] == 1:
                starts = starts.insert(0, predictions.index[0])
            if predictions[target_col].iloc[-1] == 1:
                ends = ends.insert(len(ends), predictions.index[-1])
            
            for start, end in zip(starts, ends):
                fig_time.add_vrect(
                    x0=start, x1=end,
                    fillcolor="gray", opacity=0.3,
                    layer="below", line_width=0,
                    annotation_text="Crash", annotation_position="top left",
                )

    fig_time.update_layout(
        title=f"Risk Score vs. S&P 500 Index",
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_time.update_yaxes(title_text=f"<b>{y_col}</b>", secondary_y=False, range=[0, predictions[y_col].max() * 1.2])
    fig_time.update_yaxes(title_text="<b>S&P 500 Index</b>", secondary_y=True, type="log") # Log scale for index
    
    st.plotly_chart(fig_time, width='stretch', use_container_width=True)
    
    st.markdown("---")
    st.subheader(f"🏆 Overall Model Evaluation (Out-of-Sample)")
    
    if model_type != 'Survival (CoxPH)':
        # Ensure we have targets (drop rows where Target might be NaN if any)
        valid_preds = predictions.dropna(subset=['Target', 'Crash_Prob'])
        
        if len(valid_preds) > 0:
            brier_score = brier_score_loss(valid_preds['Target'], valid_preds['Crash_Prob'])
            pr_auc = average_precision_score(valid_preds['Target'], valid_preds['Crash_Prob'])
            
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Brier Score (Lower is Better)", f"{brier_score:.4f}", help="Mean squared difference between predicted probability and actual outcome. 0 is perfect.")
            with m_col2:
                st.metric("PR-AUC (Higher is Better)", f"{pr_auc:.4f}", help="Area Under the Precision-Recall Curve. Evaluates how well the model identifies crashes without false alarms.")
                
            # PR Curve Plot
            precision, recall, _ = precision_recall_curve(valid_preds['Target'], valid_preds['Crash_Prob'])
            
            fig_pr = px.area(
                x=recall, y=precision,
                title=f'Precision-Recall Curve (AUC={pr_auc:.4f})',
                labels=dict(x='Recall (Sensitivity)', y='Precision (Positive Predictive Value)'),
            )
            # Add baseline (no skill)
            baseline = valid_preds['Target'].mean()
            fig_pr.add_shape(
                type='line', line=dict(dash='dash', color='gray'),
                x0=0, x1=1, y0=baseline, y1=baseline
            )
            fig_pr.add_annotation(x=0.5, y=baseline, text=f"Baseline (No Skill): {baseline:.2f}", showarrow=False, yshift=10)
            
            st.plotly_chart(fig_pr, use_container_width=True)
    else:
        # Survival Model Evaluation
        # Check if we have probability columns for IBS
        # Calculate approximate IBS
        surv_cols = [c for c in predictions.columns if 'Surv_Prob' in c]
        if surv_cols:
            brier_scores = []
            valid_preds = predictions.dropna(subset=surv_cols)
            
            for col in surv_cols:
                # Extract horizon from name 'Surv_Prob_12m' -> 12
                try:
                    t = int(col.split('_')[-1].replace('m', ''))
                    
                    # Evaluation Logic (same as training script)
                    mask_usable = (valid_preds['Duration'] > t) | (valid_preds['Event'] == 1)
                    subset = valid_preds[mask_usable].copy()
                    
                    if len(subset) > 0:
                        subset['Target_at_t'] = (subset['Duration'] <= t).astype(int)
                        subset['Prob_Event_at_t'] = 1 - subset[col]
                        bs = ((subset['Target_at_t'] - subset['Prob_Event_at_t']) ** 2).mean()
                        brier_scores.append(bs)
                except:
                    continue
            
            if brier_scores:
                ibs = np.mean(brier_scores)
                st.metric("Integrated Brier Score (1-3 Year Avg)", f"{ibs:.4f}", help="Average squared error of survival probability across 12, 24, 36 month horizons. Lower is better.")
            else:
                st.info("Insufficient data for Brier Score calculation.")
        else:
            st.info("Re-run pipeline to generate survival probabilities for Brier Score.")

    # Backtest Performance
    perf_label = "C-Index" if model_type == 'Survival (CoxPH)' else "Somers' D"
    st.subheader(f"Model Performance (Rolling {perf_label})")
    
    with st.expander("🎓 How is this calculated?"):
            if model_type == 'Survival (CoxPH)':
                st.markdown("""
                **Rolling C-Index (Concordance Index)**:
                - **Window**: 60-month (5 Year) out-of-sample period.
                - **Calculation**: Measures the proportion of pairs where the model correctly predicted which event would happen first.
                - **Interpretation**: A score of 0.5 is random; 1.0 is a perfect ranking of "Time-to-Crash."
                """)
            else:
                st.markdown("""
                **Rolling Somers' D**:
                - **Window**: 30-month (2.5 Year) out-of-sample period.
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
