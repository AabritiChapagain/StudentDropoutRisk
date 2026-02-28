# STUDENT RISK PREDICTION SYSTEM - STREAMLIT UI

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

THEME = {
    "primary":   "#2563eb",
    "secondary": "#7c3aed",
    "success":   "#059669",
    "danger":    "#dc2626",
    "warning":   "#f59e0b",
    "info":      "#14b8a6",
    "dark":      "#1e293b",
    "light":     "#f8fafc",
    "card":      "#ffffff",
    "text":      "#1e293b",
    "muted":     "#64748b"
}
# Add system folder to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'system'))
import prediction_system

# Page configuration
st.set_page_config(
    page_title="Student Academic Risk Predictor", 
    page_icon="📚", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Different color scheme and design
st.markdown(f"""
<style>
    .main {{ background-color: {THEME['light']}; }}

    .main-header {{ 
        font-size: 3rem; 
        font-weight: 800; 
        background: linear-gradient(120deg, {THEME['primary']} 0%, {THEME['secondary']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center; 
        margin-bottom: 0.3rem;
        padding: 1rem 0;
    }}

    .sub-header {{ 
        font-size: 1.2rem; 
        color: {THEME['muted']}; 
        text-align: center; 
        margin-bottom: 2.5rem;
        font-weight: 500;
    }}

    .risk-card-danger {{ 
        background: linear-gradient(135deg, {THEME['danger']} 0%, #991b1b 100%); 
        padding: 2.5rem; 
        border-radius: 20px; 
        color: white; 
        text-align: center; 
        font-size: 1.8rem; 
        font-weight: 800; 
        box-shadow: 0 10px 30px rgba(220, 38, 38, 0.3);
        border: 3px solid #fca5a5;
        margin: 1rem 0;
    }}

    .risk-card-safe {{ 
        background: linear-gradient(135deg, {THEME['success']} 0%, #047857 100%); 
        padding: 2.5rem; 
        border-radius: 20px; 
        color: white; 
        text-align: center; 
        font-size: 1.8rem; 
        font-weight: 800; 
        box-shadow: 0 10px 30px rgba(5, 150, 105, 0.3);
        border: 3px solid #6ee7b7;
        margin: 1rem 0;
    }}

    .info-card {{ 
        background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%); 
        border-left: 5px solid {THEME['info']}; 
        padding: 1.5rem; 
        border-radius: 12px; 
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}

    .warning-card {{ 
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
        border-left: 5px solid {THEME['warning']}; 
        padding: 1.5rem; 
        border-radius: 12px; 
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}

    .metric-container {{ 
        background: {THEME['card']}; 
        padding: 1.8rem; 
        border-radius: 15px; 
        text-align: center; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-top: 4px solid {THEME['secondary']};
        margin: 0.8rem 0;
    }}

    .stButton>button {{ 
        background: linear-gradient(135deg, {THEME['primary']} 0%, {THEME['secondary']} 100%); 
        color: white; 
        border: none; 
        padding: 0.9rem 2.5rem; 
        font-size: 1.2rem; 
        font-weight: 700; 
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
        transition: all 0.3s ease;
    }}

    .stButton>button:hover {{ 
        background: linear-gradient(135deg, {THEME['secondary']} 0%, {THEME['primary']} 100%);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
        transform: translateY(-2px);
    }}

    .css-1d391kg {{ background-color: {THEME['dark']}; }}

</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load dataset and prepare for modeling"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_student_data.csv')
    df, features = prediction_system.load_dataset(data_path)
    return df, features


@st.cache_resource
def train_models_pipeline(_df, _features):
    """Train Logistic Regression model"""
    X = _df[_features].copy()
    y = _df['Risk'].copy()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = prediction_system.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = prediction_system.scale_features(X_train, X_val, X_test)
    
    # Train Logistic Regression model
    with st.spinner("Training Logistic Regression Model..."):
        lr_model, y_test_pred = prediction_system.train_logistic_regression(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
        )
    
    # Calculate metrics
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    lr_cm = confusion_matrix(y_test, y_test_pred)
    
    # Calculate validation accuracy
    y_val_pred = lr_model.predict(X_val_scaled)
    lr_val_acc = accuracy_score(y_val, y_val_pred)
    lr_test_acc = accuracy_score(y_test, y_test_pred)
    lr_precision = precision_score(y_test, y_test_pred)
    lr_recall = recall_score(y_test, y_test_pred)
    lr_f1 = f1_score(y_test, y_test_pred)
    
    return {
        'lr_model': lr_model,
        'scaler': scaler,
        'lr_test_acc': lr_test_acc,
        'lr_val_acc': lr_val_acc,
        'lr_precision': lr_precision,
        'lr_recall': lr_recall,
        'lr_f1': lr_f1,
        'lr_cm': lr_cm,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'X_test_scaled': X_test_scaled
    }


# Load data and train models
try:
    df, FEATURES = load_and_prepare_data()
    models = train_models_pipeline(df, FEATURES)
except Exception as e:
    st.error(f"Error loading data or training models: {e}")
    st.stop()


# Sidebar Navigation
st.sidebar.markdown("## 📚 Navigation Panel")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select a Page:",
    ["🎯 Risk Prediction", "� Model Performance", "ℹ️ About System"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")
st.sidebar.metric("Total Students", len(df))
st.sidebar.metric("At Risk", (df['Risk'] == 1).sum())
st.sidebar.metric("Safe", (df['Risk'] == 0).sum())
st.sidebar.metric("Risk Rate", f"{(df['Risk'] == 1).mean()*100:.1f}%")


def page_risk_prediction():
    """Page for predicting student risk"""
    st.markdown('<h1 class="main-header">🎯 Student Academic Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Risk Analysis using Logistic Regression</p>', unsafe_allow_html=True)
    
    # Info about the model
    st.markdown(f"""
    <div class="info-card" style="color: #000000;">
    <strong>🤖 Logistic Regression Model:</strong> This system uses a highly accurate Logistic Regression classifier 
    with <strong>{len(FEATURES)}</strong> features to predict academic risk. The model achieved <strong>{models['lr_test_acc']*100:.1f}% accuracy</strong> 
    on test data with <strong>{models['lr_precision']*100:.1f}% precision</strong> and <strong>{models['lr_f1']*100:.1f}% F1-score</strong>.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📝 Input Student Information")
    
    # Create dynamic input fields based on features
    col1, col2, col3 = st.columns(3)
    
    input_data = {}
    
    # Distribute features across columns
    features_per_col = len(FEATURES) // 3 + 1
    
    with col1:
        st.markdown("#### 📊 Academic Performance")
        for i, feature in enumerate(FEATURES[:features_per_col]):
            if 'grade' in feature.lower() or 'approved' in feature.lower():
                input_data[feature] = st.slider(
                    feature.replace('_', ' ').title(), 
                    float(df[feature].min()), 
                    float(df[feature].max()), 
                    float(df[feature].median()),
                    key=f"pred_{feature}"
                )
            else:
                input_data[feature] = st.number_input(
                    feature.replace('_', ' ').title(),
                    float(df[feature].min()),
                    float(df[feature].max()),
                    float(df[feature].median()),
                    key=f"pred_{feature}"
                )
    
    with col2:
        st.markdown("#### 📚 Enrollment & Evaluation")
        for i, feature in enumerate(FEATURES[features_per_col:features_per_col*2]):
            if 'enrolled' in feature.lower() or 'evaluations' in feature.lower():
                input_data[feature] = st.slider(
                    feature.replace('_', ' ').title(),
                    float(df[feature].min()),
                    float(df[feature].max()),
                    float(df[feature].median()),
                    key=f"pred_{feature}"
                )
            else:
                input_data[feature] = st.number_input(
                    feature.replace('_', ' ').title(),
                    float(df[feature].min()),
                    float(df[feature].max()),
                    float(df[feature].median()),
                    key=f"pred_{feature}"
                )
    
    with col3:
        st.markdown("#### 🎓 Other Factors")
        for i, feature in enumerate(FEATURES[features_per_col*2:]):
            if feature.lower() in ['tuition fees up to date', 'gender', 'scholarship holder']:
                input_data[feature] = st.selectbox(
                    feature.replace('_', ' ').title(),
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    key=f"pred_{feature}"
                )
            elif 'course' in feature.lower():
                input_data[feature] = st.slider(
                    feature.replace('_', ' ').title(),
                    int(df[feature].min()),
                    int(df[feature].max()),
                    int(df[feature].median()),
                    key=f"pred_{feature}"
                )
            else:
                input_data[feature] = st.number_input(
                    feature.replace('_', ' ').title(),
                    float(df[feature].min()),
                    float(df[feature].max()),
                    float(df[feature].median()),
                    key=f"pred_{feature}"
                )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🔮 ANALYZE RISK LEVEL", use_container_width=True)
    
    if predict_btn:
        # Prepare input
        input_df = pd.DataFrame([input_data])
        input_scaled = models['scaler'].transform(input_df)
        
        # Get predictions
        lr_pred = models['lr_model'].predict(input_scaled)[0]
        lr_proba = models['lr_model'].predict_proba(input_scaled)[0]
        
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        # Display prediction in a prominent card
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🤖 Logistic Regression Prediction")
            if lr_pred == 1:
                st.markdown('<div class="risk-card-danger">⚠️ STUDENT AT RISK</div>', unsafe_allow_html=True)
                st.error(f"📉 Risk Probability: {lr_proba[1]*100:.1f}%")
            else:
                st.markdown('<div class="risk-card-safe">✅ STUDENT SAFE</div>', unsafe_allow_html=True)
                st.success(f"📊 Success Probability: {lr_proba[0]*100:.1f}%")
        
        st.markdown("---")
        
        # Detailed metrics
        st.markdown("### 📊 Detailed Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Model Accuracy", f"{models['lr_test_acc']*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Precision", f"{models['lr_precision']*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Recall", f"{models['lr_recall']*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("F1-Score", f"{models['lr_f1']*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk gauge
        st.markdown("### 🎯 Risk Level Visualization")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=lr_proba[1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Percentage", 'font': {'size': 24, 'color': '#1e293b'}},
            delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
                'bar': {'color': "#dc2626" if lr_proba[1] > 0.5 else "#059669", 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#cbd5e1",
                'steps': [
                    {'range': [0, 33], 'color': '#d1fae5'},
                    {'range': [33, 66], 'color': '#fef3c7'},
                    {'range': [66, 100], 'color': '#fecaca'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#1e293b", 'family': "Arial, sans-serif"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations based on risk
        st.markdown("---")
        st.markdown("### 💡 Recommendations")
        
        if lr_pred == 1:
            st.markdown("""
            <div class="warning-card">
            <h4>🚨 High Risk Detected - Immediate Action Required</h4>
            <p>Based on the analysis, this student shows signs of academic risk. Consider the following interventions:</p>
            <ul>
                <li>📚 Schedule one-on-one academic counseling sessions</li>
                <li>👥 Connect with peer tutoring or study groups</li>
                <li>📅 Develop a structured study plan with clear milestones</li>
                <li>📝 Review and improve course grades, especially in challenging subjects</li>
                <li>💰 Address any financial concerns (tuition fees, scholarships)</li>
                <li>✅ Focus on completing enrolled courses successfully</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-card" style="color: #000000;">
            <h4>✅ Low Risk - Keep Up the Good Work!</h4>
            <p>The student is performing well academically. To maintain this positive trajectory:</p>
            <ul>
                <li>🎯 Continue current study habits and time management strategies</li>
                <li>📚 Stay engaged with coursework and attend classes regularly</li>
                <li>🤝 Participate in collaborative learning opportunities</li>
                <li>📈 Set challenging but achievable academic goals</li>
                <li>💬 Maintain open communication with instructors</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)


def page_model_performance():
    """Page showing model performance analytics"""
    st.markdown('<h1 class="main-header">📊 Model Performance Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive analysis of Logistic Regression model accuracy and metrics</p>', unsafe_allow_html=True)
    
    # Performance metrics header
    st.markdown("### 🎯 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Test Accuracy", f"{models['lr_test_acc']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Validation Accuracy", f"{models['lr_val_acc']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Precision", f"{models['lr_precision']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Recall", f"{models['lr_recall']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics explanation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card" style="color: #000000;">
        <h4>📊 Metrics Explanation:</h4>
        <ul>
            <li><strong>Accuracy:</strong> Overall correctness of predictions</li>
            <li><strong>Precision:</strong> Accuracy of positive (at-risk) predictions</li>
            <li><strong>Recall:</strong> Ability to find all at-risk students</li>
            <li><strong>F1-Score:</strong> Harmonic mean of precision and recall</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("F1-Score", f"{models['lr_f1']*100:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if models['lr_test_acc'] == 1.0:
            st.success("✅ Perfect Model Performance!")
        elif models['lr_test_acc'] >= 0.95:
            st.success("✅ Excellent Model Performance!")
        elif models['lr_test_acc'] >= 0.90:
            st.info("ℹ️ Very Good Model Performance")
        else:
            st.warning("⚠️ Model Performance Could Be Improved")
    
    st.markdown("---")
    
    # Confusion matrix
    st.markdown("### 🎯 Confusion Matrix")
    
    st.markdown("### 🎯 Confusion Matrix")
    
    st.markdown("""
    <div class="info-card" style="color: #000000;">
    <strong>📊 Understanding the Confusion Matrix:</strong><br>
    - <strong>True Negatives (Top-Left):</strong> Correctly predicted as Safe<br>
    - <strong>False Positives (Top-Right):</strong> Incorrectly predicted as At Risk<br>
    - <strong>False Negatives (Bottom-Left):</strong> Missed At Risk students<br>
    - <strong>True Positives (Bottom-Right):</strong> Correctly predicted as At Risk
    </div>
    """, unsafe_allow_html=True)
    
    # Display confusion matrix
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        fig_cm = px.imshow(
            models['lr_cm'],
            text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
            x=['Safe (0)', 'At Risk (1)'],
            y=['Safe (0)', 'At Risk (1)']
        )
        fig_cm.update_layout(
            height=450,
            title="Logistic Regression Confusion Matrix",
            title_x=0.5,
            font=dict(size=12)
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # Confusion matrix breakdown
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("True Negatives", models['lr_cm'][0][0])
        st.markdown("Correctly Safe</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("False Positives", models['lr_cm'][0][1])
        st.markdown("False Alarms</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("False Negatives", models['lr_cm'][1][0])
        st.markdown("Missed Risks</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("True Positives", models['lr_cm'][1][1])
        st.markdown("Correctly At Risk</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ROC Analysis
    st.markdown("### 📈 ROC Curve Analysis")
    
    from sklearn.metrics import roc_curve, auc
    
    # Calculate ROC curve
    y_test = models['y_test']
    y_proba = models['lr_model'].predict_proba(models['X_test_scaled'])[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create ROC curve plot
    fig_roc = go.Figure()
    
    # Plot ROC curve
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        line=dict(color='#2563eb', width=3),
        name=f'Logistic Regression (AUC = {roc_auc:.3f})'
    ))
    
    # Plot diagonal line (random classifier)
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        name='Random Classifier'
    ))
    
    fig_roc.update_layout(
        title='ROC Curve - Receiver Operating Characteristic',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate (Recall)',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(x=0.6, y=0.1),
        font=dict(size=12)
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # AUC interpretation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("AUC Score", f"{roc_auc:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if roc_auc >= 0.90:
            st.success("✅ Excellent discrimination ability!")
        elif roc_auc >= 0.80:
            st.success("✅ Good discrimination ability")
        elif roc_auc >= 0.70:
            st.info("ℹ️ Fair discrimination ability")
        else:
            st.warning("⚠️ Needs improvement")
    
    st.markdown("""
    <div class="info-card" style="color: #000000;">
    <h4>📖 AUC Score Interpretation:</h4>
    <ul>
        <li><strong>0.90-1.00:</strong> Excellent model performance</li>
        <li><strong>0.80-0.90:</strong> Good model performance</li>
        <li><strong>0.70-0.80:</strong> Fair model performance</li>
        <li><strong>0.60-0.70:</strong> Poor model performance</li>
        <li><strong>0.50-0.60:</strong> Fail (no better than random guessing)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset distribution
    st.markdown("### 📊 Dataset Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Class distribution
        class_dist = pd.DataFrame({
            'Class': ['Safe (0)', 'At Risk (1)'],
            'Count': [(df['Risk'] == 0).sum(), (df['Risk'] == 1).sum()]
        })
        
        fig_dist = px.pie(
            class_dist,
            values='Count',
            names='Class',
            color='Class',
            color_discrete_map={'Safe (0)': '#059669', 'At Risk (1)': '#dc2626'},
            hole=0.4
        )
        
        fig_dist.update_traces(textposition='inside', textinfo='percent+label+value', textfont_size=13)
        fig_dist.update_layout(
            height=400,
            title='Target Class Distribution',
            title_x=0.5
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Students", len(df))
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Safe Students", (df['Risk'] == 0).sum())
        st.markdown(f"{(df['Risk'] == 0).mean()*100:.1f}% of total</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("At Risk Students", (df['Risk'] == 1).sum())
        st.markdown(f"{(df['Risk'] == 1).mean()*100:.1f}% of total</div>", unsafe_allow_html=True)


def page_system_info():
    """Page showing system information"""
    st.markdown('<h1 class="main-header">ℹ️ About the System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Learn about the AI model and technology behind this prediction system</p>', unsafe_allow_html=True)
    
    # Overview
    st.markdown("### 🎯 System Overview")
    st.markdown("""
    <div style="color: #000000;">
    This Academic Risk Prediction System uses <strong>Logistic Regression</strong>, a powerful machine learning algorithm, 
    to identify students who may be at risk of academic failure. The system analyzes multiple factors including 
    academic performance, enrollment patterns, and demographic information to provide accurate predictions and 
    actionable recommendations.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🤖 Logistic Regression Model")
        st.markdown("""
        <div class="info-card" style="color: #000000;">
        <h4>📉 Why Logistic Regression?</h4>
        <p>Logistic Regression is a statistical model that predicts the probability of a binary outcome 
        (Safe vs At Risk). It's particularly effective for this application because:</p>
        <ul>
            <li><strong>Interpretable:</strong> Easy to understand which features influence predictions</li>
            <li><strong>Probabilistic:</strong> Provides confidence scores (0-100%) for predictions</li>
            <li><strong>Efficient:</strong> Fast training and real-time predictions</li>
            <li><strong>Reliable:</strong> Well-tested algorithm with consistent performance</li>
            <li><strong>Balanced:</strong> Uses regularization to prevent overfitting</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card" style="color: #000000;">
        <h4>⚙️ Model Configuration</h4>
        <ul>
            <li><strong>Algorithm:</strong> Logistic Regression with L1/L2 regularization</li>
            <li><strong>Solver:</strong> liblinear (optimized for small to medium datasets)</li>
            <li><strong>Class Weight:</strong> Balanced (accounts for imbalanced classes)</li>
            <li><strong>Hyperparameter Tuning:</strong> GridSearchCV with 5-fold cross-validation</li>
            <li><strong>Feature Scaling:</strong> StandardScaler (mean=0, std=1)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Model Stats")
        
        st.markdown(f"""
        <div class="metric-container">
        <h3>{models['lr_test_acc']*100:.0f}%</h3>
        <p>Test Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
        <h3>{models['lr_precision']*100:.0f}%</h3>
        <p>Precision</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
        <h3>{models['lr_recall']*100:.0f}%</h3>
        <p>Recall</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
        <h3>{models['lr_f1']*100:.0f}%</h3>
        <p>F1-Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Dataset Information")
        
        st.markdown(f"""
        <div class="metric-container">
        <h3>{len(df)}</h3>
        <p>Total Students</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
        <h3>{len(FEATURES)}</h3>
        <p>Features Used</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
        <h3>{(df['Risk'] == 1).mean()*100:.1f}%</h3>
        <p>Students At Risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 Data Split")
        st.markdown("""
        <div class="info-card" style="color: #000000;">
        <ul>
            <li><strong>Training Set:</strong> 60% (Model learning)</li>
            <li><strong>Validation Set:</strong> 20% (Hyperparameter tuning)</li>
            <li><strong>Test Set:</strong> 20% (Final evaluation)</li>
        </ul>
        <p>This split ensures the model is properly validated and prevents overfitting.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features table
    st.markdown("### 📋 Features Used for Prediction")
    
    features_df = pd.DataFrame({
        'Feature Name': FEATURES,
        'Description': [f.replace('_', ' ').title() for f in FEATURES]
    })
    
    st.dataframe(features_df, use_container_width=True)
    
    st.markdown("---")
    
    # Target definition
    st.markdown("### 🎯 Risk Definition")
    st.markdown("""
    <div class="warning-card" style="color: #000000;">
    <h4>What does "At Risk" mean?</h4>
    <p>A student is classified as <strong>"At Risk"</strong> (Risk = 1) based on their academic 
    performance indicators and enrollment patterns. The system analyzes historical data to identify 
    patterns that correlate with academic difficulties.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technology stack
    st.markdown("### 🛠️ Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="color: #000000;">
        <strong>Frontend</strong><br>
        - Streamlit<br>
        - Plotly
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="color: #000000;">
        <strong>ML Libraries</strong><br>
        - scikit-learn<br>
        - pandas<br>
        - numpy
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="color: #000000;">
        <strong>Visualization</strong><br>
        - Plotly Express<br>
        - Plotly Graph Objects
        </div>
        """, unsafe_allow_html=True)


# Page router
if page == "🎯 Risk Prediction":
    page_risk_prediction()
elif page == "� Model Performance":
    page_model_performance()
elif page == "ℹ️ About System":
    page_system_info()

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #64748b; padding: 2rem 0;'>
    <p><strong>🎓 Student Academic Risk Prediction System</strong></p>
    <p>Powered by Logistic Regression ML Model | Built with Streamlit & Python</p>
    <p style='font-size: 0.85rem;'>Achieving {models['lr_test_acc']*100:.1f}% Accuracy with {models['lr_f1']*100:.1f}% F1-Score</p>
</div>
""", unsafe_allow_html=True)
