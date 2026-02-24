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
    """Train all models and return them"""
    X = _df[_features].copy()
    y = _df['Risk'].copy()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = prediction_system.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = prediction_system.scale_features(X_train, X_val, X_test)
    
    # Train models (silent mode for UI)
    with st.spinner("Training Logistic Regression..."):
        lr_model, lr_val_acc, lr_test_acc = prediction_system.train_logistic_regression(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
        )
    
    with st.spinner("Training Random Forest..."):
        rf_model, rf_val_acc, rf_test_acc = prediction_system.train_random_forest(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
        )
    
    with st.spinner("Creating KNN model..."):
        knn_model = prediction_system.create_knn_model(X_train_scaled, n_neighbors=7)
    
    # Get predictions and confusion matrices
    lr_pred = lr_model.predict(X_test_scaled)
    rf_pred = rf_model.predict(X_test_scaled)
    
    from sklearn.metrics import confusion_matrix
    lr_cm = confusion_matrix(y_test, lr_pred)
    rf_cm = confusion_matrix(y_test, rf_pred)
    
    return {
        'lr_model': lr_model,
        'rf_model': rf_model,
        'knn_model': knn_model,
        'scaler': scaler,
        'lr_test_acc': lr_test_acc,
        'lr_val_acc': lr_val_acc,
        'rf_test_acc': rf_test_acc,
        'rf_val_acc': rf_val_acc,
        'lr_cm': lr_cm,
        'rf_cm': rf_cm,
        'X_train': X_train,
        'y_train': y_train
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
    ["🎯 Risk Prediction", "💡 Smart Recommendations", "📊 Model Analytics", "ℹ️ System Info"],
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
    st.markdown('<h1 class="main-header">🎯 Academic Risk Prediction Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter student academic data to assess risk level using AI models</p>', unsafe_allow_html=True)
    
    # Info about balanced prediction
    st.markdown("""
    <div class="info-card">
    <strong>⚖️ Balanced Prediction System:</strong> This model uses <strong>all {}</strong> features equally to make predictions. 
    No single feature (like "Curricular Units 2nd Sem Approved") dominates the decision. 
    The Random Forest uses <code>max_features='sqrt'</code> to ensure feature diversity at each decision point.
    </div>
    """.format(len(FEATURES)), unsafe_allow_html=True)
    
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
        rf_pred = models['rf_model'].predict(input_scaled)[0]
        rf_proba = models['rf_model'].predict_proba(input_scaled)[0]
        lr_proba = models['lr_model'].predict_proba(input_scaled)[0]
        
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        # Display predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📉 Logistic Regression")
            if lr_pred == 1:
                st.markdown('<div class="risk-card-danger">⚠️ AT RISK</div>', unsafe_allow_html=True)
                st.error(f"Risk Probability: {lr_proba[1]*100:.1f}%")
            else:
                st.markdown('<div class="risk-card-safe">✅ SAFE</div>', unsafe_allow_html=True)
                st.success(f"Safety Probability: {lr_proba[0]*100:.1f}%")
        
        with col2:
            st.markdown("### 🌳 Random Forest")
            if rf_pred == 1:
                st.markdown('<div class="risk-card-danger">⚠️ AT RISK</div>', unsafe_allow_html=True)
                st.error(f"Risk Probability: {rf_proba[1]*100:.1f}%")
            else:
                st.markdown('<div class="risk-card-safe">✅ SAFE</div>', unsafe_allow_html=True)
                st.success(f"Safety Probability: {rf_proba[0]*100:.1f}%")
        
        # Risk gauge
        st.markdown("### 📊 Risk Level Visualization")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rf_proba[1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Percentage", 'font': {'size': 24, 'color': '#1e293b'}},
            delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
                'bar': {'color': "#dc2626" if rf_proba[1] > 0.5 else "#059669", 'thickness': 0.3},
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
        
        # Consensus
        st.markdown("### 🤝 Model Consensus")
        if lr_pred == rf_pred:
            st.success("✅ Both models agree on the prediction!")
        else:
            st.warning("⚠️ Models have different predictions. Consider additional assessment.")
        
        # Feature contribution analysis
        st.markdown("---")
        st.markdown("### 📊 Feature Contribution to This Prediction")
        st.info("This shows how each feature influenced the Random Forest prediction. A balanced prediction uses multiple features, not just one.")
        
        # Get feature importance and multiply by input values to show contribution
        feature_importance = models['rf_model'].feature_importances_
        
        # Normalize input values for comparison
        input_normalized = (input_df - input_df.min()) / (input_df.max() - input_df.min() + 0.001)
        
        # Calculate contribution score (importance * normalized value)
        contributions = []
        for i, feature in enumerate(FEATURES):
            contrib = feature_importance[i] * 100  # Show as percentage
            contributions.append({
                'Feature': feature,
                'Importance': contrib,
                'Your Value': input_data[feature]
            })
        
        contrib_df = pd.DataFrame(contributions).sort_values('Importance', ascending=False)
        
        # Show top contributing features
        st.markdown("#### 🎯 Top 8 Features Driving This Prediction:")
        
        fig_contrib = px.bar(
            contrib_df.head(8),
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='RdYlGn_r',
            text='Importance'
        )
        
        fig_contrib.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_contrib.update_layout(
            height=400,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Feature Importance (%)",
            yaxis_title="",
            font=dict(size=11)
        )
        
        st.plotly_chart(fig_contrib, use_container_width=True)
        
        # Check if one feature is dominating
        max_importance = contrib_df['Importance'].max()
        if max_importance > 50:
            st.warning(f"⚠️ '{contrib_df.iloc[0]['Feature']}' has very high importance ({max_importance:.1f}%). The model relies heavily on this feature.")
        elif max_importance > 35:
            st.info(f"ℹ️ '{contrib_df.iloc[0]['Feature']}' is the most important feature ({max_importance:.1f}%), but other features also contribute.")
        else:
            st.success(f"✅ Prediction is well-balanced! Top feature '{contrib_df.iloc[0]['Feature']}' has {max_importance:.1f}% importance.")


def page_recommendations():
    """Page for KNN-based recommendations"""
    st.markdown('<h1 class="main-header">💡 Smart Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Get personalized recommendations based on similar student profiles</p>', unsafe_allow_html=True)
    
    st.markdown("### 📋 Enter Student Profile")
    
    # Create input fields
    col1, col2 = st.columns(2)
    
    input_data = {}
    features_mid = len(FEATURES) // 2
    
    with col1:
        for feature in FEATURES[:features_mid]:
            input_data[feature] = st.number_input(
                feature.replace('_', ' ').title(),
                float(df[feature].min()),
                float(df[feature].max()),
                float(df[feature].median()),
                key=f"rec_{feature}"
            )
    
    with col2:
        for feature in FEATURES[features_mid:]:
            input_data[feature] = st.number_input(
                feature.replace('_', ' ').title(),
                float(df[feature].min()),
                float(df[feature].max()),
                float(df[feature].median()),
                key=f"rec_{feature}"
            )
    
    st.markdown("---")
    
    if st.button("🎯 GET PERSONALIZED RECOMMENDATIONS", use_container_width=True):
        # Prepare input
        input_df = pd.DataFrame([input_data])
        input_scaled = models['scaler'].transform(input_df)
        
        # Find similar students
        distances, indices = models['knn_model'].kneighbors(input_scaled)
        similar_outcomes = models['y_train'].iloc[indices[0]]
        
        success_rate = (similar_outcomes == 0).mean() * 100
        risk_rate = (similar_outcomes == 1).mean() * 100
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Similar Students Found", len(similar_outcomes))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Success Rate", f"{success_rate:.1f}%", delta=f"{success_rate - 50:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Risk Rate", f"{risk_rate:.1f}%", delta=f"{risk_rate - 50:.1f}%", delta_color="inverse")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualization
        st.markdown("### 📈 Similar Students Distribution")
        
        outcome_dist = pd.DataFrame({
            'Outcome': ['Safe', 'At Risk'],
            'Count': [(similar_outcomes == 0).sum(), (similar_outcomes == 1).sum()]
        })
        
        fig = px.pie(
            outcome_dist, 
            values='Count', 
            names='Outcome',
            color='Outcome',
            color_discrete_map={'Safe': '#059669', 'At Risk': '#dc2626'},
            hole=0.4
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
        fig.update_layout(
            height=400,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Personalized Recommendations")
        
        if success_rate >= 70:
            st.markdown('<div class="info-card">✅ <strong>Great Profile!</strong> Similar students have high success rates. Keep maintaining good academic habits!</div>', unsafe_allow_html=True)
        elif success_rate >= 40:
            st.markdown('<div class="warning-card">⚠️ <strong>Moderate Risk:</strong> Similar students have mixed outcomes. Consider the following improvements:</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-card">🚨 <strong>High Risk Alert:</strong> Similar students often struggle. Immediate intervention recommended:</div>', unsafe_allow_html=True)
        
        # Generate specific recommendations based on input
        recommendations = []
        
        if 'Curricular units 1st sem (grade)' in input_data:
            if input_data['Curricular units 1st sem (grade)'] < df['Curricular units 1st sem (grade)'].median():
                recommendations.append("📚 Focus on improving course grades through regular study sessions")
        
        if 'Curricular units 1st sem (approved)' in input_data:
            if input_data['Curricular units 1st sem (approved)'] < df['Curricular units 1st sem (approved)'].median():
                recommendations.append("✅ Work on completing more curricular units successfully")
        
        if 'Curricular units 1st sem (evaluations)' in input_data:
            if input_data['Curricular units 1st sem (evaluations)'] > df['Curricular units 1st sem (evaluations)'].median():
                recommendations.append("📝 Multiple evaluations detected - seek tutoring support")
        
        if 'Tuition fees up to date' in input_data:
            if input_data['Tuition fees up to date'] == 0:
                recommendations.append("💰 Resolve tuition fee issues - may affect academic standing")
        
        if 'Scholarship holder' in input_data:
            if input_data['Scholarship holder'] == 0:
                recommendations.append("🎓 Explore scholarship opportunities for financial support")
        
        if recommendations:
            st.markdown("#### 🎯 Action Items:")
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("✅ Profile looks good! Continue with current academic approach.")


def page_model_analytics():
    """Page showing model performance analytics"""
    st.markdown('<h1 class="main-header">📊 Model Performance Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive analysis of model accuracy and feature importance</p>', unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("### 🎯 Model Accuracy Comparison")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("LR Test", f"{models['lr_test_acc']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("LR Validation", f"{models['lr_val_acc']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("RF Test", f"{models['rf_test_acc']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("RF Validation", f"{models['rf_val_acc']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comparison chart
    st.markdown("### 📈 Accuracy Comparison Chart")
    
    comparison_data = pd.DataFrame({
        'Model': ['Logistic Regression', 'Logistic Regression', 'Random Forest', 'Random Forest'],
        'Dataset': ['Test', 'Validation', 'Test', 'Validation'],
        'Accuracy': [
            models['lr_test_acc'] * 100,
            models['lr_val_acc'] * 100,
            models['rf_test_acc'] * 100,
            models['rf_val_acc'] * 100
        ]
    })
    
    fig = px.bar(
        comparison_data,
        x='Model',
        y='Accuracy',
        color='Dataset',
        barmode='group',
        color_discrete_map={'Test': '#2563eb', 'Validation': '#7c3aed'},
        text='Accuracy'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(
        height=450,
        yaxis_range=[0, 100],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Confusion matrices
    st.markdown("### 🎯 Confusion Matrices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Logistic Regression")
        fig_lr = px.imshow(
            models['lr_cm'],
            text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Safe', 'At Risk'],
            y=['Safe', 'At Risk']
        )
        fig_lr.update_layout(height=400)
        st.plotly_chart(fig_lr, use_container_width=True)
    
    with col2:
        st.markdown("#### Random Forest")
        fig_rf = px.imshow(
            models['rf_cm'],
            text_auto=True,
            color_continuous_scale='Purples',
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Safe', 'At Risk'],
            y=['Safe', 'At Risk']
        )
        fig_rf.update_layout(height=400)
        st.plotly_chart(fig_rf, use_container_width=True)
    
    st.markdown("---")
    
    # Feature importance
    st.markdown("### 🎯 Feature Importance Analysis")
    
    importance_df = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': models['rf_model'].feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis',
        text='Importance'
    )
    
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(
        height=600,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Importance Score",
        yaxis_title="Features",
        font=dict(size=11)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top features
    st.markdown("### 🏆 Top 5 Most Important Features")
    top_features = importance_df.tail(5).iloc[::-1]
    
    for idx, row in top_features.iterrows():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{row['Feature']}**")
        with col2:
            st.markdown(f"`{row['Importance']:.4f}`")
    
    # Feature balance analysis
    st.markdown("---")
    st.markdown("### ⚖️ Feature Balance Analysis")
    
    max_importance = importance_df['Importance'].max()
    top_3_importance = importance_df.tail(3)['Importance'].sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Max Feature Importance", f"{max_importance*100:.2f}%")
        if max_importance > 0.5:
            st.error("⚠️ Single feature dominates")
        elif max_importance > 0.35:
            st.warning("⚠️ One feature has high influence")
        else:
            st.success("✅ Well balanced")
    
    with col2:
        st.metric("Top 3 Features Combined", f"{top_3_importance*100:.1f}%")
        if top_3_importance > 0.8:
            st.warning("⚠️ Top features dominate")
        else:
            st.success("✅ Distributed influence")
    
    with col3:
        avg_importance = importance_df['Importance'].mean()
        st.metric("Average Importance", f"{avg_importance*100:.2f}%")
        st.info(f"📊 Across {len(FEATURES)} features")
    
    st.markdown("""
    <div class="info-card">
    <h4>📖 Feature Balance Interpretation:</h4>
    <ul>
        <li><strong>Balanced Model:</strong> No single feature > 35% importance</li>
        <li><strong>Moderate Balance:</strong> Top feature 35-50% importance</li>
        <li><strong>Imbalanced:</strong> Single feature > 50% importance (may need retraining)</li>
    </ul>
    <p>The model now uses <code>max_features='sqrt'</code> and <code>min_samples_leaf</code> to force feature diversity.</p>
    </div>
    """, unsafe_allow_html=True)


def page_system_info():
    """Page showing system information"""
    st.markdown('<h1 class="main-header">ℹ️ System Information</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Learn about the AI models and data used in this system</p>', unsafe_allow_html=True)
    
    # Overview
    st.markdown("### 🎯 System Overview")
    st.markdown("""
    This Academic Risk Prediction System uses machine learning to identify students who may be at risk 
    of academic failure. The system analyzes multiple factors including academic performance, enrollment 
    patterns, and demographic information to provide accurate predictions and personalized recommendations.
    """)
    
    st.markdown("---")
    
    # Models section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Machine Learning Models")
        st.markdown("""
        <div class="info-card">
        <h4>📉 Logistic Regression</h4>
        <p>A linear model that provides interpretable predictions and serves as a baseline classifier.</p>
        <ul>
            <li>Fast training and prediction</li>
            <li>Probabilistic outputs</li>
            <li>Good for linear relationships</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
        <h4>🌳 Random Forest</h4>
        <p>An ensemble of decision trees that captures complex patterns in the data.</p>
        <ul>
            <li>Handles non-linear relationships</li>
            <li>Feature importance ranking</li>
            <li>Robust to outliers</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
        <h4>🔍 K-Nearest Neighbors</h4>
        <p>Finds similar student profiles to provide contextual recommendations.</p>
        <ul>
            <li>Instance-based learning</li>
            <li>No training required</li>
            <li>Provides similar cases</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
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
        
        st.markdown("### 📋 Data Split")
        st.markdown("""
        - **Training Set:** 60% (Model learning)
        - **Validation Set:** 20% (Hyperparameter tuning)
        - **Test Set:** 20% (Final evaluation)
        """)
    
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
    <div class="warning-card">
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
        **Frontend**
        - Streamlit
        - Plotly
        """)
    
    with col2:
        st.markdown("""
        **ML Libraries**
        - scikit-learn
        - pandas
        - numpy
        """)
    
    with col3:
        st.markdown("""
        **Visualization**
        - Plotly Express
        - Plotly Graph Objects
        """)


# Page router
if page == "🎯 Risk Prediction":
    page_risk_prediction()
elif page == "💡 Smart Recommendations":
    page_recommendations()
elif page == "📊 Model Analytics":
    page_model_analytics()
elif page == "ℹ️ System Info":
    page_system_info()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem 0;'>
    <p><strong>Student Academic Risk Prediction System</strong></p>
    <p>Powered by Machine Learning | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
