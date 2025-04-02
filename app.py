# Import necessary libraries
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .prediction-result {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 20px 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .sidebar .stSlider {
        padding-top: 1rem;
    }
    .sidebar .stSelectbox {
        padding-top: 1rem;
    }
    footer {
        text-align: center;
        padding: 20px;
        font-size: 0.8rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_models():
    ridge_model = joblib.load('models/best_ridge_model.pkl')
    rf_model = joblib.load('models/best_rf_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return ridge_model, rf_model, scaler

ridge_model, rf_model, scaler = load_models()

# Sidebar
with st.sidebar:
    st.image("https://www.example.com/edu_logo.png", width=100)  # Replace with your logo URL
    st.markdown("<h2 style='text-align: center;'>Student Parameters</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Study Habits")
    hours_studied = st.slider("Hours Studied Per Day", 0, 24, 8, 
                             help="Average number of hours the student studies daily")
    sample_papers_practiced = st.slider("Sample Question Papers Practiced", 0, 100, 10,
                                      help="Number of practice papers completed")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Performance History")
    previous_scores = st.slider("Previous Scores", 0, 100, 50,
                              help="Student's score in previous assessments")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Lifestyle Factors")
    sleep_hours = st.slider("Sleep Hours Per Night", 0, 12, 8, 
                          help="Average hours of sleep per night")
    extracurricular = st.selectbox("Extracurricular Activities", 
                                 ("Yes", "No"),
                                 help="Whether the student participates in extracurricular activities")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Model Selection")
    model_choice = st.radio("Choose Prediction Model", 
                         ("Ridge Regression", "Random Forest"),
                         help="Select which machine learning model to use for prediction")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # About section
    with st.expander("About This App"):
        st.write("""
        This application uses machine learning to predict student performance based on various factors.
        The models have been trained on historical student data and can help identify potential areas for improvement.
        """)

# Main content
st.markdown("<h1 class='main-header'>📚 Student Performance Prediction</h1>", unsafe_allow_html=True)

# Introduction tabs
tab1, tab2 = st.tabs(["Prediction", "Model Comparison"])

with tab1:
    # Process the input data
    extracurricular_numeric = 1 if extracurricular == "Yes" else 0
    input_data = np.array([[hours_studied, previous_scores, extracurricular_numeric, sleep_hours, sample_papers_practiced]])
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction based on the selected model
    if model_choice == "Ridge Regression":
        prediction = ridge_model.predict(input_data_scaled)
        model_confidence = 0.989  # R² value for Ridge
    else:
        prediction = rf_model.predict(input_data_scaled)
        model_confidence = 0.987  # R² value for Random Forest
    
    # Show input summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Study Metrics")
        st.metric("Hours Studied", f"{hours_studied} hrs")
        st.metric("Practice Papers", sample_papers_practiced)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Academic History")
        st.metric("Previous Score", f"{previous_scores}/100")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Lifestyle")
        st.metric("Sleep", f"{sleep_hours} hrs")
        st.metric("Extracurricular", extracurricular)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display the prediction result
    st.markdown("<div class='prediction-result'>", unsafe_allow_html=True)
    st.markdown("<h2>Predicted Performance Index</h2>", unsafe_allow_html=True)
    st.markdown(f"<p class='prediction-value'>{prediction[0]:.2f}</p>", unsafe_allow_html=True)
    
    # Add a gauge chart for the prediction
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction[0],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Performance Index"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1E88E5"},
            'steps': [
                {'range': [0, 40], 'color': "#EF5350"},
                {'range': [40, 70], 'color': "#FFCA28"},
                {'range': [70, 100], 'color': "#66BB6A"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prediction[0]
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"<p>Prediction made using {model_choice} model (R² = {model_confidence:.3f})</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Interpretation of the result
    st.markdown("<h3 class='sub-header'>What does this mean?</h3>", unsafe_allow_html=True)
    
    performance_level = ""
    if prediction[0] < 40:
        performance_level = "needs improvement"
        suggestions = ["Increase study hours", "Practice more sample papers", "Establish a regular sleep schedule"]
    elif prediction[0] < 70:
        performance_level = "satisfactory"
        suggestions = ["Slightly increase study hours", "Continue with current sleep schedule", "Consider more practice papers"]
    else:
        performance_level = "excellent"
        suggestions = ["Maintain current study habits", "Balance academics with extracurriculars", "Continue with consistent sleep schedule"]
    
    st.markdown(f"<div class='card'>The predicted performance is <strong>{performance_level}</strong>. Here are some suggestions:", unsafe_allow_html=True)
    for suggestion in suggestions:
        st.markdown(f"- {suggestion}")
    st.markdown("</div>", unsafe_allow_html=True)
    

with tab2:
    st.markdown("<h3 class='sub-header'>Model Performance Comparison</h3>", unsafe_allow_html=True)
    
    # Model comparison data
    model_comparison = {
        "Model": ["Linear Regression", "Random Forest", "Ridge Regression"],
        "MAE": [1.612, 1.721, 1.612],
        "MSE": [4.087, 4.672, 4.089],
        "R²": [0.989, 0.987, 0.989]
    }
    
    model_comparison_df = pd.DataFrame(model_comparison)
    
    # Show styled table
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.dataframe(model_comparison_df.style.highlight_max(axis=0, subset=['R²']).highlight_min(axis=0, subset=['MAE', 'MSE']))
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualize model comparison
    fig = go.Figure()
    
    # Add model metrics
    for model in model_comparison_df['Model']:
        model_data = model_comparison_df[model_comparison_df['Model'] == model]
        fig.add_trace(go.Bar(
            name=model,
            x=['MAE', 'MSE', 'R²'],
            y=[model_data['MAE'].values[0], model_data['MSE'].values[0], model_data['R²'].values[0]],
            marker_color=['#f44336', '#f44336', '#4caf50'] if model == model_choice else ['#90caf9', '#90caf9', '#90caf9']
        ))
    
    fig.update_layout(
        title='Model Metrics Comparison',
        xaxis_title='Metric',
        yaxis_title='Value',
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample prediction comparison
    models = ["Ridge Regression", "Random Forest"]
    ridge_pred = ridge_model.predict(input_data_scaled)[0]
    rf_pred = rf_model.predict(input_data_scaled)[0]
    predictions = [ridge_pred, rf_pred]
    
    fig = px.bar(
        x=models,
        y=predictions,
        color=predictions,
        color_continuous_scale=px.colors.sequential.Blues,
        labels={'x': 'Model', 'y': 'Predicted Performance Index'},
        title='Current Prediction Comparison',
        text=[f"{p:.2f}" for p in predictions]
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Explain the difference
    diff = abs(ridge_pred - rf_pred)
    st.markdown(f"<div class='card'>The difference between model predictions is {diff:.2f} points. This is due to how each model learns from the data:</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Ridge Regression")
        st.markdown("A linear model that reduces overfitting by penalizing large coefficients.")
        st.markdown("**Best for**: Understanding straightforward relationships between factors.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Random Forest")
        st.markdown("An ensemble model that uses multiple decision trees to make predictions.")
        st.markdown("**Best for**: Capturing complex, non-linear relationships in the data.")
        st.markdown("</div>", unsafe_allow_html=True)

# Add a what-if analysis section
st.markdown("<h3 class='sub-header'>What-If Analysis</h3>", unsafe_allow_html=True)
st.markdown("<div class='card'>Explore how changing different factors would affect the predicted performance:</div>", unsafe_allow_html=True)

what_if_col1, what_if_col2 = st.columns(2)

with what_if_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    # Create what-if scenarios for study hours
    what_if_hours = np.arange(max(1, hours_studied - 3), min(24, hours_studied + 4))
    what_if_predictions = []
    
    for h in what_if_hours:
        what_if_data = np.array([[h, previous_scores, extracurricular_numeric, sleep_hours, sample_papers_practiced]])
        what_if_data_scaled = scaler.transform(what_if_data)
        if model_choice == "Ridge Regression":
            what_if_predictions.append(ridge_model.predict(what_if_data_scaled)[0])
        else:
            what_if_predictions.append(rf_model.predict(what_if_data_scaled)[0])
    
    # Plot what-if analysis for study hours
    fig = px.line(
        x=what_if_hours,
        y=what_if_predictions,
        markers=True,
        labels={'x': 'Study Hours', 'y': 'Predicted Performance'},
        title='Effect of Changing Study Hours'
    )
    
    # Add current point
    fig.add_trace(
        go.Scatter(
            x=[hours_studied],
            y=[prediction[0]],
            mode='markers',
            marker=dict(color='red', size=12),
            name='Current'
        )
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with what_if_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    # Create what-if scenarios for practice papers
    what_if_papers = np.arange(max(0, sample_papers_practiced - 5), min(100, sample_papers_practiced + 6), 1)
    what_if_predictions = []
    
    for p in what_if_papers:
        what_if_data = np.array([[hours_studied, previous_scores, extracurricular_numeric, sleep_hours, p]])
        what_if_data_scaled = scaler.transform(what_if_data)
        if model_choice == "Ridge Regression":
            what_if_predictions.append(ridge_model.predict(what_if_data_scaled)[0])
        else:
            what_if_predictions.append(rf_model.predict(what_if_data_scaled)[0])
    
    # Plot what-if analysis for practice papers
    fig = px.line(
        x=what_if_papers,
        y=what_if_predictions,
        markers=True,
        labels={'x': 'Practice Papers', 'y': 'Predicted Performance'},
        title='Effect of Practice Papers'
    )
    
    # Add current point
    fig.add_trace(
        go.Scatter(
            x=[sample_papers_practiced],
            y=[prediction[0]],
            mode='markers',
            marker=dict(color='red', size=12),
            name='Current'
        )
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Recommendation summary
st.markdown("<h3 class='sub-header'>Personalized Recommendations</h3>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
# Calculate optimal parameters
optimal_hours = hours_studied
if hours_studied < 8:
    optimal_hours = min(hours_studied + 2, 8)
elif hours_studied > 10:
    optimal_hours = 9  # Reduce if studying too much

optimal_sleep = sleep_hours
if sleep_hours < 7:
    optimal_sleep = 7.5
elif sleep_hours > 9:
    optimal_sleep = 8

optimal_papers = min(sample_papers_practiced + 5, 20)

st.markdown("### Optimize Your Performance")
st.markdown(f"""
Based on our analysis, here are personalized recommendations to improve your performance:

1. **Study Duration**: Target around {optimal_hours} hours of focused study per day
2. **Practice Tests**: Increase practice papers to at least {optimal_papers}
3. **Rest & Health**: Maintain {optimal_sleep} hours of sleep for optimal cognitive function
4. **Study Quality**: Focus on understanding concepts rather than memorization
""")
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<footer>Developed for educational purposes. This is a predictive model and results should be used as guidance only.</footer>", unsafe_allow_html=True)