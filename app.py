import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import requests

st.set_page_config(page_title="ML Student Classifier", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 2.5em;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #666;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">🎓 Student Dropout Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning Model Comparison for Student Success Prediction</p>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Data Loading
st.sidebar.subheader("1️⃣ Data Loading")
data_source = st.sidebar.radio("Select data source:", ["Upload CSV", "Download from Kaggle"])

df = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    if st.sidebar.button("📥 Download Dataset from Kaggle"):
        st.sidebar.info("Note: You need to manually download from Kaggle:")
        st.sidebar.markdown("[Students Dropout Dataset](https://www.kaggle.com/datasets/mahwiz/students-dropout-and-academic-success-dataset)")

# If no dataframe yet, try to load from local
if df is None:
    try:
        df = pd.read_csv('data.csv')
        st.sidebar.success("✅ Local data.csv loaded!")
    except:
        st.sidebar.warning("Please upload a CSV file or place data.csv in the repo root")

if df is not None:
    st.sidebar.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Data Processing
    st.sidebar.subheader("2️⃣ Data Processing")
    
    # Display original structure
    with st.expander("📋 View Original Data"):
        st.write("**Original Data Shape:**", df.shape)
        st.write("**First rows:**")
        st.dataframe(df.head())
        st.write("**Column names:**", df.columns.tolist())
    
    # Move Target to first column
    if 'Target' in df.columns:
        target_col = df.pop('Target')
        df.insert(0, 'Target', target_col)
        st.sidebar.success("✅ Target moved to first column")
    else:
        st.sidebar.error("⚠️ 'Target' column not found!")
        st.stop()
    
    # Remove 'Enrolled' instances
    initial_rows = len(df)
    df = df[df['Target'] != 'Enrolled']
    removed_rows = initial_rows - len(df)
    st.sidebar.info(f"🗑️ Removed {removed_rows} 'Enrolled' instances")
    st.sidebar.info(f"📊 Remaining: {len(df)} rows")
    
    # Encode Target variable
    target_mapping = {'Graduate': 1, 'Dropout': 0}
    df['Target'] = df['Target'].map(target_mapping)
    df = df.dropna(subset=['Target'])
    
    # Encode categorical variables
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    st.sidebar.success("✅ Data preprocessing completed")
    
    # Display processed data
    with st.expander("📊 View Processed Data"):
        st.write("**Processed Data Shape:**", df.shape)
        st.dataframe(df.head())
        st.write("**Data Info:**")
        st.write(df.describe())
    
    # Data Split Configuration
    st.sidebar.subheader("3️⃣ Train/Validation/Test Split")
    train_pct = st.sidebar.slider("Training Data %", min_value=50, max_value=85, value=70, step=5)
    val_pct = st.sidebar.slider("Validation Data %", min_value=5, max_value=25, value=15, step=5)
    test_pct = 100 - train_pct - val_pct
    
    st.sidebar.info(f"📈 Train: {train_pct}% | Validation: {val_pct}% | Test: {test_pct}%")
    
    # Prepare data
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # First split: train + temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(100-train_pct)/100, random_state=42, stratify=y
    )
    
    # Second split: validation and test
    val_ratio = val_pct / (val_pct + test_pct)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-val_ratio, random_state=42, stratify=y_temp
    )
    
    st.sidebar.success(f"✅ Data split completed")
    st.sidebar.metric("Training samples", len(X_train))
    st.sidebar.metric("Validation samples", len(X_val))
    st.sidebar.metric("Test samples", len(X_test))
    
    # Display class distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Class Distribution")
        class_dist = y.value_counts()
        fig_dist = go.Figure(data=[
            go.Bar(
                x=['Dropout', 'Graduate'],
                y=[class_dist[0], class_dist[1]],
                marker_color=['#ef553b', '#00cc96']
            )
        ])
        fig_dist.update_layout(title="Target Variable Distribution", xaxis_title="Class", yaxis_title="Count")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.subheader("📈 Data Split Visualization")
        split_data = [len(X_train), len(X_val), len(X_test)]
        fig_split = go.Figure(data=[
            go.Pie(labels=['Train', 'Validation', 'Test'], values=split_data,
                   marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ])
        st.plotly_chart(fig_split, use_container_width=True)
    
    # Model Selection and Training
    st.sidebar.subheader("4️⃣ Model Selection")
    models_to_train = st.sidebar.multiselect(
        "Select models to train:",
        ["Logistic Regression", "Decision Tree", "SVM", "Random Forest", "MLP", "Naive Bayes"],
        default=["Logistic Regression", "Decision Tree", "SVM", "Random Forest", "MLP", "Naive Bayes"]
    )
    
    if st.sidebar.button("🚀 Train Models", key="train_button"):
        st.info("🔄 Training models... This may take a moment.")
        
        # Define models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "SVM": SVC(kernel='rbf', random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            "Naive Bayes": GaussianNB()
        }
        
        # Train selected models
        trained_models = {}
        results = []
        
        for model_name in models_to_train:
            with st.spinner(f"Training {model_name}..."):
                model = models[model_name]
                model.fit(X_train, y_train)
                trained_models[model_name] = model
                
                # Predictions on all sets
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_acc = accuracy_score(y_train, y_train_pred)
                val_acc = accuracy_score(y_val, y_val_pred)
                test_acc = accuracy_score(y_test, y_test_pred)
                
                test_prec = precision_score(y_test, y_test_pred, average='weighted')
                test_rec = recall_score(y_test, y_test_pred, average='weighted')
                
                results.append({
                    'Model': model_name,
                    'Train Accuracy': train_acc,
                    'Val Accuracy': val_acc,
                    'Test Accuracy': test_acc,
                    'Precision': test_prec,
                    'Recall': test_rec,
                    'y_test_pred': y_test_pred
                })
        
        # Store results in session state
        st.session_state.results = results
        st.session_state.trained_models = trained_models
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.success("✅ Training completed!")
    
    # Display Results
    if 'results' in st.session_state and st.session_state.results:
        st.header("📊 Model Comparison Results")
        
        results_df = pd.DataFrame(st.session_state.results)
        results_df = results_df.drop('y_test_pred', axis=1)
        
        # Results table
        st.subheader("📋 Metrics Summary")
        st.dataframe(
            results_df.style.format("{:.4f}").highlight_max(
                subset=['Test Accuracy', 'Precision', 'Recall'], color='lightgreen'
            ),
            use_container_width=True
        )
        
        # Comparison Charts
        col1, col2, col3 = st.columns(3)
        
        # Accuracy Comparison
        with col1:
            st.subheader("🏆 Accuracy")
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Bar(
                x=results_df['Model'],
                y=results_df['Test Accuracy'],
                name='Test Accuracy',
                marker_color='#1f77b4'
            ))
            fig_acc.update_layout(yaxis_range=[0, 1], height=400)
            st.plotly_chart(fig_acc, use_container_width=True)
        
        # Precision Comparison
        with col2:
            st.subheader("📊 Precision")
            fig_prec = go.Figure()
            fig_prec.add_trace(go.Bar(
                x=results_df['Model'],
                y=results_df['Precision'],
                name='Precision',
                marker_color='#ff7f0e'
            ))
            fig_prec.update_layout(yaxis_range=[0, 1], height=400)
            st.plotly_chart(fig_prec, use_container_width=True)
        
        # Recall Comparison
        with col3:
            st.subheader("📈 Recall (Exhaustividad)")
            fig_rec = go.Figure()
            fig_rec.add_trace(go.Bar(
                x=results_df['Model'],
                y=results_df['Recall'],
                name='Recall',
                marker_color='#2ca02c'
            ))
            fig_rec.update_layout(yaxis_range=[0, 1], height=400)
            st.plotly_chart(fig_rec, use_container_width=True)
        
        # Train vs Val vs Test Accuracy
        st.subheader("📈 Training Performance: Train vs Validation vs Test")
        fig_train_val = go.Figure()
        
        fig_train_val.add_trace(go.Bar(
            x=results_df['Model'],
            y=results_df['Train Accuracy'],
            name='Train Accuracy',
            marker_color='#1f77b4'
        ))
        fig_train_val.add_trace(go.Bar(
            x=results_df['Model'],
            y=results_df['Val Accuracy'],
            name='Validation Accuracy',
            marker_color='#ff7f0e'
        ))
        fig_train_val.add_trace(go.Bar(
            x=results_df['Model'],
            y=results_df['Test Accuracy'],
            name='Test Accuracy',
            marker_color='#2ca02c'
        ))
        
        fig_train_val.update_layout(barmode='group', height=400, yaxis_range=[0, 1])
        st.plotly_chart(fig_train_val, use_container_width=True)
        
        # Radar chart for overall comparison
        st.subheader("🏆 Overall Model Comparison (Radar Chart)")
        fig_radar = go.Figure()
        
        for idx, row in results_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Test Accuracy'], row['Precision'], row['Recall']],
                theta=['Accuracy', 'Precision', 'Recall'],
                fill='toself',
                name=row['Model']
            ))
        
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=500)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Detailed Analysis per Model
        st.header("🔍 Detailed Model Analysis")
        
        selected_model = st.selectbox("Select model for detailed analysis:", results_df['Model'].tolist())
        
        model_idx = results_df[results_df['Model'] == selected_model].index[0]
        y_test_pred = st.session_state.results[model_idx]['y_test_pred']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 Confusion Matrix - {selected_model}")
            cm = confusion_matrix(st.session_state.y_test, y_test_pred)
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="True"),
                x=['Dropout', 'Graduate'],
                y=['Dropout', 'Graduate'],
                color_continuous_scale='Blues',
                text_auto=True
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.subheader(f"📈 Classification Report - {selected_model}")
            report = classification_report(
                st.session_state.y_test,
                y_test_pred,
                target_names=['Dropout', 'Graduate'],
                output_dict=True
            )
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
        
        # Model Recommendations
        st.header("💡 Analysis & Recommendations")
        
        best_acc = results_df.loc[results_df['Test Accuracy'].idxmax()]
        best_prec = results_df.loc[results_df['Precision'].idxmax()]
        best_rec = results_df.loc[results_df['Recall'].idxmax()]
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            st.metric("🏆 Best Accuracy", best_acc['Model'], f"{best_acc['Test Accuracy']:.4f}")
        with rec_col2:
            st.metric("🏆 Best Precision", best_prec['Model'], f"{best_prec['Precision']:.4f}")
        with rec_col3:
            st.metric("🏆 Best Recall", best_rec['Model'], f"{best_rec['Recall']:.4f}")
        
        st.subheader("📝 Key Insights:")
        insights = f"""
        ### Model Performance Analysis
        
        **Overall Best Model:** {best_acc['Model']}
        - Achieved {best_acc['Test Accuracy']:.2%} accuracy on test data
        - {best_acc['Precision']:.2%} precision and {best_acc['Recall']:.2%} recall
        
        **Precision Leader:** {best_prec['Model']}
        - False positive rate: Very low ({best_prec['Precision']:.2%})
        - Best for minimizing incorrect positive predictions
        - **Use case:** When incorrectly marking someone as likely to graduate has high cost
        
        **Recall Leader:** {best_rec['Model']}
        - Successfully identifies {best_rec['Recall']:.2%} of actual dropouts
        - Best for catching potential dropouts
        - **Use case:** Early intervention programs
        
        ### Model Interpretations:
        
        **Accuracy (Exactitud):** Overall correctness of the model
        - Measures: (True Positives + True Negatives) / Total Predictions
        - Best for: Balanced datasets with equal class importance
        
        **Precision (Precisión):** Reliability of positive predictions
        - Measures: True Positives / (True Positives + False Positives)
        - Best for: Minimizing false alarms (e.g., intervention resources)
        
        **Recall (Exhaustividad):** Ability to find all positive cases
        - Measures: True Positives / (True Positives + False Negatives)
        - Best for: Ensuring all at-risk students are identified
        
        ### Recommendations:
        
        1. **For Early Intervention:** Use {best_rec['Model']} (highest recall)
           - Captures more potential dropouts for proactive support
        
        2. **For Resource Allocation:** Use {best_prec['Model']} (highest precision)
           - Minimizes wasted resources on false positives
        
        3. **For Overall Performance:** Use {best_acc['Model']} (highest accuracy)
           - Best general-purpose model for deployment
        
        4. **Overfitting Check:** 
           - Models with Train Accuracy >> Test Accuracy may overfit
           - Consider using simpler models or regularization
        """
        st.markdown(insights)
        
        # Export Results
        st.subheader("💾 Export Results")
        csv_results = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv_results,
            file_name="model_results.csv",
            mime="text/csv"
        )
        
else:
    st.warning("⚠️ Please upload or load a CSV file to start.")
    st.info("""
    ### Getting Started:
    
    1. **Upload your CSV** or place `data.csv` in the repository root
    2. **Data will be automatically processed:**
       - Target column moved to first position
       - 'Enrolled' instances removed
       - Categorical variables encoded
    3. **Configure data split ratios** (Train/Validation/Test)
    4. **Select models** to compare
    5. **Click 'Train Models'** to run the analysis
    6. **View results** with interactive visualizations
    
    ### Dataset Source:
    [Students Dropout and Academic Success Dataset](https://www.kaggle.com/datasets/mahwiz/students-dropout-and-academic-success-dataset)
    """)
