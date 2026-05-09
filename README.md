# 🎓 Student Dropout Classifier - ML Application

A comprehensive Machine Learning application for predicting student graduation vs. dropout using multiple algorithms and comparing their performance.

## 📋 Project Overview

This application uses 6 different machine learning models from scikit-learn to classify students as either **Graduates** or **Dropouts** based on educational attributes.

### Dataset
- **Source:** [Students Dropout and Academic Success Dataset (Kaggle)](https://www.kaggle.com/datasets/mahwiz/students-dropout-and-academic-success-dataset)
- **Format:** CSV
- **Processing:** 
  - Target variable moved to first column
  - 'Enrolled' instances removed
  - Categorical variables automatically encoded

## 🤖 Machine Learning Models

The application compares 6 algorithms:

1. **Logistic Regression**
   - Linear classification model
   - Fast and interpretable
   - Good baseline model

2. **Decision Tree**
   - Tree-based classification
   - Easy to visualize and interpret
   - Prone to overfitting

3. **Support Vector Machine (SVM)**
   - Finds optimal hyperplane
   - Effective in high-dimensional spaces
   - Good for binary classification

4. **Random Forest**
   - Ensemble of decision trees
   - Reduces overfitting
   - Generally robust performance

5. **Multilayer Perceptron (MLP)**
   - Neural network classifier
   - Captures complex non-linear patterns
   - Requires more computational resources

6. **Naive Bayes**
   - Probabilistic classifier
   - Fast and lightweight
   - Good baseline for comparison

## 📊 Performance Metrics

The application evaluates models on three key metrics:

### 1. **Accuracy (Exactitud)**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- Measures overall correctness
- Best for balanced datasets
- Scale: 0 to 1 (higher is better)

### 2. **Precision (Precisión)**
```
Precision = TP / (TP + FP)
```
- Reliability of positive predictions
- Minimizes false positives
- Scale: 0 to 1 (higher is better)
- **Use when:** Cost of false alarms is high

### 3. **Recall (Exhaustividad)**
```
Recall = TP / (TP + FN)
```
- Ability to find all positive cases
- Minimizes false negatives
- Scale: 0 to 1 (higher is better)
- **Use when:** Missing positive cases is costly

### Legend:
- **TP (True Positive):** Correctly predicted dropout/graduate
- **TN (True Negative):** Correctly predicted opposite class
- **FP (False Positive):** Incorrectly predicted as positive
- **FN (False Negative):** Incorrectly predicted as negative

## 🚀 Deployment on Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at [streamlit.io/cloud](https://streamlit.io/cloud))
- This repository pushed to GitHub

### Deployment Steps

#### 1. **Prepare Repository**
```bash
# Clone or navigate to your repository
cd headofdyes/proba

# Create and checkout the branch
git checkout -b ml-students-classifier

# Add files
git add .
git commit -m "Initial ML app commit"
git push origin ml-students-classifier
```

#### 2. **Deploy on Streamlit Cloud**
1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Click "New app"
3. Fill in the deployment form:
   - **Repository:** `headofdyes/proba`
   - **Branch:** `ml-students-classifier`
   - **Main file path:** `app.py`
4. Click "Deploy"
5. Wait 2-3 minutes for deployment to complete

#### 3. **Access Your App**
- Your app will be available at: `https://share.streamlit.io/headofdyes/proba/ml-students-classifier/app.py`
- Share the URL with others

### Environment Configuration
- Streamlit automatically installs dependencies from `requirements.txt`
- Configuration is loaded from `.streamlit/config.toml`

## 🎯 How to Use the Application

### Step 1: Load Data
- **Option A:** Upload a CSV file from your computer
- **Option B:** Download the dataset manually from Kaggle
- **Option C:** Place `data.csv` in the repository root

### Step 2: Review Processed Data
- Expand "View Processed Data" section
- Verify Target variable is in first column
- Check that categorical variables are encoded

### Step 3: Configure Data Split
- **Training Data %:** Set percentage for model training (50-85%)
- **Validation Data %:** Set percentage for validation (5-25%)
- **Test Data %:** Automatically calculated as remainder
- Recommended: Train 70%, Validation 15%, Test 15%

### Step 4: Select Models
- Check all models you want to compare
- Or use default (all 6 models)
- Uncheck to skip slower models

### Step 5: Train Models
- Click "🚀 Train Models" button
- Wait for training to complete
- Monitor progress in the status message

### Step 6: Analyze Results
- View summary metrics table
- Compare 3 metrics across all models
- Examine train/validation/test accuracy
- Study radar chart for overall comparison

### Step 7: Detailed Analysis
- Select a specific model for deep dive
- View confusion matrix
- Read detailed classification report
- Understand per-class performance

### Step 8: Review Recommendations
- Read AI-generated analysis
- Understand metric interpretations
- Get model selection guidance
- Export results as CSV

## 📈 Visualizations

The application provides interactive charts:

1. **Class Distribution** - Shows balance of Dropout vs Graduate
2. **Data Split Visualization** - Pie chart of train/val/test split
3. **Accuracy Comparison** - Bar chart of accuracy per model
4. **Precision Comparison** - Bar chart of precision per model
5. **Recall Comparison** - Bar chart of recall per model
6. **Train vs Validation vs Test** - Grouped bar chart showing performance across datasets
7. **Radar Chart** - Overall model comparison on 3 metrics
8. **Confusion Matrix** - Detailed predictions for selected model
9. **Classification Report** - Per-class metrics

## 🔧 Installation (Local Development)

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/headofdyes/proba.git
cd proba

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and save as data.csv
# Then run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
proba/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## 💡 Recommendations by Use Case

### Early Intervention Programs
**Objective:** Identify all at-risk students for support
**Metric Priority:** Recall > Accuracy > Precision
**Recommendation:** Choose model with highest Recall
**Reasoning:** Better to help false positives than miss actual dropouts

### Resource Allocation
**Objective:** Efficiently target limited intervention resources
**Metric Priority:** Precision > Accuracy > Recall
**Recommendation:** Choose model with highest Precision
**Reasoning:** Only flag students most likely to drop out

### General Deployment
**Objective:** Balanced, reliable predictions
**Metric Priority:** Accuracy > Precision ≥ Recall
**Recommendation:** Choose model with highest Accuracy
**Reasoning:** Maximizes overall correctness

### Risk Management
**Objective:** Minimize both false positives and false negatives
**Metric Priority:** F1-Score (harmonic mean of Precision and Recall)
**Recommendation:** Balance precision and recall
**Reasoning:** No single metric perfectly captures both concerns

## 🔍 Interpreting Results

### Overfitting Detection
```
If: Train Accuracy (95%) >> Test Accuracy (75%)
Then: Model is overfitting
Action: Use simpler model or add regularization
```

### Underfitting Detection
```
If: Both Train and Test Accuracy < 65%
Then: Model is underfitting
Action: Use more complex model or more features
```

### Imbalanced Results
```
If: One class has high recall but low precision
Then: Model biased toward that class
Action: Adjust class weights or use different threshold
```

## 📊 Data Processing Details

### 1. Target Variable Reorganization
- Moves 'Target' column to position 0 (first)
- Enables consistent data handling

### 2. Removing 'Enrolled' Instances
- Keeps only 'Graduate' and 'Dropout' records
- Binary classification problem
- Creates balanced training scenario

### 3. Encoding
- **Target:** 'Dropout' → 0, 'Graduate' → 1
- **Categorical Features:** Label encoding (0, 1, 2, ...)
- **Numeric Features:** Left unchanged

### 4. Stratified Split
- Maintains class distribution in all splits
- Prevents data leakage
- Ensures representative samples

## ⚙️ Configuration Options

### app.py Configuration
- `random_state=42` - Ensures reproducibility
- `max_iter=1000` - Logistic Regression iterations
- `hidden_layer_sizes=(100, 50)` - MLP architecture
- `n_estimators=100` - Random Forest trees

### Streamlit Configuration (config.toml)
- Custom color scheme
- Theme settings
- Upload/message size limits
- Server headless mode for cloud deployment

## 🐛 Troubleshooting

### Dataset Not Loading
**Problem:** "Target column not found"
**Solution:** 
1. Verify dataset has 'Target' column
2. Check CSV file is not corrupted
3. Ensure correct file format

### Training Takes Too Long
**Problem:** SVM or MLP models are slow
**Solution:**
1. Uncheck slower models
2. Reduce dataset size
3. Adjust model hyperparameters

### Memory Issues
**Problem:** App crashes with large datasets
**Solution:**
1. Sample dataset before upload
2. Reduce number of features
3. Use simpler models

### Streamlit Cloud Deploy Fails
**Problem:** "Missing dependencies"
**Solution:**
1. Verify requirements.txt is in repo root
2. Check for typos in package names
3. Ensure compatible versions

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Plotly Charts](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/mahwiz/students-dropout-and-academic-success-dataset)

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add new models
- Enhance visualizations

## 👨‍💻 Author

**headofdyes**
- GitHub: [@headofdyes](https://github.com/headofdyes)

## ✨ Features

✅ 6 Machine Learning Models
✅ 3 Performance Metrics
✅ Interactive Visualizations
✅ Flexible Data Split Configuration
✅ Detailed Analysis Reports
✅ AI-Powered Recommendations
✅ Export Results to CSV
✅ One-Click Streamlit Deployment
✅ Fully Responsive Design
✅ Production-Ready Code

---

**Happy Predicting! 🚀**
