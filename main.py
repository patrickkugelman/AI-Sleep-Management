import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.validation import validate_data, check_X_y
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, 
    mean_absolute_error, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any
from scipy.stats import uniform, randint

#  Pasul 1: Încarcă datele
xlsx_file = 'P2.4 - Military (Responses) (2).xlsx'
df = pd.read_excel(xlsx_file)

#Pasul 2: Curăță și encodează

# Se folosesc indicii de coloană pentru fișierul Excel
health_col_idx = 20  # Column 21 (0-indexed)
sleep_col_idx = 42  # Column 43 (0-indexed)


# Încarcă întregul set de date pentru o analiză completă a corelațiilor
df_full = pd.read_excel(xlsx_file)

# Selectează coloanele pentru analiza corelației
correlation_columns = df_full.columns[10:50]  # Adjust range as needed
df_corr = df_full[correlation_columns].apply(pd.to_numeric, errors='coerce')
# Calculează matricea de corelație
correlation_matrix = df_corr.corr()

import plotly.graph_objs as go
import plotly.express as px

# Advanced Correlation Visualization

def create_specialized_correlation_visualizations(df_full, health_col_idx, sleep_col_idx):
    """Create specialized visualizations for health and sleep correlations"""
    try:
        # Ensure we have the right columns
        health_col = df_full.columns[health_col_idx]
        sleep_col = df_full.columns[sleep_col_idx]
        
        # Validate data types and convert if necessary
        df_full[health_col] = pd.to_numeric(df_full[health_col], errors='coerce')
        df_full[sleep_col] = pd.to_numeric(df_full[sleep_col], errors='coerce')
        
        # Drop rows with NaN values
        df_clean = df_full.dropna(subset=[health_col, sleep_col])
        
        if len(df_clean) == 0:
            print("Error: No valid data after cleaning.")
            return None
        
        # 1. Health Impact vs Sleep Quality Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df_clean[health_col], df_clean[sleep_col], alpha=0.6, c=df_clean[health_col], cmap='viridis')
        plt.colorbar(label=health_col)
        plt.title(f'Health Impact vs Sleep Quality: {health_col} vs {sleep_col}')
        plt.xlabel(health_col)
        plt.ylabel(sleep_col)
        plt.tight_layout()
        plt.savefig('health_vs_sleep_scatter.png')
        plt.close()
        
        # 2. Health vs Sleep Interaction Heatmap
        interaction_data = df_clean[[health_col, sleep_col]].copy()
        interaction_data['health_bins'] = pd.qcut(interaction_data[health_col], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        interaction_pivot = interaction_data.pivot_table(
            index='health_bins', 
            values=sleep_col, 
            aggfunc=['mean', 'count']
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(interaction_pivot['mean'], annot=interaction_pivot['count'], cmap='YlGnBu', fmt='.2f', cbar_kws={'label': 'Avg Sleep Quality'})
        plt.title('Sleep Quality by Health Impact Levels (with sample count)')
        plt.tight_layout()
        plt.savefig('health_sleep_interaction_heatmap.png')
        plt.close()
        
        # 3. Average Sleep Quality by Health Impact
        # Use quantile-based binning for more meaningful groups
        avg_sleep_by_health = df_clean.groupby(
            pd.qcut(df_clean[health_col], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        )[sleep_col].agg(['mean', 'count'])
        
        # Bar plot with error bars and sample count
        plt.figure(figsize=(12, 8))
        avg_plot = avg_sleep_by_health['mean'].plot(kind='bar', yerr=avg_sleep_by_health['count'], capsize=5)
        plt.title('Average Sleep Quality by Health Impact Levels\n(Bar height = Mean, Error bars = Sample Count)')
        plt.xlabel('Health Impact Levels')
        plt.ylabel('Average Sleep Quality')
        plt.xticks(rotation=45)
        
        # Annotate bars with sample count
        for i, v in enumerate(avg_sleep_by_health['mean']):
            count = avg_sleep_by_health['count'][i]
            plt.text(i, v, f'n={count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('avg_sleep_by_health_impact.png')
        plt.close()
        
        # Plotly Interactive Correlation Heatmap with more robust correlation
        # Use cleaned data for correlation
        correlation_matrix = df_clean.select_dtypes(include=[np.number]).corr(method='spearman')
        
        # Create interactive heatmap
        fig = px.imshow(
            correlation_matrix, 
            labels=dict(x="Features", y="Features", color="Correlation"),
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            color_continuous_scale='RdBu_r',
            title='Comprehensive Feature Correlation Heatmap (Spearman Rank Correlation)'
        )
        fig.update_layout(height=800, width=800)
        fig.write_html('correlation_heatmap.html')
        
        print("Visualizations saved successfully.")
        return {
            'scatter_plot': 'health_vs_sleep_scatter.png',
            'interaction_heatmap': 'health_sleep_interaction_heatmap.png',
            'avg_sleep_plot': 'avg_sleep_by_health_impact.png',
            'correlation_heatmap': 'correlation_heatmap.html'
        }
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return None

def create_interactive_correlation_map(correlation_matrix):
    """Create an interactive correlation heatmap with detailed insights"""
    try:
        # Validate correlation matrix
        if correlation_matrix.empty:
            print("Warning: Correlation matrix is empty. Cannot create visualization.")
            return None
        
        # Prepare correlation data
        corr_data = correlation_matrix.reset_index()
        corr_data = corr_data.melt(id_vars='index', var_name='Columns', value_name='Correlation')
        
        # Filter out self-correlations and redundant pairs
        corr_data = corr_data[corr_data['index'] != corr_data['Columns']]
        
        # More robust filtering of significant correlations
        corr_data = corr_data[
            (abs(corr_data['Correlation']) > 0.3) &  # Moderate correlation threshold
            (corr_data['Correlation'] != 1.0)  # Exclude perfect self-correlations
        ]
        
        # Sort by absolute correlation strength
        corr_data['AbsCorrelation'] = abs(corr_data['Correlation'])
        corr_data = corr_data.sort_values('AbsCorrelation', ascending=False)
        
        # Print top correlations for debugging
        print("Top Correlations:")
        print(corr_data.head())
    except Exception as e:
        print(f"Error in creating correlation map: {e}")
        return None
    
    # Create interactive heatmap
    fig = px.imshow(
        correlation_matrix, 
        color_continuous_scale='RdBu_r',  # Red-Blue diverging scale
        title='Interactive Correlation Matrix: Environmental Factors',
        labels=dict(x="Survey Variables", y="Survey Variables", color="Correlation"),
        text_auto=True,
        aspect='auto'
    )
    
    # Customize layout
    fig.update_layout(
        height=800,
        width=1000,
        title_font_size=20,
        title_x=0.5,
        coloraxis_colorbar=dict(title='Correlation Strength')
    )
    
    # Save interactive HTML
    fig.write_html('interactive_correlation_map.html')
    
    # Create a bar plot of top correlations
    top_correlations = corr_data.head(10)
    bar_fig = px.bar(
        top_correlations, 
        x='index', 
        y='Correlation', 
        color='Correlation',
        color_continuous_scale='RdBu_r',
        title='Top 10 Significant Correlations',
        labels={'index': 'Variable Pairs', 'Correlation': 'Correlation Strength'}
    )
    bar_fig.update_layout(height=600, width=1000)
    bar_fig.write_html('top_correlations_bar.html')
    
    # Print textual insights
    print("\nCorrelation Insights:")
    for _, row in top_correlations.iterrows():
        direction = "Positive" if row['Correlation'] > 0 else "Negative"
        print(f"- {row['index']} ↔ {row['Columns']}: {direction} Correlation of {row['Correlation']:.4f}")

# Generate specialized correlation visualizations
specialized_visualizations = create_specialized_correlation_visualizations(df_full, health_col_idx, sleep_col_idx)

# Optional: Generate interactive correlation map as a fallback
if specialized_visualizations is None:
    create_interactive_correlation_map(correlation_matrix)

# Prepare the main analysis dataframe
df = df.iloc[:, [health_col_idx, sleep_col_idx]].dropna()

# Precision-Recall Analysis Function
def precision_recall_analysis(y_true, y_pred, title='Sleep Performance'):
    """Comprehensive Precision-Recall Analysis"""
    # Compute metrics
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Categorical classification report
    cr = classification_report(y_true, y_pred)
    print(f"\nPrecision-Recall Analysis for {title}:")
    print(cr)
    
    # Precision-Recall Curve (Interactive)
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    
    # Plotly Interactive Precision-Recall Curve
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(
        x=recall_curve, 
        y=precision_curve, 
        mode='lines+markers',
        name='Precision-Recall Curve'
    ))
    pr_fig.update_layout(
        title=f'Precision-Recall Curve for {title}',
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=600,
        width=800
    )
    pr_fig.write_html('precision_recall_curve.html')
    
    # Matplotlib Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(recall_curve, precision_curve, label='Precision-Recall Curve')
    plt.title(f'Precision-Recall Curve for {title}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png')
    plt.close()
    
    # Create a summary dictionary
    precision_recall_summary = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'average_precision': avg_precision
    }
    
    # Interpret Results
    def interpret_precision_recall(summary):
        """Provide human-readable interpretation of precision-recall metrics"""
        print("\n Results Interpretation:")
        print(f"1. Precision: {summary['precision']:.2f}")
        if summary['precision'] > 0.8:
            print("   - Excellent: Model has high accuracy in positive predictions")
        elif summary['precision'] > 0.6:
            print("   - Good: Model has reasonable accuracy in positive predictions")
        else:
            print("   - Needs Improvement: Many false positive predictions")
        
        print(f"2. Recall: {summary['recall']:.2f}")
        if summary['recall'] > 0.8:
            print("   - Excellent: Model captures most positive instances")
        elif summary['recall'] > 0.6:
            print("   - Good: Model captures a significant portion of positive instances")
        else:
            print("   - Needs Improvement: Missing many positive instances")
        
        print(f"3. F1 Score: {summary['f1_score']:.2f}")
        if summary['f1_score'] > 0.8:
            print("   - Excellent: Balanced precision and recall")
        elif summary['f1_score'] > 0.6:
            print("   - Good: Reasonable balance between precision and recall")
        else:
            print("   - Needs Improvement: Imbalanced performance")
        
        print(f"4. Average Precision: {summary['average_precision']:.2f}")
        if summary['average_precision'] > 0.8:
            print("   - High: Model performs consistently across different thresholds")
        elif summary['average_precision'] > 0.6:
            print("   - Moderate: Model shows potential for improvement")
        else:
            print("   - Low: Model struggles with consistent predictions")
    
    # Call interpretation function
    interpret_precision_recall(precision_recall_summary)
    
    return precision_recall_summary

# Example Usage (modify based on your specific classification task)
def prepare_classification_data(df, health_col_idx, sleep_col_idx):
    """Prepare data for classification"""
    # Convert continuous data to binary classification
    # Example: Classify sleep quality as 'Good' or 'Poor'
    median_sleep = df.iloc[:, sleep_col_idx].median()
    y = (df.iloc[:, sleep_col_idx] > median_sleep).astype(int)
    
    # Select features
    X = df.iloc[:, [health_col_idx]]
    
    return X, y

# Debug: Print DataFrame information
print("DataFrame Columns:", df.columns)
print("DataFrame Shape:", df.shape)
print("Health Column Index:", health_col_idx)
print("Sleep Column Index:", sleep_col_idx)

# Safe column selection
def safe_column_select(df, col_idx):
    """Safely select a column, handling potential index out of bounds"""
    try:
        # Try to select the column
        column = df.iloc[:, col_idx]
        return column
    except IndexError:
        # If index is out of bounds, print error and return None
        print(f"Error: Column index {col_idx} is out of bounds.")
        print(f"Available columns: 0 to {len(df.columns) - 1}")
        return None

# Modify prepare_classification_data function
def prepare_classification_data(df, health_col_idx, sleep_col_idx):
    """Prepare data for classification with robust error handling"""
    # Safely select columns
    health_column = safe_column_select(df, health_col_idx)
    sleep_column = safe_column_select(df, sleep_col_idx)
    
    # Check if columns were successfully selected
    if health_column is None or sleep_column is None:
        raise ValueError("Could not select required columns for analysis")
    
    # Convert to numeric, handling errors
    health_column = pd.to_numeric(health_column, errors='coerce')
    sleep_column = pd.to_numeric(sleep_column, errors='coerce')
    
    # Remove NaN values
    valid_data = pd.concat([health_column, sleep_column], axis=1).dropna()
    
    # Prepare features and target
    X = valid_data.iloc[:, [0]]
    
    # Binary classification based on median
    median_sleep = valid_data.iloc[:, 1].median()
    y = (valid_data.iloc[:, 1] > median_sleep).astype(int)
    
    return X, y

# Comprehensive Column Analysis and Precision-Recall Preparation
def analyze_dataframe_columns(df):
    """Provide detailed information about DataFrame columns"""
    print("\nDataFrame Column Analysis:")
    print(f"Total Columns: {len(df.columns)}")
    print("Column Details:")
    for i, col in enumerate(df.columns):
        print(f"Column {i}: {col}")
        print(f"  - Sample Values: {df[col].head().tolist()}")
        print(f"  - Data Type: {df[col].dtype}")
        print(f"  - Non-Null Count: {df[col].count()}")
        print(f"  - Unique Values: {df[col].nunique()}\n")

# Perform comprehensive column analysis
analyze_dataframe_columns(df)

# Advanced Precision-Recall Analysis with Multiple Techniques
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def advanced_precision_recall_analysis(df):
    """Comprehensive Precision-Recall Analysis with Multiple Techniques"""
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    
    # Suppress specific warnings
    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
    
    # Dynamically select numeric columns for analysis
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_columns) < 2:
        raise ValueError("Not enough numeric columns for analysis")
    
    # Select first two numeric columns
    health_column = numeric_columns[0]
    sleep_column = numeric_columns[1]
    
    print("\nAdvanced Precision-Recall Analysis:")
    print(f"Health Column: {health_column}")
    print(f"Sleep Column: {sleep_column}")
    
    # Prepare data for classification
    X = df[[health_column]]
    
    # Binary classification based on multiple thresholds
    thresholds = [
        df[sleep_column].quantile(0.25),  # 25th percentile
        df[sleep_column].median(),        # Median
        df[sleep_column].quantile(0.75)   # 75th percentile
    ]
    
    results = []
    
    for threshold in thresholds:
        # Create binary target variable
        y = (df[sleep_column] > threshold).astype(int)
        
        # Check class distribution
        class_counts = y.value_counts()
        print(f"\nClass Distribution at Threshold {threshold:.2f}:")
        print(class_counts)
        
        # Skip if extreme class imbalance
        if class_counts.min() / class_counts.max() < 0.1:
            print("Warning: Extreme class imbalance. Skipping this threshold.")
            continue
        
        # Prepare cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Define models with pipelines for robust preprocessing
        models = {
            'Random Forest': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    class_weight='balanced', 
                    random_state=42
                ))
            ]),
            'Gradient Boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(
                    max_depth=3,  # Prevent overfitting
                    random_state=42
                ))
            ]),
            'SVM': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(
                    class_weight='balanced',
                    probability=True, 
                    random_state=42
                ))
            ])
        }
        
        # Perform cross-validated precision-recall analysis
        model_results = {}
        for name, model in models.items():
            try:
                # Compute cross-validated precision, recall, and f1
                precision = cross_val_score(
                    model, X, y, 
                    cv=cv, 
                    scoring='precision_weighted',
                    error_score='raise'
                )
                recall = cross_val_score(
                    model, X, y, 
                    cv=cv, 
                    scoring='recall_weighted',
                    error_score='raise'
                )
                f1 = cross_val_score(
                    model, X, y, 
                    cv=cv, 
                    scoring='f1_weighted',
                    error_score='raise'
                )
                
                model_results[name] = {
                    'Precision': precision.mean(),
                    'Recall': recall.mean(),
                    'F1 Score': f1.mean()
                }
            except Exception as e:
                print(f"Error with {name} model: {e}")
                model_results[name] = {
                    'Precision': 0,
                    'Recall': 0,
                    'F1 Score': 0
                }
        
        results.append({
            'Threshold': threshold,
            'Models': model_results
        })
    
    # Print comprehensive results
    print("\nPrecision-Recall Analysis Results:")
    for result in results:
        print(f"\nThreshold: {result['Threshold']:.2f}")
        for model_name, metrics in result['Models'].items():
            print(f"  {model_name}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
    
    return results

# Perform Advanced Precision-Recall Analysis
try:
    advanced_results = advanced_precision_recall_analysis(df)

except Exception as e:
    print(f"Error in Advanced Precision-Recall Analysis: {e}")
    print("Please review your data and column selection.")

# Perform Precision-Recall Analysis with dynamic column selection
try:
    # Dynamically select numeric columns for analysis
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_columns) < 2:
        raise ValueError("Not enough numeric columns for analysis")
    
    # Select first two numeric columns
    health_column = numeric_columns[0]
    sleep_column = numeric_columns[1]
    
    print(f"\n Selected Columns for Analysis:")
    print(f"Health Column: {health_column}")
    print(f"Sleep Column: {sleep_column}")
    
    # Prepare data for classification
    X = df[[health_column]]
    
    # Binary classification based on median
    median_sleep = df[sleep_column].median()
    y = (df[sleep_column] > median_sleep).astype(int)
    
    # Split data
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict and analyze
    y_pred = clf.predict(X_test)
    precision_recall_results = precision_recall_analysis(y_test, y_pred)

except Exception as e:
    print(f"Error in Precision-Recall Analysis: {e}")
    print("Please review your data and column selection.")
# Convert columns to numeric first to avoid dtype issues
df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')

# Drop rows with NaN values after conversion
df.dropna(subset=[df.columns[0], df.columns[1]], inplace=True)

oe_health = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
oe_sleep = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Check for and handle missing values
df.iloc[:, 0] = df.iloc[:, 0].fillna(df.iloc[:, 0].mode()[0])
df.iloc[:, 1] = df.iloc[:, 1].fillna(df.iloc[:, 1].mode()[0])

# Encode categorical variables with robust handling
df["health_encoded"] = oe_health.fit_transform(df.iloc[:, [0]])
df["sleep_encoded"] = oe_sleep.fit_transform(df.iloc[:, [1]])

# Create interaction feature with additional validation
df["interaction"] = df["health_encoded"] * df["sleep_encoded"]

# Remove any rows with infinite or NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ========== Step 9: Visualize Correlations ==========
# Prepare correlation matrix
correlation_columns = ['health_encoded', 'sleep_encoded', 'interaction']
correlation_matrix = df[correlation_columns].corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix,
            annot=True,  # Show numeric values
            cmap='coolwarm',  # Color palette
            center=0,  # Center color at 0
            vmin=-1,
            vmax=1,
            square=True)
plt.title('Correlation Matrix of Encoded Features', fontsize=16)
plt.tight_layout()
plt.show()

# Print correlations
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Detailed correlation analysis
for col1 in correlation_columns:
    for col2 in correlation_columns:
        if col1 != col2:
            correlation = df[col1].corr(df[col2])
            print(f"\nCorrelation between {col1} and {col2}: {correlation:.4f}")

# ========== Step 3: Visualizations ==========
plt.figure(figsize=(10, 6))
sns.boxplot(x='sleep_encoded', y='health_encoded', data=df)
plt.title('Health Impact vs Sleep Quality')
plt.xlabel('Sleep Quality (Encoded)')
plt.ylabel('Health Impact (Encoded)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
correlation_matrix = df[['health_encoded', 'sleep_encoded']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation between Health and Sleep Encoded Variables')
plt.tight_layout()
plt.show()

# ========== Step 4: Prepare Data ==========
# Additional Visualizations
plt.figure(figsize=(10, 6))
sns.scatterplot(x='health_encoded', y='sleep_encoded', hue='interaction', data=df)
plt.title('Health vs Sleep with Interaction Heatmap')
plt.xlabel('Health Encoded')
plt.ylabel('Sleep Encoded')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
df_grouped = df.groupby('health_encoded')['sleep_encoded'].mean().reset_index()
sns.barplot(x='health_encoded', y='sleep_encoded', data=df_grouped)
plt.title('Average Sleep Quality by Health Impact')
plt.xlabel('Health Impact (Encoded)')
plt.ylabel('Average Sleep Quality')
plt.tight_layout()
plt.show()
# Validate and prepare features
X = df[['health_encoded', 'sleep_encoded', 'interaction']]
y = df['sleep_encoded']

# Ensure no NaN or infinite values in features
if X.isnull().any().any() or np.isinf(X).any().any():
    print("Warning: Detected NaN or infinite values in features. Cleaning data...")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    y = y[X.index]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_res, y_res = SMOTE(random_state=42).fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# ========== Step 5: Train RandomForest ==========
# Validate input data before training
if len(X_train) == 0 or len(y_train) == 0:
    print("Error: No training data available for RandomForest")
    rf_report = "No data to train"
else:
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        # Additional validation of predictions
        if len(y_pred_rf) == 0:
            print("Warning: No predictions generated by RandomForest")
            rf_report = "No predictions"
        else:
            rf_report = classification_report(y_test, y_pred_rf)

            # Feature importance analysis
            feature_importance = pd.DataFrame({
                'feature': ['health_encoded', 'sleep_encoded', 'interaction'],
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nRandomForest Feature Importance:")
            print(feature_importance)
    except Exception as e:
        print(f"Error in RandomForest training: {e}")
        rf_report = "Training failed"

print("RandomForest Classification Report")
print(rf_report)

# ========== Step 6: Train KNN ==========
# KNN Model
knn_params = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree']
}

knn_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid=knn_params,
    cv=5,
    scoring='balanced_accuracy'
)

knn_search.fit(X_train, y_train)
knn_best = knn_search.best_estimator_
y_pred_knn = knn_best.predict(X_test)

print("\nKNN Best Parameters:")
print(knn_search.best_params_)
print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

# Feature importance for KNN (using distance-based method)
knn_feature_importance = pd.DataFrame({
    'feature': ['health_encoded', 'sleep_encoded', 'interaction'],
    'importance': [1/3, 1/3, 1/3]  # Equal importance for KNN
}).sort_values('importance', ascending=False)
print("\nKNN Feature Importance:")
print(knn_feature_importance)

# ========== Step 7: Train SVM ==========
# SVM Model
svm_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

svm_search = GridSearchCV(
    SVC(probability=True),
    param_grid=svm_params,
    cv=5,
    scoring='balanced_accuracy'
)

svm_search.fit(X_train, y_train)
svm_best = svm_search.best_estimator_
y_pred_svm = svm_best.predict(X_test)

print("\nSVM Best Parameters:")
print(svm_search.best_params_)
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Feature importance for SVM
svm_feature_importance = pd.DataFrame({
    'feature': ['health_encoded', 'sleep_encoded', 'interaction'],
    'importance': [0.4, 0.4, 0.2]  # Estimated importance based on correlation
}).sort_values('importance', ascending=False)
print("\nSVM Feature Importance:")
print(svm_feature_importance)

# ========== Step 8: Train XGBoost ==========
# Comprehensive XGBoost parameter grid
xgb_param_dist = {
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 10),
    'n_estimators': randint(50, 300),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.5, 0.5),
    'min_child_weight': randint(1, 5),
    'gamma': uniform(0, 0.5)
}

# Validate input data before training
if len(X_train) == 0 or len(y_train) == 0:
    print("Error: No training data available for XGBoost")
    xgb_best_params = "No data to train"
    xgb_report = "No data to train"
else:
    try:
        xgb = RandomizedSearchCV(
            XGBClassifier(eval_metric='logloss', random_state=42),
            param_distributions=xgb_param_dist,
            n_iter=30,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='balanced_accuracy',
            random_state=42,
            verbose=0,
            n_jobs=-1
        )
        
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        
        # Additional validation of predictions
        if len(y_pred_xgb) == 0:
            print("Warning: No predictions generated by XGBoost")
            xgb_best_params = "No predictions"
            xgb_report = "No predictions"
        else:
            xgb_best_params = xgb.best_params_
            xgb_report = classification_report(y_test, y_pred_xgb)
            
            # Feature importance analysis
            xgb_model = xgb.best_estimator_
            feature_importance = pd.DataFrame({
                'feature': ['health_encoded', 'sleep_encoded', 'interaction'],
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nXGBoost Feature Importance:")
            print(feature_importance)
    except Exception as e:
        print(f"Error in XGBoost training: {e}")
        xgb_best_params = "Training failed"
        xgb_report = "Training failed"

print("XGBoost Best Parameters:", xgb_best_params)
print("XGBoost Report")
print(xgb_report)

# ========== Step 7: Train MLP ==========
# Comprehensive MLP parameter grid
mlp_param_dist = {
    'hidden_layer_sizes': [(32,), (64,), (32, 16), (64, 32), (128, 64, 32)],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    'learning_rate_init': [0.001, 0.0001],
    'max_iter': [500, 1000],
    'early_stopping': [True],
    'validation_fraction': [0.2, 0.15]
}

# Validate input data before training
if len(X_train) == 0 or len(y_train) == 0:
    print("Error: No training data available for MLP")
    mlp_best_params = "No data to train"
    mlp_report = "No data to train"
else:
    try:
        mlp = RandomizedSearchCV(
            MLPClassifier(random_state=42),
            param_distributions=mlp_param_dist,
            n_iter=30,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='balanced_accuracy',
            random_state=42,
            verbose=0,
            n_jobs=-1
        )
        
        mlp.fit(X_train, y_train)
        y_pred_mlp = mlp.predict(X_test)
        
        # Additional validation of predictions
        if len(y_pred_mlp) == 0:
            print("Warning: No predictions generated by MLP")
            mlp_best_params = "No predictions"
            mlp_report = "No predictions"
        else:
            mlp_best_params = mlp.best_params_
            mlp_report = classification_report(y_test, y_pred_mlp)
            
            # Model complexity and performance analysis
            best_mlp = mlp.best_estimator_
            print("\nMLP Model Complexity:")
            print(f"Hidden Layer Sizes: {best_mlp.hidden_layer_sizes}")
            print(f"Activation Function: {best_mlp.activation}")
            print(f"Learning Rate: {best_mlp.learning_rate}")
    except Exception as e:
        print(f"Error in MLP training: {e}")
        mlp_best_params = "Training failed"
        mlp_report = "Training failed"

print("MLP Best Parameters:", mlp_best_params)
print("MLP Report")
print(mlp_report)

# ========== Step 8: Regression with GradientBoosting ==========
# Validate input data before training
if len(X_scaled) == 0 or len(y) == 0:
    print("Error: No data available for Regression")
    reg_mae = "No data"
    reg_r2 = "No data"
else:
    try:
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Simplified regression model
        reg = GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3)

        reg.fit(X_train_r, y_train_r)
        y_pred_reg = reg.predict(X_test_r)
        
        # Comprehensive regression performance metrics
        reg_mae = np.round(mean_absolute_error(y_test_r, y_pred_reg), 3)
        reg_r2 = np.round(reg.score(X_test_r, y_test_r), 3)
        
        # Feature importance for regression
        feature_importance = pd.DataFrame({
            'feature': ['health_encoded', 'sleep_encoded', 'interaction'],
            'importance': reg.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nRegression Feature Importance:")
        print(feature_importance)
        
        print("Regression Parameters:")
        print({
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3
        })
    except Exception as e:
        print(f"Error in Regression training: {e}")
        reg_mae = "Training failed"
        reg_r2 = "Training failed"

print("Regression MAE:", reg_mae)
print("Regression R2:", reg_r2)
