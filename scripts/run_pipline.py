import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import select_k_best_features
from src.train_model import train_random_forest, save_model
from src.evaluate_model import evaluate

# Load and preprocess the data
df = load_data('/home/foxtech/SHAHROZ_PROJ/Heart Attack Analysis and Prediction/data/heart.csv')
X_train, X_test, y_train, y_test = preprocess_data(df)

# Convert to DataFrames with column names
feature_columns = df.columns[:-1] # all columns except the target
X_train_df = pd.DataFrame(X_train, columns=feature_columns)
X_test_df = pd.DataFrame(X_test, columns=feature_columns)

# Feature Selection (SelectKBest with k=8)
X_train_selected, selected_columns = select_k_best_features(X_train_df, y_train, k=8)
X_test_selected = X_test_df[selected_columns]

# Train the model
model = train_random_forest(X_train_selected, y_train)

# Save the model and selected feature names
os.makedirs('models', exist_ok=True)
save_model(model, 'models/heart_attack_model.pkl')
joblib.dump(selected_columns, 'models/selected_features.pkl')

# Evaluate the model
evaluate(model, X_test_selected, y_test)

print("✅ Training pipeline completed successfully!")