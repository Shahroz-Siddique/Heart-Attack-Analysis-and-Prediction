import os
import pandas as pd
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import select_k_best_features
from src.train_model import train_random_forest, save_model
from src.evaluate_model import evaluate

# Load and preprocess
df = load_data('data/raw/heart.csv')
X_train, X_test, y_train, y_test = preprocess_data(df)

# Optional feature selection
X_train_df = pd.DataFrame(X_train, columns=df.columns[:-1])
X_test_df = pd.DataFrame(X_test, columns=df.columns[:-1])
X_train_new, selected = select_k_best_features(X_train_df, y_train)
X_test_new = X_test_df[selected]

# Train model
model = train_random_forest(X_train_new, y_train)

# Save model
os.makedirs('models', exist_ok=True)
save_model(model, 'models/heart_attack_model.pkl')

# Evaluate
evaluate(model, X_test_new, y_test)
