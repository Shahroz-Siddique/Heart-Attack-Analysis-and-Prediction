import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pytest
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import select_k_best_features
from src.train_model import train_random_forest
from sklearn.metrics import accuracy_score

# Fixtures
@pytest.fixture
def raw_df():
    return load_data("/home/foxtech/SHAHROZ_PROJ/Heart Attack Analysis and Prediction/data/heart.csv")

def test_load_data(raw_df):
    assert isinstance(raw_df, pd.DataFrame)
    assert not raw_df.empty
    assert "output" in raw_df.columns

def test_preprocessing(raw_df):
    X_train, X_test, y_train, y_test = preprocess_data(raw_df)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]

def test_feature_selection(raw_df):
    X = raw_df.drop('output', axis=1)
    y = raw_df['output']
    X_new, selected = select_k_best_features(X, y, k=8)
    assert X_new.shape[1] == 8
    assert len(selected) == 8

def test_model_training(raw_df):
    X = raw_df.drop('output', axis=1)
    y = raw_df['output']
    X_train, X_test, y_train, y_test = preprocess_data(raw_df)
    model = train_random_forest(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc > 0.6  # reasonable threshold for a basic model
