import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

def train_fraud_models(X_train, y_train, X_test, y_test):
    models = {}
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    rf.fit(X_train, y_train)
    models['rf'] = rf
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=7, random_state=42)
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10)
    models['xgb'] = xgb_model
    
    return models

def evaluate(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        print(f"\n{name.upper()} Results:")
        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
