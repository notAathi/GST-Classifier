from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,roc_auc_score
import pandas as pd
import joblib


class Model:
    def __init__(self, file_path: str):
        self.model=joblib.load(file_path)
    
    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)

    def metrics(self, X: pd.DataFrame, y: pd.Series):
        y_true=y.values.ravel()
        y_predict = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        accuracy = accuracy_score(y_true, y_predict)
        recall = recall_score(y_true, y_predict)
        precision = precision_score(y_true, y_predict)
        f1= f1_score(y_true, y_predict)
        roc = roc_auc_score(y_true, y_proba)
        
        print(f'Accuracy Score: {accuracy:.3f}')
        print(f'Recall Score: {recall:.3f}')
        print(f'Precision Score: {precision:.3f}')
        print(f'F1 Score: {f1:.3f}')
        print(f'ROC AUC Score: {roc:.5f}')