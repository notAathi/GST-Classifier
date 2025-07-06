from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd

# Custom transformer to remove columns with more than a certain percentage of NaN values
class NanColumnsRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.columns_to_drop = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Calculate the percentage of NaN values for each column
        nan_percentage = round(X.isnull().sum()/X.shape[0],3)
        # Identify columns to drop based on the threshold
        self.columns_to_drop = nan_percentage[nan_percentage > self.threshold].index.tolist()
        return self

    def transform(self, X):
        # Drop the identified columns
        return X.drop(self.columns_to_drop, axis=1)
    
# Custom transformer to remove columns with multicollinearity above a given threshold
class MultiColinearityRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.columns_to_drop = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        abs_corr_matrix = X.corr().abs()
        # Create a set to store columns that should be removed
        columns_to_remove = set()
        # Iterate over the upper triangle of the correlation matrix
        for i in range(len(abs_corr_matrix.columns)):
            for j in range(i+1, len(abs_corr_matrix.columns)):
                # If correlation exceeds the threshold, mark one of the columns for removal
                if abs_corr_matrix.iloc[i, j] > self.threshold:
                    col1 = abs_corr_matrix.columns[i]
                    col2 = abs_corr_matrix.columns[j]
                    # Add the second column to the removal set
                    columns_to_remove.add(col1)
        self.columns_to_drop = list(columns_to_remove)
        return self

    def transform(self, X):
        # Drop the identified columns
        return X.drop(self.columns_to_drop, axis=1)

