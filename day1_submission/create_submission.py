import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer


class ColumnSplitter(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        
        if 'feature_8,feature_15' in X_copy.columns:
            split_df = X_copy['feature_8,feature_15'].str.split(',', expand=True)
            X_copy['feature_8'] = split_df[0]
            X_copy['feature_15'] = split_df[1]
            X_copy = X_copy.drop(columns=['feature_8,feature_15'])
        
        if 'feature_21,feature_10' in X_copy.columns:
            split_df = X_copy['feature_21,feature_10'].str.split(',', expand=True)
            X_copy['feature_21'] = split_df[0]
            X_copy['feature_10'] = split_df[1]
            X_copy = X_copy.drop(columns=['feature_21,feature_10'])
        
        if 'feature_1,feature_6' in X_copy.columns:
            split_df = X_copy['feature_1,feature_6'].str.split(',', expand=True)
            X_copy['feature_1'] = split_df[0]
            X_copy['feature_6'] = split_df[1]
            X_copy = X_copy.drop(columns=['feature_1,feature_6'])
        return X_copy

class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns_to_keep):
        self.columns_to_keep = columns_to_keep

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns_to_keep]


df = pd.read_csv('DAY1/train_dataset.csv')

X = df.drop('class', axis=1)
y = df['class']

y_encoded = LabelEncoder().fit_transform(y)

top_10_features = [
    'feature_9', 'feature_6', 'feature_12', 'feature_15', 'feature_18',
    'feature_3', 'feature_14', 'feature_20', 'feature_17', 'feature_4'
]

best_c_value = 10

submission_pipeline = ImbPipeline(steps=[
   
    ('splitter', ColumnSplitter()),
    ('selector', FeatureSelector(columns_to_keep=top_10_features)),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ('scaler', StandardScaler()), 
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(C=best_c_value, random_state=42, max_iter=1000))
])


submission_pipeline.fit(X, y_encoded)

filename = 'DAY1/shrihari_telang.pkl'
joblib.dump(submission_pipeline, filename)

