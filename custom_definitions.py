

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['Cement'] = df['Cement_component_1kg_in_a_m3_mixture'].replace(0, 0.0001)
        df['Water'] = df['Water_component_4kg_in_a_m3_mixture'].replace(0, 0.0001)
        df['Superplasticizer'] = df['Superplasticizer_component_5kg_in_a_m3_mixture'].replace(0, 0.0001)

        df['water_cement_ratio'] = df['Water'] / df['Cement']
        df['total_binder'] = (
            df['Cement_component_1kg_in_a_m3_mixture'] +
            df['Blast_Furnace_Slag_component_2kg_in_a_m3_mixture'] +
            df['Fly_Ash_component_3kg_in_a_m3_mixture']
        )
        df['total_aggregate'] = (
            df['Coarse_Aggregate_component_6kg_in_a_m3_mixture'] +
            df['Fine_Aggregate_component_7kg_in_a_m3_mixture']
        )
        df['plasticizer_per_cement'] = df['Superplasticizer'] / df['Cement']
        df['cement_agg_ratio'] = df['Cement'] / df['total_aggregate']
        df['water_binder_ratio'] = df['Water'] / df['total_binder']
        df['mix_density'] = (
            df['Cement_component_1kg_in_a_m3_mixture'] +
            df['Blast_Furnace_Slag_component_2kg_in_a_m3_mixture'] +
            df['Fly_Ash_component_3kg_in_a_m3_mixture'] +
            df['Water'] +
            df['Superplasticizer'] +
            df['Coarse_Aggregate_component_6kg_in_a_m3_mixture'] +
            df['Fine_Aggregate_component_7kg_in_a_m3_mixture']
        )
        return df

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.skewed_cols = [
            'Age_day',
            'water_cement_ratio',
            'Superplasticizer_component_5kg_in_a_m3_mixture',
            'plasticizer_per_cement',
            'Blast_Furnace_Slag_component_2kg_in_a_m3_mixture',
            'cement_agg_ratio',
            'Fly_Ash_component_3kg_in_a_m3_mixture',
            'Cement_component_1kg_in_a_m3_mixture'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.skewed_cols:
            if col in X.columns:
                if X[col].min() > 0:
                    X_transformed[col] = np.log1p(X[col])
                else:
                    X_transformed[col] = np.log1p(X[col] - X[col].min() + 1)
        return X_transformed

class LowercaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.columns:
            if X_copy[col].dtype == 'object' or X_copy[col].dtype == 'string':
                X_copy[col] = X_copy[col].astype(str).str.lower()
        return X_copy


class RemoveUnusualChars(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, pattern=r'[^a-zA-Z0-9\s]'):
        self.columns = columns
        self.pattern = pattern

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        columns_to_clean = self.columns or X_copy.select_dtypes(include=['object', 'string']).columns

        for col in columns_to_clean:
            X_copy[col] = X_copy[col].astype(str).apply(lambda x: re.sub(self.pattern, '', x))

        return X_copy

# shrihari telang,
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
