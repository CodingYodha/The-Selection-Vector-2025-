import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer , LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline as ImbPipeline   
from imblearn.over_sampling import SMOTE   
from sklearn.linear_model import LogisticRegression   
from splitter import split_columns


#==============Your Name======================
# Your code
#==============Your Name=====================
#Dont remove the following snippet and follow the same

#==========Arjun E Naik==============
class YesNoConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        mapping = {'yes': 1, 'y': 1, 'no': 0, 'n': 0, '1': 1, '0': 0}

        return X.iloc[:, 0].apply(lambda x: mapping.get(str(x).lower(), 0)).values.reshape(-1, 1)

class ReplaceInfWithNan(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        return X.replace([np.inf, -np.inf], np.nan)

class ConvertToString(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.astype(str)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X_future = X.copy()
        X_future['new_feature_1'] = X_future['portland_cement_kg'] * X_future['mixing_water_kg']
        X_future['new_feature_2'] = X_future['ground_slag_kg'] + X_future['coal_ash_kg']
        return X_future
#==========Arjun E Naik==============

#=========Rudra======================
import re

from word2number import w2n
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Define custom column dropper class
class ColumnDropper(BaseEstimator, TransformerMixin):
    def _init_(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop, errors='ignore')

# Define classes for feature engineering
class CementGradeParser:
    def _call_(self, X):
        def parse(val):
            try:
                return float(val)
            except:
                val = str(val).lower().strip()
                num_match = re.findall(r"\d+\.?\d*", val)
                if num_match:
                    return float(num_match[0])
                try:
                    return w2n.word_to_num(val)
                except:
                    return None
        X['cement_grade'] = X['cement_grade'].apply(parse)
        return X

class DatePreprocessor:
    def _call_(self, X):
        X['last_modified'] = pd.to_datetime(X['last_modified'], errors='coerce')
        X['last_modified_year'] = X['last_modified'].dt.year.astype('int64')
        X['last_modified_month'] = X['last_modified'].dt.month.astype('int64')
        X['last_modified_day'] = X['last_modified'].dt.day.astype('int64')
        X['last_modified_weekday'] = X['last_modified'].dt.dayofweek.astype('int64')
        X.drop(columns=['last_modified'], inplace=True)

        X['inspection_timestamp'] = pd.to_datetime(X['inspection_timestamp'], errors='coerce')
        X['inspection_year'] = X['inspection_timestamp'].dt.year.fillna(-1).astype('int64')
        X['inspection_month'] = X['inspection_timestamp'].dt.month.fillna(-1).astype('int64')
        X['inspection_day'] = X['inspection_timestamp'].dt.day.fillna(-1).astype('int64')
        X['inspection_weekday'] = X['inspection_timestamp'].dt.dayofweek.fillna(-1).astype('int64')
        X.drop(columns=['inspection_timestamp'], inplace=True)
        return X

class CategoryConverter:
    def _call_(self, X):
        approval_mapping = {'yes': True, 'y': True, '1': True, 'no': False, 'n': False, '0': False}
        X['is_approved'] = X['is_approved'].astype(str).str.strip().str.lower().map(approval_mapping).astype(int)
        rating_map = {'A ++': 6, 'A+': 5, 'AA': 4.5, 'A': 4, 'A-': 3.5, 'B': 3, 'C': 2}
        X['supplier_rating'] = X['supplier_rating'].map(rating_map)
        X['is_valid_strength'] = X['is_valid_strength'].astype(int)
        X['static_col'] = X['static_col'].astype(int)
        return X

class TimeDeltaAndRatios:
    def _call_(self, X):
        X['time_since_casting'] = pd.to_timedelta(X['time_since_casting'], errors='coerce')
        X['time_since_casting_days'] = X['time_since_casting'].dt.total_seconds() / (24 * 3600)
        X.drop('time_since_casting', axis=1, inplace=True)
        X['water_binder_ratio'] = X['mixing_water_kg'] / (X['total_binder_kg'] + 1e-6)
        X['admixture_binder_ratio'] = X['chemical_admixture_kg'] / (X['total_binder_kg'] + 1e-6)
        return X

class RedundantDropper:
    def _call_(self, X):
        X.drop(columns=[
            'time_since_casting_days', 'days_since_last_modified',
            'days_since_inspection', 'random_noise'
        ], inplace=True, errors='ignore')
        return X

class FinalCleaner:
    def _call_(self, X):
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        imputer = SimpleImputer(strategy='mean')
        X[num_cols] = imputer.fit_transform(X[num_cols])
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        return X

#===============Rudra=============

#==================================shrihari=====================================================================

class FinalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.columns_)

        cement = df['Cement_component_1kg_in_a_m3_mixture'].replace(0, 0.0001)
        df['total_binder'] = (
            df['Cement_component_1kg_in_a_m3_mixture'] +
            df['Blast_Furnace_Slag_component_2kg_in_a_m3_mixture'] +
            df['Fly_Ash_component_3kg_in_a_m3_mixture']
        )
        total_binder_for_ratio = df['total_binder'].replace(0, 0.0001)
        df['water_cement_ratio'] = df['Water_component_4kg_in_a_m3_mixture'] / cement
        total_aggregate = (
            df['Coarse_Aggregate_component_6kg_in_a_m3_mixture'] +
            df['Fine_Aggregate_component_7kg_in_a_m3_mixture']
        )
        df['agg_binder_ratio'] = total_aggregate / total_binder_for_ratio

        safe_age = df['Age_day'].clip(lower=0)
        df['log_Age_day'] = np.log1p(safe_age)

        df = df.drop(columns=[
            'Cement_component_1kg_in_a_m3_mixture',
            'Blast_Furnace_Slag_component_2kg_in_a_m3_mixture',
            'Fly_Ash_component_3kg_in_a_m3_mixture',
            'Water_component_4kg_in_a_m3_mixture',
            'Coarse_Aggregate_component_6kg_in_a_m3_mixture',
            'Fine_Aggregate_component_7kg_in_a_m3_mixture'
        ])
        
        return df
    
column_mapping = {
    'portland_cement_kg': 'Cement_component_1kg_in_a_m3_mixture',
    'ground_slag_kg': 'Blast_Furnace_Slag_component_2kg_in_a_m3_mixture',
    'coal_ash_kg': 'Fly_Ash_component_3kg_in_a_m3_mixture',
    'mixing_water_kg': 'Water_component_4kg_in_a_m3_mixture',
    'chemical_admixture_kg': 'Superplasticizer_component_5kg_in_a_m3_mixture',
    'gravel_aggregate_kg': 'Coarse_Aggregate_component_6kg_in_a_m3_mixture',
    'sand_aggregate_kg': 'Fine_Aggregate_component_7kg_in_a_m3_mixture',
    'specimen_age_days': 'Age_day'
}
df = df.rename(columns=column_mapping)

TARGET = 'compressive_strength_mpa'
CORE_FEATURES = list(column_mapping.values())
df_clean = df[CORE_FEATURES + [TARGET]]

df_clean.dropna(subset=[TARGET], inplace=True)

# 2. Handle Outliers
for column in df_clean.columns:
    if df_clean[column].dtype in ['int64', 'float64']:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean[column] = np.where(df_clean[column] > upper_bound, upper_bound, df_clean[column])
        df_clean[column] = np.where(df_clean[column] < lower_bound, lower_bound, df_clean[column])

# 3. Prepare Final Data and Pipeline
X = df_clean.drop(TARGET, axis=1)
y = df_clean[TARGET]


def split_columns(df):
    df = df.copy()
    for col in df.columns:
        if ',' in col:
            parts = col.split(',')
            df[parts[0]] = df[col].str[0]
            df[parts[1]] = df[col].str[1]
            df.drop(columns=col, inplace=True)
    return df


def fill_missing(df):

    return df.fillna('')
    
def split_bycomma(X_df):
    X = X_df.copy()
    X[['feature_8', 'feature_15']] = X['feature_8,feature_15'].str.split(',', expand=True)
    X[['feature_21', 'feature_10']] = X['feature_21,feature_10'].str.split(',', expand=True)
    X[['feature_1', 'feature_6']] = X['feature_1,feature_6'].str.split(',', expand=True)
    X.drop(['feature_8,feature_15', 'feature_21,feature_10', 'feature_1,feature_6'], axis=1, inplace=True)
    return X
fillna_transformer = FunctionTransformer(fill_missing, validate=False)


def replacer(X):
   X_temp=X.copy()
   for col in X.columns:
      X_temp[f"{col}_letters"] = X_temp[col].str.replace(r'[^a-zA-Z]', "", regex=True)
      X_temp[f"{col}_symbols"] = X_temp[col].str.replace(r'[a-zA-Z0-9]', "", regex=True)
      X_temp.drop(columns=[col],inplace=True)
    
 
   return X_temp

custom_replacer=FunctionTransformer(replacer,validate=False)



class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, top_features=None):
        self.top_features = top_features if top_features else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[['feature_8', 'feature_15']] = X['feature_8,feature_15'].str.split(",", expand=True)
        X[['feature_21', 'feature_10']] = X['feature_21,feature_10'].str.split(",", expand=True)
        X[['feature_1', 'feature_6']] = X['feature_1,feature_6'].str.split(",", expand=True)
        X.drop(['feature_8,feature_15', 'feature_21,feature_10', 'feature_1,feature_6'], axis=1, inplace=True)

        string_cols = X.select_dtypes(include='object').columns
        for col in string_cols:
            X[col] = X[col].str.lower().str.strip()

        return X

class FeatureCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, top_features=None):
        self.top_features = top_features if top_features else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for f1, f2 in combinations(self.top_features, 2):
            new_col = f"{f1}_{f2}_comb"
            X[new_col] = X[f1].astype(str) + "_" + X[f2].astype(str)

        X["count_U"] = (X[self.top_features] == "U").sum(axis=1)
        X["unique_top_cats"] = X[self.top_features].nunique(axis=1)
        return X

class ImputerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = SimpleImputer(strategy="constant")

    def fit(self, X, y=None):
        self.imputer.fit(X)
        self.columns = X.columns
        return self

    def transform(self, X):
        return pd.DataFrame(self.imputer.transform(X), columns=self.columns, index=X.index)


class OneHotEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, X, y=None):
        self.obj_cols = X.select_dtypes(include="object").columns
        self.encoder.fit(X[self.obj_cols])
        return self

    def transform(self, X):
        X = X.copy()
        encoded = self.encoder.transform(X[self.obj_cols])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.obj_cols), index=X.index)
        return pd.concat([X.drop(columns=self.obj_cols), encoded_df], axis=1)


class KMeansClusterWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    def fit(self, X, y=None):
        self.kmeans.fit(X)
        return self

    def transform(self, X):
        cluster_labels = self.kmeans.predict(X)
        return pd.concat([X.reset_index(drop=True), pd.Series(cluster_labels, name="cluster_label")], axis=1)


class ScalerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)


# --- Full Preprocessing Pipeline ---
def create_pipeline():
    top_features = ["feature_18", "feature_1"]
    pipeline = Pipeline([
        ("custom_feature_engineering", CustomFeatureEngineer(top_features=top_features)),
        ("feature_combination", FeatureCombiner(top_features=top_features)),
        ("imputation", ImputerWrapper()),
        ("onehot_encode", OneHotEncoderWrapper()),
        ("clustering", KMeansClusterWrapper()),
        ("scaling", ScalerWrapper())
    ])
    return pipeline




# Define custom column dropper class
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop, errors='ignore')

# Define classes for feature engineering
class CementGradeParser:
    def __call__(self, X):
        def parse(val):
            try:
                return float(val)
            except:
                val = str(val).lower().strip()
                num_match = re.findall(r"\d+\.?\d*", val)
                if num_match:
                    return float(num_match[0])
                try:
                    return w2n.word_to_num(val)
                except:
                    return None
        X['cement_grade'] = X['cement_grade'].apply(parse)
        return X

class DatePreprocessor:
    def __call__(self, X):
        X['last_modified'] = pd.to_datetime(X['last_modified'], errors='coerce')
        X['last_modified_year'] = X['last_modified'].dt.year.astype('int64')
        X['last_modified_month'] = X['last_modified'].dt.month.astype('int64')
        X['last_modified_day'] = X['last_modified'].dt.day.astype('int64')
        X['last_modified_weekday'] = X['last_modified'].dt.dayofweek.astype('int64')
        X.drop(columns=['last_modified'], inplace=True)

        X['inspection_timestamp'] = pd.to_datetime(X['inspection_timestamp'], errors='coerce')
        X['inspection_year'] = X['inspection_timestamp'].dt.year.fillna(-1).astype('int64')
        X['inspection_month'] = X['inspection_timestamp'].dt.month.fillna(-1).astype('int64')
        X['inspection_day'] = X['inspection_timestamp'].dt.day.fillna(-1).astype('int64')
        X['inspection_weekday'] = X['inspection_timestamp'].dt.dayofweek.fillna(-1).astype('int64')
        X.drop(columns=['inspection_timestamp'], inplace=True)
        return X

class CategoryConverter:
    def __call__(self, X):
        approval_mapping = {'yes': True, 'y': True, '1': True, 'no': False, 'n': False, '0': False}
        X['is_approved'] = X['is_approved'].astype(str).str.strip().str.lower().map(approval_mapping).astype(int)
        rating_map = {'A ++': 6, 'A+': 5, 'AA': 4.5, 'A': 4, 'A-': 3.5, 'B': 3, 'C': 2}
        X['supplier_rating'] = X['supplier_rating'].map(rating_map)
        X['is_valid_strength'] = X['is_valid_strength'].astype(int)
        X['static_col'] = X['static_col'].astype(int)
        return X

class TimeDeltaAndRatios:
    def __call__(self, X):
        X['time_since_casting'] = pd.to_timedelta(X['time_since_casting'], errors='coerce')
        X['time_since_casting_days'] = X['time_since_casting'].dt.total_seconds() / (24 * 3600)
        X.drop('time_since_casting', axis=1, inplace=True)
        X['water_binder_ratio'] = X['mixing_water_kg'] / (X['total_binder_kg'] + 1e-6)
        X['admixture_binder_ratio'] = X['chemical_admixture_kg'] / (X['total_binder_kg'] + 1e-6)
        return X

class RedundantDropper:
    def __call__(self, X):
        X.drop(columns=[
            'time_since_casting_days', 'days_since_last_modified',
            'days_since_inspection', 'random_noise'
        ], inplace=True, errors='ignore')
        return X

class FinalCleaner:
    def __call__(self, X):
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        imputer = SimpleImputer(strategy='mean')
        X[num_cols] = imputer.fit_transform(X[num_cols])
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        return X
# This pipeline uses the best configuration we found
submission_pipeline = Pipeline([
    ('feature_engineer', FinalFeatureEngineer()),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=10.0, random_state=42)) # Using the best alpha
])

#===============Shrihari=============================================

