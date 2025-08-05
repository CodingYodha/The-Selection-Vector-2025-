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
import imblearn
import re


#==============Your Name======================
# Your code
#==============Your Name=====================
#Dont remove the following snippet and follow the same


#==================================================shrihari====================================================================
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class ST_DiamondFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
        self.color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
        self.clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
        self.ordinal_encoder = OrdinalEncoder(categories=[self.cut_order, self.color_order, self.clarity_order])
        self.numerical_features = []
        self.categorical_cols = ['cut_grade', 'color_grade', 'clarity_grade']

    def fit(self, X, y=None):
        
        df = X.copy()
        
        self.ordinal_encoder.fit(df[self.categorical_cols])
        
        self.numerical_features = df.select_dtypes(include=np.number).columns.drop(self.categorical_cols, errors='ignore')
        return self

    def transform(self, X, y=None):
        df = X.copy()

        if 'price_per_carat' in df.columns:
            df = df.drop(columns=['price_per_carat'])

        df[self.categorical_cols] = self.ordinal_encoder.transform(df[self.categorical_cols])

        for col in self.numerical_features:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)

        df['density'] = df['carat_weight'] / (df['volume_mm3'] + 0.0001)
        df['depth_per_width'] = df['depth_percentage'] / (df['width_mm'] + 0.0001)
        df['carat_x_clarity'] = df['carat_weight'] * (df['clarity_grade'] + 1)
        df['carat_x_color'] = df['carat_weight'] * (df['color_grade'] + 1)
        df['carat_x_cut'] = df['carat_weight'] * (df['cut_grade'] + 1)
        
        return df
    

submission_pipeline = Pipeline(steps=[
    ('feature_engineer', ST_DiamondFeatureEngineer()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])
#==============================shrihari=====================================================================================