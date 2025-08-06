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

#==============Arjun E Naik=====================
class Features(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()


        if 'depth_percentage' not in X_copy.columns or 'table_percentage' not in X_copy.columns:
            raise ValueError("Both 'depth_percentage' and 'table_percentage' must be in the input data to compute density.")


        X_copy['depth_table_ratio'] = X_copy['depth_percentage'] / (X_copy['table_percentage'].replace(0, np.nan))
        return X_copy


#==============Arjun E Naik=====================
