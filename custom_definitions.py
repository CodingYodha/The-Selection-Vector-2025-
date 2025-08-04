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

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from scipy.stats import zscore

class lsZScoreOutlierRemover(BaseEstimator, TransformerMixin):
    def _init_(self, threshold=3.0):
        self.threshold = threshold

    def fit(self, Xls, yls=None):
        return self  # stateless

    def transform(self, Xls, yls=None):
        Xls = Xls.copy()

        # Feature engineering first (to include them in outlier handling)
        Xls["Proximity_Temp"] = Xls["Proximity to Star"] / (Xls["Surface Temperature"] + 1e-5)
        Xls["Surface Density"] = Xls["Mineral Abundance"] * (Xls["Surface Temperature"] + 1e-5)
        Xls["log_Density"] = np.log1p(Xls["Atmospheric Density"] - Xls["Atmospheric Density"].min() + 1)

        # Select numeric columns
        num_colsls = Xls.select_dtypes(include=[np.number]).columns.tolist()

        # Ensure all numeric data is float (important for NaNs)
        Xls[num_colsls] = Xls[num_colsls].astype(float)

        # Compute Z-scores
        z_scoresls = np.abs(zscore(Xls[num_colsls], nan_policy='omit'))

        # Handle 1D zscore output (edge case)
        if z_scoresls.ndim == 1:
            z_scoresls = z_scoresls[:, np.newaxis]

        # Replace outliers with NaN
        Xls[num_colsls] = Xls[num_colsls].where(z_scoresls < self.threshold, np.nan)

        return Xls
