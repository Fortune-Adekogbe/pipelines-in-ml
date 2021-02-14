# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

from catboost import CatBoostClassifier
import joblib


train_df = pd.read_csv('/content/drive/My Drive/DSN Challenge - Insurance Prediction/train_data.csv')

Claim = train_df.pop('Claim')
train_df.pop('Customer Id');
train_df


X_train, X_valid, y_train,y_valid = train_test_split(train_df,Claim,test_size=0.25,
                                                   stratify=Claim,random_state=50
                                              )


class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    
class NanFillerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, normalize_ = False):
        self.column = column
        self.normalize_ = normalize_

    def fit(self, X, y=None):
        return self

    def replace_nan(self, X,feature_name,new_values,weights):
        '''
        A function that replaces the nan values with some different values in 
        different proportions.
        ------------------------------------------------------------------------
        Parameters:
        
            X: a dataframe  
            feature_name: the name of the feature to be worked on as a string
            new_values: this is a list of the values to replace the nan values 
                        as strings
            weights: this is a list of the weights assigned to each value. 
        ------------------------------------------------------------------------
        Returns:
            The edited feature as a numpy array.
        '''
        assert len(new_values)==len(weights),'New values do not correspond with\
         weights'
        from random import choices
        mask= X[feature_name].isna()
        length = sum(mask)
        replacement = choices(new_values,weights =weights,k=length)
        X.loc[mask,feature_name]=replacement
        return X[feature_name]

    def transform(self, X):
        x = X[self.column].value_counts(normalize=True)
        X[self.column] = self.replace_nan(X,self.column,x.keys(),x.values)
        if self.normalize_: X[self.column] = X[self.column]/X[self.column].max()
        return X


class EncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,show_map = False):
        self.show_map = show_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['NoWindows'] = X['NumberOfWindows'].map(lambda x: 1 if x == '   .' else 0)
        X['3-7Windows'] = X['NumberOfWindows'].map(lambda x: 1 if x in '34567' else 0)
        X['Other_Windows'] = X['NumberOfWindows'].map(lambda x: 1 if x in '1 2 8 9 >=10'.split(' ') else 0)
        X['Sectioned_Insured_Period'] = X['Insured_Period'].map(lambda x: 1 if x==1 else 0)
        X['Grouped_Date_of_Occupancy'] = X['Date_of_Occupancy'].map(lambda x: 1 if x>1900 else 0)
        return X

class YearEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, show_map = False):
        self.show_map = show_map

    def fit(self, X, y=None):
        return self
    
    def map_counts(self, X,feat):
        mapp = X[feat].value_counts(normalize=True)
        X[feat].map(lambda x:mapp[x])
        return mapp

    def transform(self, X):
        X['2012-13YearOfObs'] = X['YearOfObservation'].map(lambda x: 1 if x in [2012,2013] else 0)
        X['2014YearOfObs'] = X['YearOfObservation'].map(lambda x: 1 if x in [2014] else 0)
        X['2015-16YearOfObs'] = X['YearOfObservation'].map(lambda x: 1 if x in [2015,2016] else 0)
        X['YearOfObservation']= X['YearOfObservation'].map(lambda x: 2021 - x)
        if self.show_map:
            self.map_counts(datum,'YearOfObservation')
        return X

class FeatureCombiningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, drop_any=[]):
        self.columns = columns
        self.drop_any = drop_any

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        suffix = ''.join([i[0] for i in self.columns])
        X[f'Combined_{suffix}'] = X[self.columns].sum(axis=0)
        for j in self.drop_any:
          X.pop(self.columns[j])
          print(f">>> Removed {self.columns[j]} from dataframe")
        return X

class DummyFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column = None):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.column:   
            X = pd.get_dummies(X,columns=[self.column])
        else:
            X = pd.get_dummies(X)
        return X

        
class NormalizedFrequencyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, replace_original= True):
        self.column = column
        self.replace_original = replace_original

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mapper = X[self.column].value_counts(normalize=True)
        if self.replace_original:
          X[self.column] = X[self.column].map(lambda x:mapper[x])    
        else:
          X[f"Coded_{self.column}"] = X[self.column].map(lambda x:mapper[x])    
        return X


# The Pipeline
data_pipeline = Pipeline([
        ('et',EncoderTransformer()),
        ('yet', YearEncoderTransformer()),
        ('fct',FeatureCombiningTransformer(['Garden', 'Building_Fenced', 'Settlement'], [0])),
        ('nanft1', NanFillerTransformer('Building Dimension', normalize_ = True)),
        ('nanft2', NanFillerTransformer('Date_of_Occupancy')),
        ('normft1', NormalizedFrequencyTransformer("Date_of_Occupancy")),
        ('nanft3', NanFillerTransformer('Geo_Code')),
        ('normft2', NormalizedFrequencyTransformer("Geo_Code")),
        ('normft3', NormalizedFrequencyTransformer("Insured_Period", replace_original = False)),
        ('normft4', NormalizedFrequencyTransformer("YearOfObservation", replace_original = False)),
        ('dft', DummyFeaturesTransformer('')),
])


class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.column]

class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        le = LabelEncoder()
        return le.fit_transform(X.values).reshape(-1,1)

# Feature union
Settlement_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['Settlement'])),
    ("le",Encoder())
])

Building_Fenced_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['Building_Fenced'])),
    ("le",Encoder())

])

Building_Painted_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['Building_Painted'])),
    ("le",Encoder())

])

categorical_features = FeatureUnion([
    ("Settlement_onehot", Settlement_onehot),
    ("Building_Fenced_onehot", Building_Fenced_onehot),
    ("Building_Painted_onehot", Building_Painted_onehot),
])


data_transformer = FeatureUnion([
    ('features', data_pipeline),
    ('categorical', categorical_features),
])

train = data_transformer.fit_transform(X_train)
validate = data_transformer.fit_transform(X_valid)

# Modelling
cbc1 = CatBoostClassifier(verbose=0,learning_rate = 0.01, n_estimators=400)
cbc1.fit(train, y_train)
print(f'Train score: {log_loss(y_train, cbc1.predict_proba(train))}')
print(f'Validation score: {log_loss(y_valid, cbc1.predict_proba(validate))}')

# Saving our Model
filename = 'Insurance_model.sav'
joblib.dump(cbc1,open(filename,'wb'))
