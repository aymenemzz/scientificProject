import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer










#Pipeline pour les variables catégorielles sans ordre (one-hot encoding)
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

# Pipeline pour les variables ordinales (ST_Slope)
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(categories=[st_slope_categories]))
]) if ordinal_features else None

# Pipeline pour les variables numériques avec KNNImputer et normalisation
numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())  # Standardisation des variables numériques
])

# Création de la liste des transformateurs
transformers = [
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_no_order)
]

# Ajouter le transformateur ordinal s'il y a des variables ordinales
if ordinal_features:
    transformers.append(('ord', ordinal_transformer, ordinal_features))

# Assemblage des transformations avec ColumnTransformer
preprocessor = ColumnTransformer(transformers=transformers)