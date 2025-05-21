#!/usr/bin/env python
# coding: utf-8

# #  Data & IA Project
# ### Aymene Mazouz - Yasmina Moussaoui - Toby Pollock

# ## Objectif du Projet
# Les maladies cardiaques constituent l'une des principales causes de mortalité dans le monde. La capacité à prédire le risque de développer une maladie cardiaque à partir de données médicales est cruciale pour mettre en place des mesures préventives adaptées, améliorer les traitements, et sauver des vies. Ce projet a pour but de développer un modèle prédictif capable d'évaluer le risque qu'un patient soit atteint d'une maladie cardiaque en se basant sur ses données cliniques, et ceci avec une bonne precision.
# pas de préprocesseur, knn inputer to handle nanvlues 

# ## Description des Données
# Le dataset utilisé dans ce projet, intitulé heart_disease_data.csv, contient les informations médicales de plusieurs patients.
# Les attributs de ces données sont les suivants :
# - **Age** : âge du patient (en années).
# - **Sex** : sexe du patient (M : Homme, F : Femme).
# - **ChestPainType** : type de douleur thoracique (TA : Angine typique, ATA : Angine atypique, NAP : Douleur non angineuse, ASY : Asymptomatique).
# - **RestingBP** : tension artérielle au repos (en mm Hg).
# - **Cholesterol** : cholestérol sérique (en mg/dl).
# - **FastingBS** : glycémie à jeun (1 : si > 120 mg/dl, 0 : sinon).
# - **RestingECG** : résultats de l’électrocardiogramme au repos (Normal : Normal, ST : anomalies ST-T, LVH : Hypertrophie ventriculaire gauche).
# - **MaxHR** : fréquence cardiaque maximale atteinte.
# - **ExerciseAngina** : angine provoquée par l’effort (Y : Oui, N : Non).
# - **Oldpeak** : dépression du segment ST mesurée (valeur numérique).
# - **ST_Slope** : pente du segment ST à l’effort (Up : ascendante, Flat : plate, Down :
# descendante).
# - **HeartDisease** : présence ou absence de maladie cardiaque (1 : Oui, 0 : Non).
# - knn inputer 

# In[288]:


# DataCleaning.py
#Import libraries
# DataCleaning.py
def initialize_libraries():
    global np, pd, plt, sns, chi2_contingency, kendalltau, xgb
    global StandardScaler, OneHotEncoder, OrdinalEncoder, KNNImputer, ColumnTransformer, Pipeline, VarianceThreshold
    global train_test_split, GridSearchCV, cross_val_score, KNeighborsClassifier
    global classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, auc, roc_auc_score

    # Pour le prétraitement des données

    # Pour la modélisation


# In[289]:


initialize_libraries()


# In[290]:


# Configuration de l'affichage
plt.style.use('seaborn-v0_8-notebook')
sns.set_palette('Set2')


# In[291]:


#Chargement des données
df = pd.read_csv('./data/heart_disease_data.csv')


# In[260]:


#Affichage des premiéres lignes
df.head()


# ### Informations genérales

# In[261]:


#Comptage des valeurs True et false de la target qui est HeartDisease, pour voir si on a un dataset uniforme
count_values = df['HeartDisease'].value_counts()
print(count_values)


# In[262]:


#Informations générales 
print("Informations sur les données :")
df.info()

print("\nStatistiques descriptives :")
df.describe()


# In[263]:


# Vérification des valeurs manquantes
print("\nValeurs manquantes :")
print(df.isnull().sum())


# In[264]:


## Visualisation des variables catégorielles
# Analyse des variables catégorielles
cat_features = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nVariables catégorielles détectées : {cat_features}")

# Variables numériques
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
num_features.remove('HeartDisease')  # Retirer la variable cible des features numériques

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, feature in enumerate(cat_features):
    sns.countplot(data=df, x=feature, hue=feature, palette='Set3', ax=axes[idx])
    axes[idx].set_title(f"Répartition de {feature}")
    axes[idx].tick_params(axis='x', rotation=45)

# Supprimer les subplots vides (si nombre impair)
for i in range(len(cat_features), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
# Supprimer les subplots vides (si nombre impair)
for i in range(len(cat_features), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# ### Matrice de corrélation 

# In[265]:


# Fonction pour calculer le coefficient V de Cramér (pour les variables catégorielles)
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

# Création d'une matrice de corrélation mixte
corr_matrix = pd.DataFrame(index=cat_features+num_features+['HeartDisease'], 
                           columns=cat_features+num_features+['HeartDisease'])

print("\nConstruction de la matrice de corrélation mixte...")
for var1 in corr_matrix.index:
    for var2 in corr_matrix.columns:
        if var1 in cat_features and var2 in cat_features:
            # Corrélation entre variables catégorielles (V de Cramér)
            corr_matrix.loc[var1,var2] = cramers_v(df[var1], df[var2])
        elif var1 in num_features+['HeartDisease'] and var2 in num_features+['HeartDisease']:
            # Corrélation entre variables numériques (Pearson)
            corr_matrix.loc[var1,var2] = df[var1].corr(df[var2])
        else:
            # Corrélation entre variables numériques et catégorielles (Tau de Kendall)
            if var1 in cat_features and var2 in num_features+['HeartDisease']:
                corr_matrix.loc[var1,var2] = kendalltau(df[var1].astype('category').cat.codes, df[var2])[0]
            else:
                corr_matrix.loc[var1,var2] = kendalltau(df[var2].astype('category').cat.codes, df[var1])[0]

corr_matrix = corr_matrix.astype(float)

# Visualisation de la matrice de corrélation
plt.figure(figsize=(14,12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Matrice de corrélation mixte (numérique + catégorielle)")
plt.show()


# In[266]:


# Identification des variables fortement corrélées entre elles
# On crée une copie de la matrice sans la diagonale pour trouver les corrélations élevées
corr_matrix_no_diag = corr_matrix.copy()
np.fill_diagonal(corr_matrix_no_diag.values, 0)
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(corr_matrix_no_diag['HeartDisease'])
# Seuil de corrélation élevée
high_corr_threshold = 0.6
high_corr_pairs = []

# Trouver les paires de variables fortement corrélées
for i in range(len(corr_matrix_no_diag.columns)):
    for j in range(i+1, len(corr_matrix_no_diag.columns)):
        if abs(corr_matrix_no_diag.iloc[i, j]) >= high_corr_threshold:
            var1 = corr_matrix_no_diag.index[i]
            var2 = corr_matrix_no_diag.columns[j]
            correlation = corr_matrix_no_diag.iloc[i, j]
            high_corr_pairs.append((var1, var2, correlation))

# Afficher les paires fortement corrélées
if high_corr_pairs:
    print("\nVariables fortement corrélées entre elles (|corr| >= 0.6):")
    for var1, var2, corr in high_corr_pairs:
        print(f"- {var1} et {var2}: {corr:.3f}")
else:
    print("\nAucune paire de variables fortement corrélées n'a été trouvée (seuil: 0.7)")


# ### Sélection des variables à conserver en éliminant les redondances

# In[267]:


# 1. Sélection des variables corrélées avec HeartDisease
corr_with_target = corr_matrix['HeartDisease'].drop('HeartDisease').abs()
important_features = corr_with_target[corr_with_target > 0.2].index.tolist()
print(f"\nVariables corrélées avec HeartDisease (|corr| > 0.2): {important_features}")

# 2. Élimination des variables redondantes
features_to_remove = set()

for var1, var2, _ in high_corr_pairs:
    # Si les deux variables sont importantes pour la prédiction
    if var1 in important_features and var2 in important_features:
        # On garde celle qui a la plus forte corrélation avec la cible
        if abs(corr_matrix.loc[var1, 'HeartDisease']) >= abs(corr_matrix.loc[var2, 'HeartDisease']):
            features_to_remove.add(var2)
        else:
            features_to_remove.add(var1)
    # Si une seule est importante, on supprime l'autre
    elif var1 in important_features:
        features_to_remove.add(var2)
    elif var2 in important_features:
        features_to_remove.add(var1)
    # Si aucune n'est importante mais qu'elles sont fortement corrélées entre elles
    else:
        # On garde arbitrairement la première
        features_to_remove.add(var2)

# Variables finales à conserver
final_features = [f for f in df.columns if f != 'HeartDisease' and f not in features_to_remove]
print(f"\nVariables sélectionnées après élimination des redondances: {final_features}")


# ### Visualisations des variables sélectionnées par rapport à HeartDisease

# In[268]:


for feature in final_features:
    if feature != 'HeartDisease' and feature in num_features:  # Pour les variables numériques
        plt.figure(figsize=(12, 5))

        # Boxplot
        plt.subplot(1, 2, 1)
        sns.boxplot(x='HeartDisease', y=feature, data=df)
        plt.title(f'Boxplot de {feature} en fonction de HeartDisease')

        # Violin plot
        plt.subplot(1, 2, 2)
        sns.violinplot(x='HeartDisease', y=feature, data=df)
        plt.title(f'Violin Plot de {feature} en fonction de HeartDisease')

        plt.tight_layout()
        plt.show()
    elif feature != 'HeartDisease' and feature in cat_features:  # Pour les variables catégorielles
        plt.figure(figsize=(10, 6))
        contingency_table = pd.crosstab(df[feature], df['HeartDisease'])
        contingency_table_percentage = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

        contingency_table_percentage.plot(kind='bar', stacked=True)
        plt.title(f'Distribution de {feature} selon HeartDisease')
        plt.xlabel(feature)
        plt.ylabel('Pourcentage (%)')
        plt.legend(title='HeartDisease', labels=['0 (Non)', '1 (Oui)'])
        plt.tight_layout()
        plt.show()


# ## 1) PRÉPARATION DES DONNÉES POUR LA MODÉLISATION

# ### 1. Séparation des features et de la variable cible

# In[269]:


X = df[final_features]  # Utiliser uniquement les variables sélectionnées
y = df['HeartDisease']


# ### 2. Identification des types de variables

# In[270]:


categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\nVariables catégorielles sélectionnées:", categorical_features)
print("Variables numériques sélectionnées:", numerical_features)

# 3. DIVISION DES DONNÉES EN ENSEMBLES D'ENTRAÎNEMENT ET DE TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


# ### 3. Préparation du pipeline de traitement

# In[271]:


# Pour les variables catégorielles avec ordre médical naturel (ST_Slope)
# Up (ascendant) < Flat (plat) < Down (descendant) en termes de risque cardiaque
st_slope_categories = ['Up', 'Flat', 'Down']
ordinal_features = ['ST_Slope'] if 'ST_Slope' in categorical_features else []

# Séparation des variables catégorielles sans ordre et avec ordre
categorical_no_order = [feat for feat in categorical_features if feat not in ordinal_features]

# Pipeline pour les variables catégorielles sans ordre (one-hot encoding)
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

# Pipeline pour les variables ordinales (ST_Slope)
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(categories=[st_slope_categories]))
]) if ordinal_features else None

# Pipeline pour les variables numériques avec KNNImputer comme demandé
numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),  # Utilisation de KNNImputer au lieu de la médiane
    ('scaler', StandardScaler())
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


# ### 4. APPLICATION DU PRÉTRAITEMENT

# In[272]:


X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Récupération des noms des colonnes après transformation
transformed_feature_names = []

# Ajout des colonnes numériques (gardent le même nom)
transformed_feature_names.extend(numerical_features)

# Ajout des colonnes catégorielles avec one-hot encoding
if categorical_no_order:
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    for i, feat_name in enumerate(categorical_no_order):
        categories = ohe.categories_[i][1:]  # drop='first' donc on commence à l'indice 1
        for category in categories:
            transformed_feature_names.append(f"{feat_name}_{category}")

# Ajout des colonnes ordinales (gardent le même nom)
if ordinal_features:
    transformed_feature_names.extend(ordinal_features)

print("\nNoms des colonnes après transformation:", transformed_feature_names)

# Création d'un DataFrame avec les données transformées pour inspection
X_train_df = pd.DataFrame(X_train_transformed, columns=transformed_feature_names)
print("\nAperçu du DataFrame après transformation:")
print(X_train_df.head())

# Statistiques du DataFrame transformé
print("\nStatistiques descriptives du DataFrame transformé:")
print(X_train_df.describe())


# ### 5. Recherche du meilleur K

# In[273]:


# Évaluation pour différentes valeurs de k (accuracy et f1-score)
k_values = range(1, 31, 2)  # k impairs de 1 à 29
accuracies = []
f1_scores = []

print("\nRecherche du meilleur k pour KNN...")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_results = cross_val_score(knn, X_train_transformed, y_train, cv=5, scoring='accuracy')
    accuracies.append(cv_results.mean())

    cv_results_f1 = cross_val_score(knn, X_train_transformed, y_train, cv=5, scoring='f1')
    f1_scores.append(cv_results_f1.mean())
    print(f"k={k}: Accuracy={cv_results.mean():.4f}, F1-score={cv_results_f1.mean():.4f}")

# Afficher le k optimal selon les deux métriques
best_k_accuracy = k_values[np.argmax(accuracies)]
best_k_f1 = k_values[np.argmax(f1_scores)]

print(f"\nMeilleur k pour l'accuracy: {best_k_accuracy} (accuracy: {max(accuracies):.4f})")
print(f"Meilleur k pour le F1-score: {best_k_f1} (F1-score: {max(f1_scores):.4f})")

# Validation avec GridSearchCV pour une recherche plus exhaustive
param_grid = {'n_neighbors': range(1, 31, 2)}
knn = KNeighborsClassifier()

# GridSearch avec F1-score comme métrique
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_transformed, y_train)

best_k_grid = grid_search.best_params_['n_neighbors']
print(f"\nMeilleur k selon GridSearchCV: {best_k_grid}")
print(f"Meilleur F1-score CV: {grid_search.best_score_:.4f}")

# On utilise le meilleur k trouvé par GridSearchCV
best_k = best_k_grid


# ### 6. ENTRAÎNEMENT ET ÉVALUATION DU MODÈLE KNN

# In[274]:


# Modèle KNN avec le meilleur k
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_transformed, y_train)

# Prédictions
y_pred_knn = best_knn.predict(X_test_transformed)
y_prob_knn = best_knn.predict_proba(X_test_transformed)[:, 1]

# Évaluation du modèle
print("\nÉvaluation du modèle KNN:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_knn):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_knn):.4f}")

print("\nRapport de classification KNN:\n", classification_report(y_test, y_pred_knn))


# ### 7. IMPLÉMENTATION DU MODÈLE XGBOOST en prenant en compte les variables catégorielles

# In[275]:


# Préparation des données pour XGBoost - sans one-hot encoding
X_train_xgb = X_train.copy()
X_test_xgb = X_test.copy()

# Encodage des variables catégorielles pour XGBoost (qui nécessite des valeurs numériques)
for col in categorical_features:
    if col in ordinal_features:
        # Encodage ordinal pour les variables avec ordre naturel
        encoder = OrdinalEncoder(categories=[st_slope_categories] if col == 'ST_Slope' else None)
        X_train_xgb[col] = encoder.fit_transform(X_train_xgb[[col]])
        X_test_xgb[col] = encoder.transform(X_test_xgb[[col]])
    else:
        # Encodage numérique simple pour les autres variables catégorielles
        encoder = OrdinalEncoder()
        X_train_xgb[col] = encoder.fit_transform(X_train_xgb[[col]])
        X_test_xgb[col] = encoder.transform(X_test_xgb[[col]])

# Imputation des valeurs manquantes avec KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_train_xgb = pd.DataFrame(imputer.fit_transform(X_train_xgb), columns=X_train_xgb.columns)
X_test_xgb = pd.DataFrame(imputer.transform(X_test_xgb), columns=X_test_xgb.columns)

# Liste des indices des colonnes catégorielles après transformation
categorical_indices = [X_train_xgb.columns.get_loc(col) for col in categorical_features]

print("\nAperçu des données préparées pour XGBoost:")
print(X_train_xgb.head())

# Entraînement du modèle XGBoost avec feature_types
# Utiliser 'q' pour numérique et 'c' pour catégoriel selon la documentation XGBoost
xgb_model = xgb.XGBClassifier(
    enable_categorical=True,  # Activer le support des variables catégorielles
    feature_types=['q' if i not in categorical_indices else 'c' for i in range(X_train_xgb.shape[1])],
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

# Fit du modèle
xgb_model.fit(X_train_xgb, y_train)

# Prédictions
y_pred_xgb = xgb_model.predict(X_test_xgb)
y_prob_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]

# Évaluation du modèle XGBoost
print("\nÉvaluation du modèle XGBoost (baseline):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_xgb):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_xgb):.4f}")

print("\nRapport de classification XGBoost (baseline):\n", classification_report(y_test, y_pred_xgb))

# Matrice de confusion pour XGBoost 
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
tn, fp, fn, tp = cm_xgb.ravel()
print(f"\nMatrice de confusion XGBoost :")
print(f"Vrais Négatifs (TN): {tn}")
print(f"Faux Positifs (FP): {fp}")
print(f"Faux Négatifs (FN): {fn}")
print(f"Vrais Positifs (TP): {tp}")
print(f"Taux de faux négatifs: {fn/(fn+tp):.4f} ({fn} sur {fn+tp} cas positifs)")


# ### 8.1 AMÉLIORATION DU MODÈLE XGBOOST

# In[276]:


# Optimisation des hyperparamètres avec GridSearchCV pour améliorer spécifiquement la détection des cas positifs
print("\nOptimisation des hyperparamètres de XGBoost pour réduire les faux négatifs...")

# Définir les paramètres à optimiser
param_grid_xgb = {
    'scale_pos_weight': [1, 2, 3],  # Donne plus de poids aux cas positifs
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'learning_rate': [0.05, 0.1, 0.2]
}

# Nous utilisons f1_score comme métrique d'optimisation pour équilibrer précision et rappel
grid_search_xgb = GridSearchCV(
    estimator=xgb.XGBClassifier(
        enable_categorical=True,
        feature_types=['q' if i not in categorical_indices else 'c' for i in range(X_train_xgb.shape[1])],
        objective='binary:logistic',
        random_state=42
    ),
    param_grid=param_grid_xgb,
    scoring='f1',  # Optimiser pour F1-score qui tient compte des faux négatifs
    cv=5,
    verbose=1
)

grid_search_xgb.fit(X_train_xgb, y_train)

# Meilleurs paramètres trouvés
print(f"Meilleurs paramètres: {grid_search_xgb.best_params_}")
print(f"Meilleur F1-score (CV): {grid_search_xgb.best_score_:.4f}")

# Appliquer les meilleurs paramètres au modèle XGBoost
best_xgb_model = grid_search_xgb.best_estimator_

# Prédictions avec le modèle optimisé
y_pred_xgb_optimized = best_xgb_model.predict(X_test_xgb)
y_prob_xgb_optimized = best_xgb_model.predict_proba(X_test_xgb)[:, 1]

# Évaluation du modèle XGBoost optimisé
print("\nÉvaluation du modèle XGBoost optimisé:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb_optimized):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_xgb_optimized):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_xgb_optimized):.4f}")

print("\nRapport de classification XGBoost optimisé:\n", classification_report(y_test, y_pred_xgb_optimized))

# Matrice de confusion pour XGBoost optimisé
cm_xgb_optimized = confusion_matrix(y_test, y_pred_xgb_optimized)
tn_opt, fp_opt, fn_opt, tp_opt = cm_xgb_optimized.ravel()
print(f"\nMatrice de confusion XGBoost optimisé:")
print(f"Vrais Négatifs (TN): {tn_opt}")
print(f"Faux Positifs (FP): {fp_opt}")
print(f"Faux Négatifs (FN): {fn_opt}")
print(f"Vrais Positifs (TP): {tp_opt}")
print(f"Taux de faux négatifs: {fn_opt/(fn_opt+tp_opt):.4f} ({fn_opt} sur {fn_opt+tp_opt} cas positifs)")


# ### 8.2 ANALYSE PAR SEXE

# In[277]:


# Récupérer le sexe depuis le jeu de données initial
test_indices = y_test.index
sex_in_test = df.loc[test_indices, 'Sex']

# Créer un DataFrame avec les résultats de prédiction et le sexe
results_df = pd.DataFrame({
    'Sex': sex_in_test,
    'TrueLabel': y_test,
    'Predicted_Baseline': y_pred_xgb,
    'Predicted_Optimized': y_pred_xgb_optimized
})

# Analyse des performances par sexe pour le modèle de base
male_results_baseline = results_df[results_df['Sex'] == 'M']
female_results_baseline = results_df[results_df['Sex'] == 'F']

male_accuracy_baseline = accuracy_score(male_results_baseline['TrueLabel'], male_results_baseline['Predicted_Baseline'])
female_accuracy_baseline = accuracy_score(female_results_baseline['TrueLabel'], female_results_baseline['Predicted_Baseline'])

male_f1_baseline = f1_score(male_results_baseline['TrueLabel'], male_results_baseline['Predicted_Baseline'])
female_f1_baseline = f1_score(female_results_baseline['TrueLabel'], female_results_baseline['Predicted_Baseline'])

# Analyse des performances par sexe pour le modèle optimisé
male_accuracy_optimized = accuracy_score(male_results_baseline['TrueLabel'], male_results_baseline['Predicted_Optimized'])
female_accuracy_optimized = accuracy_score(female_results_baseline['TrueLabel'], female_results_baseline['Predicted_Optimized'])

male_f1_optimized = f1_score(male_results_baseline['TrueLabel'], male_results_baseline['Predicted_Optimized'])
female_f1_optimized = f1_score(female_results_baseline['TrueLabel'], female_results_baseline['Predicted_Optimized'])

print("\nPerformances du modèle par sexe:")
print("\nModèle de base:")
print(f"Hommes (n={len(male_results_baseline)}) - Accuracy: {male_accuracy_baseline:.4f}, F1-score: {male_f1_baseline:.4f}")
print(f"Femmes (n={len(female_results_baseline)}) - Accuracy: {female_accuracy_baseline:.4f}, F1-score: {female_f1_baseline:.4f}")

print("\nModèle optimisé:")
print(f"Hommes (n={len(male_results_baseline)}) - Accuracy: {male_accuracy_optimized:.4f}, F1-score: {male_f1_optimized:.4f}")
print(f"Femmes (n={len(female_results_baseline)}) - Accuracy: {female_accuracy_optimized:.4f}, F1-score: {female_f1_optimized:.4f}")

# Calcul des taux de faux négatifs par sexe pour le modèle de base
male_cm_baseline = confusion_matrix(male_results_baseline['TrueLabel'], male_results_baseline['Predicted_Baseline'])
female_cm_baseline = confusion_matrix(female_results_baseline['TrueLabel'], female_results_baseline['Predicted_Baseline'])

male_tn_base, male_fp_base, male_fn_base, male_tp_base = male_cm_baseline.ravel()
female_tn_base, female_fp_base, female_fn_base, female_tp_base = female_cm_baseline.ravel()

male_fnr_base = male_fn_base / (male_fn_base + male_tp_base) if (male_fn_base + male_tp_base) > 0 else 0
female_fnr_base = female_fn_base / (female_fn_base + female_tp_base) if (female_fn_base + female_tp_base) > 0 else 0

# Calcul des taux de faux négatifs par sexe pour le modèle optimisé
male_cm_opt = confusion_matrix(male_results_baseline['TrueLabel'], male_results_baseline['Predicted_Optimized'])
female_cm_opt = confusion_matrix(female_results_baseline['TrueLabel'], female_results_baseline['Predicted_Optimized'])

male_tn_opt, male_fp_opt, male_fn_opt, male_tp_opt = male_cm_opt.ravel()
female_tn_opt, female_fp_opt, female_fn_opt, female_tp_opt = female_cm_opt.ravel()

male_fnr_opt = male_fn_opt / (male_fn_opt + male_tp_opt) if (male_fn_opt + male_tp_opt) > 0 else 0
female_fnr_opt = female_fn_opt / (female_fn_opt + female_tp_opt) if (female_fn_opt + female_tp_opt) > 0 else 0

print("\nTaux de faux négatifs par sexe:")
print(f"Modèle de base - Hommes: {male_fnr_base:.4f}, Femmes: {female_fnr_base:.4f}")
print(f"Modèle optimisé - Hommes: {male_fnr_opt:.4f}, Femmes: {female_fnr_opt:.4f}")


# ### 8.3 VISUALISATIONS

# In[278]:


# Plot des matrices de confusion avec annotations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Matrice de confusion du modèle baseline global
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Matrice de confusion - XGBoost Baseline')
axes[0, 0].set_xlabel('Prédictions')
axes[0, 0].set_ylabel('Valeurs réelles')
axes[0, 0].xaxis.set_ticklabels(['Négatif', 'Positif'])
axes[0, 0].yaxis.set_ticklabels(['Négatif', 'Positif'])

# Matrice de confusion du modèle optimisé global
sns.heatmap(cm_xgb_optimized, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Matrice de confusion - XGBoost Optimisé')
axes[0, 1].set_xlabel('Prédictions')
axes[0, 1].set_ylabel('Valeurs réelles')
axes[0, 1].xaxis.set_ticklabels(['Négatif', 'Positif'])
axes[0, 1].yaxis.set_ticklabels(['Négatif', 'Positif'])

# Comparaison des performances par sexe (accuracies)
gender_accuracies = {
    'Hommes - Baseline': male_accuracy_baseline,
    'Femmes - Baseline': female_accuracy_baseline,
    'Hommes - Optimisé': male_accuracy_optimized,
    'Femmes - Optimisé': female_accuracy_optimized
}
sns.barplot(x=list(gender_accuracies.keys()), y=list(gender_accuracies.values()), ax=axes[1, 0])
axes[1, 0].set_title('Accuracies par sexe')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_xticklabels(gender_accuracies.keys(), rotation=45)
for i, v in enumerate(gender_accuracies.values()):
    axes[1, 0].text(i, v + 0.01, f"{v:.4f}", ha='center')

# Comparaison des taux de faux négatifs par sexe
fnr_values = {
    'Hommes - Baseline': male_fnr_base,
    'Femmes - Baseline': female_fnr_base,
    'Hommes - Optimisé': male_fnr_opt,
    'Femmes - Optimisé': female_fnr_opt
}
sns.barplot(x=list(fnr_values.keys()), y=list(fnr_values.values()), ax=axes[1, 1])
axes[1, 1].set_title('Taux de faux négatifs par sexe')
axes[1, 1].set_ylabel('Taux de faux négatifs')
axes[1, 1].set_xticklabels(fnr_values.keys(), rotation=45)
for i, v in enumerate(fnr_values.values()):
    axes[1, 1].text(i, v + 0.01, f"{v:.4f}", ha='center')

plt.tight_layout()
plt.show()

# Courbes ROC pour comparer les modèles
plt.figure(figsize=(10, 8))
fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_xgb)
fpr_opt, tpr_opt, _ = roc_curve(y_test, y_prob_xgb_optimized)

plt.plot(fpr_base, tpr_base, label=f'XGBoost Baseline (AUC = {roc_auc_score(y_test, y_prob_xgb):.3f})')
plt.plot(fpr_opt, tpr_opt, label=f'XGBoost Optimisé (AUC = {roc_auc_score(y_test, y_prob_xgb_optimized):.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbes ROC - Comparaison des modèles XGBoost')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# Comparaison visuelle des distributions des probabilités de prédiction
plt.figure(figsize=(12, 6))

# Créer un dataframe pour faciliter la visualisation
probs_df = pd.DataFrame({
    'Probabilité': y_prob_xgb_optimized,
    'Classe réelle': y_test,
    'Sexe': sex_in_test
})

# Distribution des probabilités par classe réelle
plt.subplot(1, 2, 1)
sns.histplot(data=probs_df, x='Probabilité', hue='Classe réelle', bins=20, alpha=0.6)
plt.title('Distribution des probabilités par classe réelle')
plt.axvline(x=0.5, color='red', linestyle='--')
plt.xlabel('Probabilité prédite (classe positive)')
plt.ylabel('Nombre d\'observations')

# Distribution des probabilités par sexe
plt.subplot(1, 2, 2)
sns.histplot(data=probs_df, x='Probabilité', hue='Sexe', bins=20, alpha=0.6)
plt.title('Distribution des probabilités par sexe')
plt.axvline(x=0.5, color='red', linestyle='--')
plt.xlabel('Probabilité prédite (classe positive)')
plt.ylabel('Nombre d\'observations')

plt.tight_layout()
plt.show()


# ### 9. COMPARAISON DES PERFORMANCES DES MODÈLES

# In[279]:


print("\nComparaison des performances:")
print(f"KNN (k={best_k}) - Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}, F1-score: {f1_score(y_test, y_pred_knn):.4f}, AUC-ROC: {roc_auc_score(y_test, y_prob_knn):.4f}")
print(f"XGBoost - Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}, F1-score: {f1_score(y_test, y_pred_xgb):.4f}, AUC-ROC: {roc_auc_score(y_test, y_prob_xgb):.4f}")

# 10. IMPORTANCE DES VARIABLES POUR XGBOOST

# Obtenir l'importance des variables pour XGBoost
feature_importance = xgb_model.feature_importances_
sorted_idx = np.argsort(feature_importance)

print("\nImportance des variables selon XGBoost:")
for i in sorted_idx[::-1]:
    print(f"{X_train_xgb.columns[i]}: {feature_importance[i]:.4f}")


# In[280]:


if __name__ == "__main__":
    pass  

