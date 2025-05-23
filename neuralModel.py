
import warnings
from sklearn.exceptions import ConvergenceWarning
import sys
import os
import joblib
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


def predict(predict_file):
    if not os.path.exists("trained_model.pkl"):
        print("🔧 Aucun modèle trouvé. Entraînement automatique en cours...")
        model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', max_iter=500, early_stopping=True, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump({
            "model": model,
            "X_test": X_test,
            "y_test": y_test
        }, "trained_model.pkl")
        print("✅ Modèle entraîné et sauvegardé.")
    else:
        model_data = joblib.load("trained_model.pkl")
        model = model_data["model"]

    if not os.path.exists(predict_file):
        raise FileNotFoundError(f"Fichier de données non trouvé : {predict_file}")
    model_data = joblib.load("trained_model.pkl")
    model = model_data["model"]
    print("🔍 Prédiction à partir du modèle sauvegardé")
    df_pred = pd.read_csv(predict_file)
    df_pred = pd.get_dummies(df_pred, drop_first=True)
    for col in X.columns:
        if col not in df_pred.columns:
            df_pred[col] = 0
    df_pred = df_pred[X.columns]
    df_pred_scaled = scaler.transform(df_pred)
    proba = model.predict_proba(df_pred_scaled)[0][1]
    print(f"🩺 Probabilité d'être malade : {proba:.2%}")
    print(f"🩺 Verdict : {'Malade' if proba >= 0.5 else 'Pas malade'}")
    verdict = "Malade" if proba >= 0.5 else "Pas malade"
    return f"{verdict} (précision : {proba:.2%})"

df = pd.read_csv("data/heart_disease_data.csv")
df = pd.get_dummies(df, drop_first=True)
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


if len(sys.argv) > 1 and sys.argv[1] == "train":
    max_iter = 500
    if len(sys.argv) > 2:
        try:
            max_iter = int(sys.argv[2])
        except ValueError:
            print("❌ Erreur : le deuxième argument doit être un entier (nombre d'itérations)")
            sys.exit(1)
    print(f"🧠 Mode entraînement activé avec max_iter={max_iter}")
    model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', max_iter=max_iter, early_stopping=True, random_state=42)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=ConvergenceWarning)
        model.fit(X_train, y_train)
        converged = True
        for warning in w:
            if issubclass(warning.category, ConvergenceWarning):
                converged = False
                print("⚠️ Attention : l'entraînement n'est pas allé jusqu'à la convergence.")
                break
        if converged:
            print("✅ Convergence atteinte avant l'itération maximale.")
    y_pred = model.predict(X_test)
    f1_new = f1_score(y_test, y_pred)

    # Si aucun modèle sauvegardé n'existe, on sauvegarde directement
    if not os.path.exists("trained_model.pkl"):
        joblib.dump({
            "model": model,
            "X_test": X_test,
            "y_test": y_test
        }, "trained_model.pkl")
        print("📦 Modèle sauvegardé (premier entraînement)")
    else:
        old_data = joblib.load("trained_model.pkl")
        old_model = old_data["model"]
        X_test_saved = old_data["X_test"]
        y_test_saved = old_data["y_test"]
        old_pred = old_model.predict(X_test_saved)
        f1_old = f1_score(y_test_saved, old_pred)
        if f1_new > f1_old:
            joblib.dump({
                "model": model,
                "X_test": X_test,
                "y_test": y_test
            }, "trained_model.pkl")
            print(f"✅ Nouveau modèle meilleur (F1: {f1_new:.2f} > {f1_old:.2f}) → sauvegardé")
        else:
            print(f"❌ Nouveau modèle moins bon (F1: {f1_new:.2f} <= {f1_old:.2f}) → ignoré")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

elif len(sys.argv) > 2 and sys.argv[1] == "predict":
    predict(sys.argv[2])

elif not os.path.exists("trained_model.pkl"):
    raise FileNotFoundError("Aucun modèle trouvé. Lance d'abord le script avec l'argument 'train'")
else:
    model_data = joblib.load("trained_model.pkl")
    model = model_data["model"]
    X_test = model_data["X_test"]
    y_test = model_data["y_test"]
    print("📥 Modèle chargé depuis le fichier")
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))