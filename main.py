import neuralModel
import joblib
import pandas as pd

def predict(model_name):
    if model_name == "knn":
        df_new_patient = pd.read_csv('./newPatient.csv')
        preprocessor = joblib.load("models/knn_preprocessor.pkl")
        X_transformed = preprocessor.transform(df_new_patient)
        knn_model = joblib.load("models/knn_model.pkl")
        prediction = knn_model.predict(X_transformed)[0]
        return prediction
    elif model_name == "xgboost":
        return predict("newPatient.csv", "xgb")
    elif model_name == "xgboost_opt":
        return predict("newPatient.csv", "xgb_opt")
    elif model_name == "neural":
        return neuralModel.predict("newPatient.csv")
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

def main():
    print("hello world !")

if __name__ == "__main__":
    print("=== TEST DES MODÈLES ===")
    try:
        print("[KNN]")
        result1 = predict("knn")
    except Exception as e:
        print(f"Erreur KNN : {e}")

    try:
        print("[XGBOOST]")
        result2 = predict("xgboost")
    except Exception as e:
        print(f"Erreur XGBOOST : {e}")

    try:
        print("[XGBOOST OPT]")
        result3 = predict("xgboost_opt")
    except Exception as e:
        print(f"Erreur XGBOOST OPT : {e}")

    try:
        print("[NEURAL]")
        result4 = predict("neural")
    except Exception as e:
        print(f"Erreur NEURAL : {e}")

    print("=== FIN DES TESTS ===")
    print("[PREDICTE]")
    print("knn :", result1 if 'result1' in locals() else "Erreur")
    print("xgboost :", result2 if 'result2' in locals() else "Erreur")
    print("xgboost opt :", result3 if 'result3' in locals() else "Erreur")
    print("neural :", result4 if 'result4' in locals() else "Erreur")