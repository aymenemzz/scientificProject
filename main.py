import neuralModel
import nbimporter
from DataCleaning import predict_model

def predict(model_name):
    if model_name == "knn":
        return predict_model("newPatient.csv", "knn")
    elif model_name == "xgboost":
        return predict_model("newPatient.csv", "xgb")
    elif model_name == "xgboost_opt":
        return predict_model("newPatient.csv", "xgb_opt")
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