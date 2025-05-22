import neuralModel
import nbimporter
from DataCleaning import predict_model

def predict(input_dict, model_name):
    if model_name == "knn":
        return predict_model("newPatient.csv", "knn")
    elif model_name == "xgboost":
        return predict_model("newPatient.csv", "xgb")
    elif model_name == "xgboost-opt":
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
        print(predict({}, "knn"))
    except Exception as e:
        print(f"Erreur KNN : {e}")

    try:
        print("[XGBOOST]")
        print(predict({}, "xgboost"))
    except Exception as e:
        print(f"Erreur XGBOOST : {e}")

    try:
        print("[XGBOOST OPT]")
        print(predict({}, "xgboost-opt"))
    except Exception as e:
        print(f"Erreur XGBOOST OPT : {e}")

    try:
        print("[NEURAL]")
        print(predict({}, "neural"))
    except Exception as e:
        print(f"Erreur NEURAL : {e}")
