
def predict_dummy(input_dict):
    return {"model": "dummy", "prediction": 0}

def predict_svm(input_dict):
    return {"model": "svm", "prediction": 1}

def predict_random_forest(input_dict):
    return {"model": "rf", "prediction": 1}

def predict(input_dict, model_name):
    if model_name == "dummy":
        return predict_dummy(input_dict)
    elif model_name == "svm":
        return predict_svm(input_dict)
    elif model_name == "rf":
        return predict_random_forest(input_dict)
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

def main():
    print("Hello world!")

if __name__ == "__main__":
    main()