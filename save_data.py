import json
import os
import pickle
import numpy as np

def save_fold_to_json(X_syn, y_syn, tr_idx, val_idx, model_gen, ratio, fold_idx, dataset, base_path="synthetic_data", isTrees = True):

    synthetic_data_type = "trees" if isTrees else "tabnet"
    
    folder = os.path.join(base_path, dataset, synthetic_data_type, model_gen, f"ratio_{ratio}")
    os.makedirs(folder, exist_ok=True)
    
    file_path = os.path.join(folder, f"fold_{fold_idx}.json")
    
    data_to_save = {
        "model_gen": model_gen,
        "ratio": ratio,
        "fold": fold_idx,
        "train_idx": tr_idx.tolist(), 
        "val_idx": val_idx.tolist(),
        "X_synthetic": X_syn.tolist() if hasattr(X_syn, "tolist") else X_syn,
        "y_synthetic": y_syn.tolist() if hasattr(y_syn, "tolist") else y_syn
    }
    
    with open(file_path, 'w') as f:
        json.dump(data_to_save, f)


def load_synthetic_fold(model_gen, dataset, ratio, fold_idx, base_path="synthetic_data", isTrees = True):

    synthetic_data_type = "trees" if isTrees else "tabnet"
    file_path = os.path.join(base_path, dataset, synthetic_data_type, model_gen, f"ratio_{ratio}", f"fold_{fold_idx}.json")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    tr_idx = np.array(data["train_idx"])
    val_idx = np.array(data["val_idx"])
    X_syn = np.array(data["X_synthetic"])
    y_syn = np.array(data["y_synthetic"])
    
    return X_syn, y_syn, tr_idx, val_idx


def save_oof_models(models_list, dataset_name, model_pred, model_gen, path="oof_models"):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, dataset_name, model_gen if model_gen != "" else "No gen model", f"{model_pred}_list.pkl", ), "wb") as f:
        pickle.dump(models_list, f)


def load_oof_models(dataset_name, model_pred, model_gen, path="oof_models"):
    file_path = os.path.join(path, dataset_name, model_gen if model_gen != "" else "No gen model", f"{model_pred}_list.pkl")
    
    with open(file_path, "rb") as f:
        models_list = pickle.load(f)
    
    return models_list





# %%



