import joblib
import json

def carregar_modelo():
    return joblib.load("app/models/modelo_xgboost.pkl")

def carregar_encoder():
    return joblib.load("app/models/ordinal_encoder.pkl")

def carregar_mapeamento():
    with open("app/data/mapeamento_categorias.json", "r", encoding="utf-8") as f:
        return json.load(f)
