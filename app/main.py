from fastapi import FastAPI
import joblib
import json
import numpy as np
from pydantic import BaseModel

# Inicializar a API
app = FastAPI(title="Classificação de Acidentes API", version="1.0")

# Carregar o modelo treinado e o encoder
modelo = joblib.load("app/models/modelo_xgboost01.pkl")
encoder = joblib.load("app/models/ordinal_encoder_modelo_01.pkl")

# Carregar o mapeamento de categorias
with open("app/data/mapeamento_categorias.json", "r", encoding="utf-8") as f:
    mapeamento_categorias = json.load(f)


# Classe para validar a entrada de dados da previsão
class AcidenteInput(BaseModel):
    dia_semana: int
    horario: float
    uf: int
    br: int
    km: float
    municipio: int
    causa_acidente: int
    tipo_acidente: int
    fase_dia: int
    sentido_via: int
    condicao_metereologica: int
    tipo_pista: int
    tracado_via: int
    uso_solo: int
    tipo_veiculo: int
    ano_fabricacao_veiculo: int
    estado_fisico: int
    idade: int
    sexo: int


# 📌 Rota para obter o mapeamento das categorias
@app.get("/mapeamento/", summary="Obter o mapeamento de categorias")
def obter_mapeamento():
    return mapeamento_categorias


# 📌 Rota para fazer previsão do modelo
@app.post("/prever/", summary="Fazer previsão da gravidade do acidente")
def prever(dados: AcidenteInput):
    entrada = np.array([[dados.dia_semana, dados.horario, dados.uf, dados.br, dados.km,
                         dados.municipio, dados.causa_acidente, dados.tipo_acidente, dados.fase_dia,
                         dados.sentido_via, dados.condicao_metereologica, dados.tipo_pista, 
                         dados.tracado_via, dados.uso_solo, dados.tipo_veiculo, 
                         dados.ano_fabricacao_veiculo, dados.estado_fisico, dados.idade, dados.sexo]])
    
    previsao = modelo.predict(entrada)
    return {"classificacao_acidente": int(previsao[0])}
