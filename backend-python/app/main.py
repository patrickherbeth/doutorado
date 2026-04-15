from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Request(BaseModel):
    pergunta: str

def processar(texto):
    artigo = re.findall(r"artigo\s*\d+", texto.lower())
    return f"Processado: {texto} | Artigos: {artigo}"

@app.post("/api/keller/consulta")
def consulta(req: Request):
    return {"resposta": processar(req.pergunta)}
