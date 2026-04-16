# =====================================================
# 🧠 KELLER-BR – VERSÃO FIEL AO ARTIGO (FAISS LEVE)
# =====================================================

import re
import torch
import faiss
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

# =====================================================
# CONFIG
# =====================================================
APP_TITLE = "KELLER-BR (FAISS TEST)"
FRONTEND_ORIGIN = "http://localhost:4200"

LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
ENCODER_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

TOP_K = 5
MAX_DOCS_FAISS = 5000

# =====================================================
# FASTAPI
# =====================================================
app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# DTO
# =====================================================
class ConsultaRequest(BaseModel):
    pergunta: str = Field(...)

# =====================================================
# DATASETS
# =====================================================
print("\n🔹 ================= DATASETS =================")

ds_br = load_dataset("joelniklaus/brazilian_court_decisions", split="train")
ds_stf = load_dataset("celsowm/jurisprudencias_stf", split="train")
ds_tjmg = load_dataset("celsowm/jurisprudencias_tjmg", split="train")
ds_stj = load_dataset("celsowm/jurisprudencias_stj", split="train")

DATASET_LOCAL = list(ds_br) + list(ds_stf) + list(ds_tjmg) + list(ds_stj)

print(f"✅ TOTAL DOCUMENTOS: {len(DATASET_LOCAL)}")

# =====================================================
# MODELOS
# =====================================================
print("🔹 Carregando modelos...")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map="cpu")
encoder = SentenceTransformer(ENCODER_MODEL_NAME)

print("✅ Modelos carregados")

# =====================================================
# EXTRAÇÃO
# =====================================================
def extrair_texto(item):
    return re.sub(r"\s+", " ", str(
        item.get("texto") or
        item.get("conteudo") or
        item.get("documento") or
        item.get("ementa") or
        item.get("decisao") or
        item.get("text") or
        item
    )).strip()

def extrair_numero(item, idx):
    return (
        item.get("numero_processo") or
        item.get("processo") or
        item.get("id") or
        f"DOC_{idx}"
    )

# =====================================================
# FAISS (LEVE)
# =====================================================
print("\n🔹 ================= FAISS =================")

TEXTOS = []
NUMEROS = []

for idx, item in enumerate(DATASET_LOCAL):
    TEXTOS.append(extrair_texto(item))
    NUMEROS.append(extrair_numero(item, idx))

    if idx >= MAX_DOCS_FAISS:
        break

print(f"📊 Documentos indexados: {len(TEXTOS)}")

print("🔹 Gerando embeddings FAISS...")
embeddings = encoder.encode(TEXTOS, convert_to_numpy=True, show_progress_bar=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print("✅ FAISS pronto")

# =====================================================
# LLM
# =====================================================
def gerar_texto_llm(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm.generate(**inputs, max_new_tokens=120, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =====================================================
# SUBFATOS (IGUAL AO PAPER)
# =====================================================
def gerar_subfatos(texto):

    prompt = f"""
Quebre o texto em fatos jurídicos curtos:

Texto:
{texto}

Responda em lista.
"""

    saida = gerar_texto_llm(prompt)

    linhas = [l.strip() for l in saida.split("\n") if len(l.strip()) > 10]

    if not linhas:
        linhas = [texto[:300]]

    return linhas[:5]

# =====================================================
# EMBEDDING
# =====================================================
def embed(textos):
    return encoder.encode(textos, convert_to_tensor=True)

# =====================================================
# SCORE (IGUAL AO PAPER)
# =====================================================
def score(q_emb, d_emb):

    sim = util.cos_sim(q_emb, d_emb)

    return float(torch.sum(torch.max(sim, dim=1).values))

# =====================================================
# RETRIEVE (FAISS REAL)
# =====================================================
def retrieve(pergunta):

    q_emb = encoder.encode([pergunta], convert_to_numpy=True)

    distances, indices = index.search(q_emb, TOP_K * 3)

    docs = []

    for i in indices[0]:
        docs.append({
            "numero": NUMEROS[i],
            "texto": TEXTOS[i]
        })

    return docs

# =====================================================
# PIPELINE (FIEL AO ARTIGO)
# =====================================================
def pipeline(pergunta):

    print("\n================ PIPELINE =================")

    # 1. Subfatos query
    q_sub = gerar_subfatos(pergunta)
    q_emb = embed(q_sub)

    # 2. Retrieve
    docs = retrieve(pergunta)

    resultados = []

    for d in docs:

        # 3. Subfatos documento
        d_sub = gerar_subfatos(d["texto"])
        d_emb = embed(d_sub)

        # 4. Score cruzado
        s = score(q_emb, d_emb)

        resultados.append({
            "processo": d["numero"],
            "score": s
        })

    # 5. Ranking
    ranking = sorted(resultados, key=lambda x: x["score"], reverse=True)[:TOP_K]

    return ranking

# =====================================================
# ENDPOINT
# =====================================================
@app.post("/api/keller/consulta")
def consultar(req: ConsultaRequest):
    try:
        return pipeline(req.pergunta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}