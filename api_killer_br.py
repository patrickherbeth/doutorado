# =====================================================
# 🧠 KELLER-BR – BACKEND DE INFERÊNCIA ONLINE (DEBUG FULL)
# =====================================================

import os
import re
import time
import torch
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

# =====================================================
# CONFIG
# =====================================================
APP_TITLE = "KELLER-BR API DEBUG"
FRONTEND_ORIGIN = "http://localhost:4200"

LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
ENCODER_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

TOP_K_DATAJUD = 8

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
# BASE JURÍDICA
# =====================================================
BASE_JURIDICA_ARTIGO_CRIME = {
    "artigo 155": "furto",
    "artigo 157": "roubo",
}

# =====================================================
# DATASETS
# =====================================================
print("\n🔹 ================= DATASETS =================")

print("🔹 Carregando brazilian_court_decisions...")
ds_br = load_dataset("joelniklaus/brazilian_court_decisions", split="train")

print("🔹 Carregando STF...")
ds_stf = load_dataset("celsowm/jurisprudencias_stf", split="train")

print("🔹 Carregando TJMG...")
ds_tjmg = load_dataset("celsowm/jurisprudencias_tjmg", split="train")

print("🔹 Carregando STJ...")
ds_stj = load_dataset("celsowm/jurisprudencias_stj", split="train")

print("🔹 Unificando datasets...")

DATASET_LOCAL = list(ds_br) + list(ds_stf) + list(ds_tjmg) + list(ds_stj)

print(f"✅ TOTAL DOCUMENTOS: {len(DATASET_LOCAL)}")
print("🔹 ==========================================\n")

# =====================================================
# MODELOS
# =====================================================
print("🔹 Carregando modelos...")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map="cpu")

encoder = SentenceTransformer(ENCODER_MODEL_NAME)

print("✅ Modelos carregados")

# =====================================================
# UTILS
# =====================================================
def normalizar(texto):
    return re.sub(r"\s+", " ", texto or "").strip()

# =====================================================
# LLM
# =====================================================
def gerar_texto_llm(prompt: str):
    print("\n🧠 PROMPT LLM:")
    print(prompt[:300])

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = llm.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False
    )

    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n🧠 RESPOSTA LLM:")
    print(resposta)

    return resposta

# =====================================================
# EXTRAÇÃO
# =====================================================
def extrair_crimes_artigos(texto):
    print("\n🔍 EXTRAÇÃO DE CRIMES/ARTIGOS")

    prompt = f"""
Liste crimes e artigos do texto:

Texto:
{texto}

Formato:
Crimes:
Artigos:
"""

    saida = gerar_texto_llm(prompt)

    crimes = []
    artigos = []

    for linha in saida.split("\n"):
        if "crime" in linha.lower():
            crimes = linha.split(":")[-1].split(";")
        if "artigo" in linha.lower():
            artigos = linha.split(":")[-1].split(";")

    crimes = [c.strip().lower() for c in crimes if c.strip()]
    artigos = [a.strip().lower() for a in artigos if a.strip()]

    print("🔎 Crimes:", crimes)
    print("🔎 Artigos:", artigos)

    return crimes, artigos

# =====================================================
# SUBFATOS
# =====================================================
def gerar_subfatos(texto):
    print("\n📌 GERANDO SUBFATOS")

    crimes, artigos = extrair_crimes_artigos(texto)

    subfatos = []

    # 🔥 CASO NORMAL
    for c in crimes:
        sub = f"Crime: {c}. Fato: {texto[:200]}"
        subfatos.append(sub)

    # 🔥 CORREÇÃO CRÍTICA
    if not subfatos:
        print("⚠️ Nenhum subfato gerado - usando fallback")
        subfatos = [texto[:300]]

    print("📌 Subfatos:", subfatos)

    return subfatos

# =====================================================
# EMBEDDING
# =====================================================
def embedding(textos):
    print("\n🧠 GERANDO EMBEDDING")
    print("Entrada:", textos)

    if not textos:
        print("⚠️ Lista vazia - retornando tensor dummy")
        return torch.zeros((1, 768))

    emb = encoder.encode(textos, convert_to_tensor=True)

    print("📊 Shape:", emb.shape)

    return emb

# =====================================================
# SCORE
# =====================================================
def score(q, d):
    print("\n📈 CALCULANDO SCORE")

    if d.shape[0] == 0:
        print("⚠️ Embedding documento vazio - score 0")
        return 0.0

    sim = util.cos_sim(q, d)
    val = float(torch.sum(torch.max(sim, dim=1).values))

    print("📊 Score:", val)

    return val

# =====================================================
# BUSCA (SUBSTITUI DATAJUD)
# =====================================================
def buscar_datajud(pergunta):
    print("\n🔎 BUSCA LOCAL NOS DATASETS")
    print("Pergunta:", pergunta)

    resultados = []

    for idx, item in enumerate(DATASET_LOCAL):
        try:
            texto = str(item)

            numero = (
                item.get("numero_processo") or
                item.get("processo") or
                item.get("id") or
                f"DOC_{idx}"
            )

            resultados.append({
                "numero": numero,
                "texto": texto
            })

        except Exception as e:
            print(f"⚠️ ERRO ITEM {idx}: {e}")

        if len(resultados) >= TOP_K_DATAJUD * 50:
            break

    print("📊 Total analisado:", len(resultados))

    final = resultados[:TOP_K_DATAJUD]

    print("📌 Retorno final:")
    for d in final:
        print("➡️", d["numero"])

    return final

# =====================================================
# PIPELINE
# =====================================================
def pipeline(pergunta):
    print("\n================ PIPELINE =================")
    print("Pergunta:", pergunta)

    print("\n🔹 ETAPA 1 - SUBFATOS QUERY")
    q_sub = gerar_subfatos(pergunta)

    print("\n🔹 ETAPA 2 - EMBEDDING QUERY")
    q_emb = embedding(q_sub)

    print("\n🔹 ETAPA 3 - BUSCA")
    docs = buscar_datajud(pergunta)

    if not docs:
        print("❌ Nenhum resultado")
        return {"erro": "Nenhum resultado"}

    resultados = []

    for i, d in enumerate(docs):
        print(f"\n========== DOC {i+1} ==========")

        print("\n🔹 ETAPA 4 - SUBFATOS DOC")
        d_sub = gerar_subfatos(d["texto"])

        print("\n🔹 ETAPA 5 - EMBEDDING DOC")
        d_emb = embedding(d_sub)

        print("\n🔹 ETAPA 6 - SCORE")
        s = score(q_emb, d_emb)

        resultados.append({
            "processo": d["numero"],
            "score": s
        })

    print("\n🔹 ETAPA 7 - RANKING")
    ranking = sorted(resultados, key=lambda x: x["score"], reverse=True)

    for r in ranking:
        print(f"🏆 {r['processo']} -> {r['score']}")

    print("\n================ FIM PIPELINE =================\n")

    return ranking

# =====================================================
# ENDPOINT
# =====================================================
@app.post("/api/keller/consulta")
def consultar(req: ConsultaRequest):
    print("\n🔥 NOVA CONSULTA:", req.pergunta)

    try:
        return pipeline(req.pergunta)
    except Exception as e:
        print("❌ ERRO:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}