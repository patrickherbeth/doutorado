# =====================================================
# 🧠 KELLER-BR – BACKEND COM ENDPOINT (VERSÃO REAL)
# Baseado em: Deng et al. (EMNLP 2024)
# =====================================================

import re
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

# =====================================================
# 🚀 FASTAPI INIT
# =====================================================

app = FastAPI(title="KELLER-BR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# 📌 REQUEST DTO
# =====================================================

class ConsultaRequest(BaseModel):
    pergunta: str


# =====================================================
# 📌 1. MODELOS (CARREGA UMA VEZ SÓ)
# =====================================================

print("🔹 Carregando modelos...")

MODEL_CHAT = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHAT)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_CHAT,
    device_map="cpu",
    low_cpu_mem_usage=True
)

# Garante token de padding para geração
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 🔥 BERTimbau
encoder = SentenceTransformer("neuralmind/bert-base-portuguese-cased")

print("✅ Modelos carregados.")


# =====================================================
# 📌 2. BASE JURÍDICA
# =====================================================

BASE_JURIDICA = {
    "artigo 250": "incêndio",
    "artigo 312": "peculato",
    "artigo 155": "furto",
    "artigo 157": "roubo"
}

casos = [
    {"id": 1, "texto": "João ateou fogo em casa. Artigo 250 do Código Penal."},
    {"id": 2, "texto": "Maria desviou dinheiro público. Artigo 312 do Código Penal."},
    {"id": 3, "texto": "José subtraiu celular sem violência. Artigo 155 do Código Penal."}
]


# =====================================================
# 📌 3. EXTRAÇÃO
# =====================================================

def extrair_artigos(texto: str):
    return re.findall(r"artigo\s*\d+", texto.lower())


def mapear_crimes(artigos):
    return [BASE_JURIDICA[a] for a in artigos if a in BASE_JURIDICA]


# =====================================================
# 📌 4. GERAÇÃO DE SUBFATOS (LLM)
# =====================================================

def gerar_subfatos(texto: str):
    artigos = extrair_artigos(texto)
    crimes = mapear_crimes(artigos)

    subfatos = []

    for crime, artigo in zip(crimes, artigos):
        prompt = f"""
Você é um especialista em direito penal.

Gere um subfato jurídico.

REGRAS:
- Máximo 2 linhas
- Objetivo
- Formato:

Crime: X
Fato: Y
Artigo: Z

Crime: {crime}
Artigo: {artigo}

Caso:
{texto}
"""

        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id
        )

        resposta = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        subfatos.append(resposta)

    # Fallback para não quebrar embeddings quando não houver artigo conhecido
    if not subfatos:
        subfatos.append(f"Crime: não identificado\nFato: {texto}\nArtigo: desconhecido")

    return subfatos


# =====================================================
# 📌 5. EMBEDDINGS (BERTIMBAU)
# =====================================================

def gerar_embeddings(subfatos):
    return encoder.encode(
        subfatos,
        convert_to_tensor=True,
        normalize_embeddings=True
    )


# =====================================================
# 📌 6. SIMILARIDADE
# =====================================================

def matriz_similaridade(q_emb, d_emb):
    return util.cos_sim(q_emb, d_emb)


# =====================================================
# 📌 7. SCORE KELLER
# =====================================================

def score_keller(q_emb, d_emb):
    sim_matrix = matriz_similaridade(q_emb, d_emb)
    max_vals = torch.max(sim_matrix, dim=1).values
    return torch.sum(max_vals).item()


# =====================================================
# 📌 8. PIPELINE COMPLETO
# =====================================================

def pipeline_keller(query: str):
    q_subfatos = gerar_subfatos(query)
    q_emb = gerar_embeddings(q_subfatos)

    resultados = []

    for caso in casos:
        d_subfatos = gerar_subfatos(caso["texto"])
        d_emb = gerar_embeddings(d_subfatos)

        score = score_keller(q_emb, d_emb)

        resultados.append({
            "id": caso["id"],
            "score": round(float(score), 4),
            "texto": caso["texto"],
            "subfatos": d_subfatos
        })

    ranking = sorted(resultados, key=lambda x: x["score"], reverse=True)

    return {
        "subfatos": q_subfatos,
        "ranking": ranking
    }


# =====================================================
# 📌 9. ENDPOINTS
# =====================================================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/keller/consulta")
def consultar(req: ConsultaRequest):
    pergunta = req.pergunta.strip()

    if not pergunta:
        raise HTTPException(status_code=400, detail="A pergunta é obrigatória.")

    try:
        resultado = pipeline_keller(pergunta)

        return {
            "pergunta": pergunta,
            "subfatos": resultado["subfatos"],
            "ranking": resultado["ranking"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))