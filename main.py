# =====================================================
# 🧠 KELLER-BR – Pipeline alinhado ao artigo KELLER
#     Versão CPU | 16GB RAM | Modelo 7B quantizado
# =====================================================

import os
import re
import torch
import numpy as np
import multiprocessing
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util

# =====================================================
# ⚙️ CONTROLE DE RECURSOS (CPU / MEMÓRIA)
# =====================================================
total_cores = multiprocessing.cpu_count()
torch.set_num_threads(max(1, total_cores // 2))
os.environ["OMP_NUM_THREADS"] = str(max(1, total_cores // 2))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =====================================================
# 🧠 MODELO DE CHAT (MISTRAL 7B – QUANTIZADO)
# =====================================================
MODEL_CHAT = "mistralai/Mistral-7B-Instruct-v0.2"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print("🔹 Carregando modelo de chat (Mistral 7B quantizado)...")

tokenizer_chat = AutoTokenizer.from_pretrained(MODEL_CHAT)
model_chat = AutoModelForCausalLM.from_pretrained(
    MODEL_CHAT,
    device_map="cpu",
    quantization_config=quant_config,
    low_cpu_mem_usage=True
)

print("✅ Modelo de chat carregado.")

# =====================================================
# 🔎 MODELO DE EMBEDDINGS (BERTIMBAU)
# =====================================================
print("🔹 Carregando BERTimbau...")
encoder = SentenceTransformer("neuralmind/bert-base-portuguese-cased")
print("✅ BERTimbau carregado.")

# =====================================================
# ⚖️ BASE JURÍDICA
# =====================================================
BASE_JURIDICA = {
    "artigo 250": "incêndio",
    "artigo 312": "peculato",
    "artigo 155": "furto",
    "artigo 157": "roubo"
}

VERBOS_PENAIS = [
    "provocou", "ateou", "desviou", "subtraiu", "arrombou"
]

# =====================================================
# 📄 CORPUS SIMULADO (PJe)
# =====================================================
casos = [
    {"id": 1, "texto": "O réu João ateou fogo em residência urbana. Artigo 250 do Código Penal.", "relevante_para": ["incêndio"]},
    {"id": 2, "texto": "A ré Maria desviou R$ 50.000 da empresa pública. Artigo 312 do Código Penal.", "relevante_para": ["peculato"]},
    {"id": 3, "texto": "O acusado José subtraiu celular sem violência. Artigo 155 do Código Penal.", "relevante_para": ["furto"]},
    {"id": 4, "texto": "O réu Pedro arrombou o cofre da loja. Artigo 157 do Código Penal.", "relevante_para": ["roubo"]},
]

# =====================================================
# 🧩 FUNÇÕES AUXILIARES
# =====================================================
def extrair_artigo(texto):
    m = re.search(r"artigo\s*\d+", texto.lower())
    return m.group(0) if m else "artigo desconhecido"

def extrair_agente(texto):
    m = re.search(r"réu\s+\w+|acusado\s+\w+|ré\s+\w+", texto.lower())
    return m.group(0) if m else "agente não identificado"

def extrair_verbo(texto):
    for v in VERBOS_PENAIS:
        if v in texto.lower():
            return v
    return "conduta descrita"

# =====================================================
# 🧠 GERAÇÃO DE SUBFATOS (KELLER – ESTILO ORIGINAL)
# =====================================================
def gerar_subfato_chat(texto):
    artigo = extrair_artigo(texto)

    prompt = f"""
Você é um jurista brasileiro.
Resuma o caso abaixo em no máximo 3 linhas.
NÃO explique, NÃO invente fatos.

Formato obrigatório:
Agente – Conduta – Artigo – Resultado.

Caso:
{texto}
"""

    inputs = tokenizer_chat(prompt, return_tensors="pt")
    outputs = model_chat.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.2,
        do_sample=False
    )

    resposta = tokenizer_chat.decode(outputs[0], skip_special_tokens=True).strip()

    # =============================
    # 🔧 PÓS-PROCESSAMENTO SIMBÓLICO
    # =============================
    if "artigo" not in resposta.lower():
        resposta += f" – {artigo}"

    if not any(v in resposta.lower() for v in VERBOS_PENAIS):
        resposta = f"{extrair_agente(texto)} – {extrair_verbo(texto)} – {artigo}"

    return resposta

# =====================================================
# 🔎 EMBEDDINGS E SIMILARIDADE
# =====================================================
def gerar_embeddings(textos):
    return encoder.encode(textos, convert_to_tensor=True, normalize_embeddings=True)

def score_maxsim(q_emb, d_emb):
    sim = util.cos_sim(q_emb, d_emb)
    return torch.max(sim, dim=1).values.sum().item()

# =====================================================
# 📊 MÉTRICAS
# =====================================================
def precision_at_k(ranked, relevantes, k):
    return len(set(ranked[:k]) & set(relevantes)) / k

# =====================================================
# 🔄 PIPELINE PRINCIPAL
# =====================================================
def pipeline_keller(query, casos):
    print("\n=== CONSULTA ===")
    print(query)

    q_subfato = gerar_subfato_chat(query)
    print("\n🧠 Subfato da consulta:")
    print(q_subfato)

    q_emb = gerar_embeddings([q_subfato])

    resultados = []
    for c in casos:
        sf = gerar_subfato_chat(c["texto"])
        emb = gerar_embeddings([sf])
        score = score_maxsim(q_emb, emb)
        resultados.append((c["id"], score))

    ranked = sorted(resultados, key=lambda x: x[1], reverse=True)
    print("\n=== RANKING ===")
    for r in ranked:
        print(r)

# =====================================================
# ▶️ EXECUÇÃO
# =====================================================
if __name__ == "__main__":
    consulta = "O réu foi acusado de provocar incêndio em área urbana. Artigo 250 do Código Penal."
    pipeline_keller(consulta, casos)