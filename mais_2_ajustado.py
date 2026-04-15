# =====================================================
# 🧠 KELLER-BR – IMPLEMENTAÇÃO FIEL AO ARTIGO
# Baseado em: Deng et al. (EMNLP 2024)
# =====================================================

import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

# =====================================================
# 📌 1. MODELOS
# =====================================================

print("🔹 Carregando modelos...")

MODEL_CHAT = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHAT)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_CHAT,
    device_map="cpu",
    low_cpu_mem_usage=True
)

# Encoder (igual artigo → embeddings)
encoder = SentenceTransformer("neuralmind/bert-base-portuguese-cased")

print("✅ Modelos carregados.")

# =====================================================
# 📌 2. BASE DE CONHECIMENTO JURÍDICA (KELLER)
# =====================================================
# 🔥 Artigo: usa conhecimento jurídico para mapear crime ↔ artigo

BASE_JURIDICA = {
    "artigo 250": "incêndio",
    "artigo 312": "peculato",
    "artigo 155": "furto",
    "artigo 157": "roubo"
}

# =====================================================
# 📌 3. DADOS MOCKADOS (CORPUS)
# =====================================================
# 🔥 Simula base de casos jurídicos

casos = [
    {"id": 1, "texto": "João ateou fogo em casa. Artigo 250 do Código Penal."},
    {"id": 2, "texto": "Maria desviou dinheiro público. Artigo 312 do Código Penal."},
    {"id": 3, "texto": "José subtraiu celular sem violência. Artigo 155 do Código Penal."}
]

# =====================================================
# 📌 4. EXTRAÇÃO (PASSO 1 DO KELLER)
# =====================================================
# 🔥 Artigo: extrair crimes + artigos

def extrair_artigos(texto):
    return re.findall(r"artigo\s*\d+", texto.lower())

def mapear_crimes(artigos):
    crimes = []
    for art in artigos:
        if art in BASE_JURIDICA:
            crimes.append(BASE_JURIDICA[art])
    return crimes

# =====================================================
# 📌 5. GERAÇÃO DE SUBFATOS (CORE DO KELLER)
# =====================================================
# 🔥 Artigo: gerar subfatos por crime

def gerar_subfatos(texto):

    artigos = extrair_artigos(texto)
    crimes = mapear_crimes(artigos)

    subfatos = []

    # 🔥 PARA CADA CRIME → UM SUBFATO (igual paper)
    for crime, artigo in zip(crimes, artigos):

        prompt = f"""
                Você é um especialista em direito penal.
                
                Gere um subfato jurídico baseado no caso abaixo.
                
                REGRAS:
                - Máximo 2 linhas
                - Seja objetivo
                - Estrutura:
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
            max_new_tokens=100,
            do_sample=True,
            temperature=0.3
        )

        resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)

        subfatos.append(resposta.strip())

    return subfatos

# =====================================================
# 📌 6. EMBEDDINGS (ARTIGO SEÇÃO 3.3)
# =====================================================
# 🔥 Cada subfato vira vetor

def gerar_embeddings(subfatos):
    return encoder.encode(subfatos, convert_to_tensor=True, normalize_embeddings=True)

# =====================================================
# 📌 7. MATRIZ DE SIMILARIDADE (Eq. 2 do paper)
# =====================================================

def matriz_similaridade(q_emb, d_emb):
    return util.cos_sim(q_emb, d_emb)

# =====================================================
# 📌 8. MAXSIM + SUM (Eq. 3 do paper)
# =====================================================
# 🔥 CORE DO RANKING

def score_keller(q_emb, d_emb):
    sim_matrix = matriz_similaridade(q_emb, d_emb)

    # MaxSim + Sum
    max_vals = torch.max(sim_matrix, dim=1).values
    return torch.sum(max_vals).item()

# =====================================================
# 📌 9. PIPELINE COMPLETO
# =====================================================

def pipeline_keller(query, casos):

    print("\n=== CONSULTA ===")
    print(query)

    # 🔥 PASSO 1 + 2 + 3 → Reformulação
    q_subfatos = gerar_subfatos(query)

    print("\n🧠 Subfatos da consulta:")
    for sf in q_subfatos:
        print("-", sf)

    q_emb = gerar_embeddings(q_subfatos)

    resultados = []

    # 🔥 PARA CADA DOCUMENTO
    for caso in casos:

        d_subfatos = gerar_subfatos(caso["texto"])
        d_emb = gerar_embeddings(d_subfatos)

        score = score_keller(q_emb, d_emb)

        resultados.append((caso["id"], score))

    # 🔥 Ranking final
    ranking = sorted(resultados, key=lambda x: x[1], reverse=True)

    print("\n=== RANKING FINAL ===")
    for r in ranking:
        print(f"Doc {r[0]} → Score: {r[1]:.4f}")

# =====================================================
# ▶️ EXECUÇÃO
# =====================================================

if __name__ == "__main__":
    query = "O réu provocou incêndio em área urbana. Artigo 250 do Código Penal."
    pipeline_keller(query, casos)