# =====================================================
# 🧠 KELLER-BR FINAL
# =====================================================

import re
import faiss
import torch
import numpy as np

from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

# =====================================================
# CONFIG
# =====================================================

APP_TITLE = "KELLER-BR"
FRONTEND_ORIGIN = "http://localhost:4200"

LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 5
TOP_K_RETRIEVE = 25
MAX_DOCS = 3000
MAX_SUBFACTS = 5
MIN_SENT_LEN = 20
MAX_PROMPT_CASES = 3

# =====================================================
# APP
# =====================================================

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConsultaRequest(BaseModel):
    pergunta: str = Field(..., min_length=3)

# =====================================================
# LOG
# =====================================================

def log(msg: str):
    print(f"🔥 {msg}")

def log_block(title: str):
    print(f"\n================ {title} ================")

# =====================================================
# MODELOS
# =====================================================

log("Carregando LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="cpu")

log("Carregando encoder...")
encoder = SentenceTransformer(EMB_MODEL)

# =====================================================
# DATASETS
# =====================================================

log("Carregando datasets de processos...")

datasets_processos = [
    load_dataset("joelniklaus/brazilian_court_decisions", split="train"),
    load_dataset("celsowm/jurisprudencias_stf", split="train"),
    load_dataset("celsowm/jurisprudencias_tjmg", split="train"),
    load_dataset("celsowm/jurisprudencias_stj", split="train"),
]

DATA: List[Dict[str, Any]] = []
for ds in datasets_processos:
    DATA.extend(list(ds))

log(f"Total bruto de documentos: {len(DATA)}")

log("Carregando Código Penal...")
ds_cp = load_dataset("celsowm/codigo_penal_brasileiro_lei_2848_1940", split="train")
CODIGO_PENAL = list(ds_cp)
log(f"Total de artigos do Código Penal: {len(CODIGO_PENAL)}")

# =====================================================
# NORMALIZAÇÃO / EXTRAÇÃO
# =====================================================

def limpar_texto(texto: str) -> str:
    texto = str(texto or "")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

def extrair_numero(item: Dict[str, Any], idx: int) -> str:
    return str(
        item.get("process_number")
        or item.get("numero_processo")
        or item.get("processo")
        or item.get("id")
        or f"DOC_{idx}"
    )

def extrair_texto(item: Dict[str, Any]) -> str:
    candidatos = [
        item.get("ementa_text"),
        item.get("ementa"),
        item.get("texto"),
        item.get("conteudo"),
        item.get("decisao"),
        item.get("documento"),
        item.get("text"),
    ]

    for c in candidatos:
        c = limpar_texto(c)
        if len(c) >= 30:
            return c

    return ""

def montar_texto_documento(item: Dict[str, Any], idx: int) -> str:
    processo = extrair_numero(item, idx)
    orgao = limpar_texto(item.get("orgao_julgador"))
    relator = limpar_texto(item.get("judge_relator"))
    data_pub = limpar_texto(item.get("publish_date"))

    ementa = extrair_texto(item)

    partes = [f"Processo: {processo}"]
    if orgao:
        partes.append(f"Órgão julgador: {orgao}")
    if relator:
        partes.append(f"Relator: {relator}")
    if data_pub:
        partes.append(f"Data de publicação: {data_pub}")
    if ementa:
        partes.append(f"Ementa: {ementa}")

    return "\n".join(partes).strip()

# =====================================================
# EXTRAÇÃO DE ARTIGOS
# =====================================================

ARTICLE_REGEX = re.compile(
    r"(?:art\.?|artigo|arts\.?)\s*(\d+[A-Za-zº°\-]*(?:\s*,\s*\d+[A-Za-zº°\-]*)*)",
    flags=re.IGNORECASE
)

def extrair_artigos_do_texto(texto: str) -> List[str]:
    encontrados = []

    for match in ARTICLE_REGEX.findall(texto or ""):
        partes = re.split(r"\s*,\s*", match)
        for p in partes:
            p = limpar_texto(p)
            if p:
                encontrados.append(p)

    vistos = set()
    saida = []
    for a in encontrados:
        if a not in vistos:
            vistos.add(a)
            saida.append(a)

    return saida[:10]

def buscar_artigos_codigo_penal(artigos: List[str]) -> List[Dict[str, str]]:
    resultados = []
    if not artigos:
        return resultados

    artigos_set = set(str(a) for a in artigos)

    for item in CODIGO_PENAL:
        artigo_id = limpar_texto(item.get("artigo") or item.get("id") or item.get("article"))
        texto = limpar_texto(item.get("text") or item.get("texto") or "")
        if artigo_id in artigos_set and texto:
            resultados.append({
                "artigo": artigo_id,
                "texto": texto
            })

    return resultados[:10]

# =====================================================
# SUBFATOS
# =====================================================

KEY_TERMS = [
    "crime", "penal", "homicídio", "homicidio", "roubo", "furto",
    "tráfico", "trafico", "arma", "arma de fogo", "revólver", "revolver",
    "pistola", "munição", "municao", "grave ameaça", "grave ameaca",
    "violência", "violencia", "disparo", "associação para o tráfico",
    "associacao para o trafico", "latrocínio", "latrocinio", "tentada",
    "tentado", "qualificado", "tentativa", "art.", "artigo",
    "porte", "posse", "munições", "armamento", "bélico", "belico",
]

ARM_TERMS = [
    "arma", "arma de fogo", "revólver", "revolver", "pistola",
    "munição", "municao", "munições", "porte", "posse",
    "armamento", "disparo", "acessório", "acessorio",
]

def split_sentencas(texto: str) -> List[str]:
    texto = limpar_texto(texto)
    if not texto:
        return []

    partes = re.split(r"(?<=[\.\;\:\!\?])\s+", texto)
    sentencas = []

    for p in partes:
        p = limpar_texto(p)
        if len(p) >= MIN_SENT_LEN:
            sentencas.append(p)

    return sentencas

def score_sentenca(sentenca: str, pergunta: str = "") -> int:
    s_lower = sentenca.lower()
    p_lower = (pergunta or "").lower()

    score = 0

    for termo in KEY_TERMS:
        if termo in s_lower:
            score += 2

    for token in re.findall(r"\w+", p_lower):
        if len(token) >= 4 and token in s_lower:
            score += 3

    if re.search(r"\bart\.?\s*\d+", s_lower):
        score += 2

    if ("arma" in p_lower or "fogo" in p_lower) and any(t in s_lower for t in ARM_TERMS):
        score += 6

    return score

def expandir_query_subfatos(pergunta: str) -> List[str]:
    p = pergunta.lower()
    subfatos = [pergunta]

    if "arma" in p or "fogo" in p:
        subfatos.extend([
            "porte ilegal de arma de fogo",
            "posse irregular de arma de fogo",
            "posse de munição e acessório de arma de fogo",
            "uso de arma de fogo em contexto criminal",
        ])

    vistos = set()
    saida = []
    for s in subfatos:
        chave = s.lower().strip()
        if chave not in vistos:
            vistos.add(chave)
            saida.append(s)

    return saida[:MAX_SUBFACTS]

def gerar_subfatos(texto: str, pergunta: str = "", is_query: bool = False) -> List[str]:
    print("\n[SUBFATOS] GERANDO...")

    if is_query:
        subfatos_query = expandir_query_subfatos(texto)
        for i, sf in enumerate(subfatos_query):
            print(f"Subfato {i+1}: {sf}")
        return subfatos_query

    sentencas = split_sentencas(texto)
    if not sentencas:
        fallback = limpar_texto(texto)[:300]
        print("⚠️ Sem sentenças válidas, usando fallback")
        print(f"Subfato 1: {fallback}")
        return [fallback]

    pontuadas = []
    for s in sentencas:
        pontuadas.append((score_sentenca(s, pergunta), s))

    pontuadas.sort(key=lambda x: x[0], reverse=True)

    subfatos = []
    vistos = set()

    for _, s in pontuadas:
        chave = s.lower()
        if chave not in vistos:
            vistos.add(chave)
            subfatos.append(s)
        if len(subfatos) >= MAX_SUBFACTS:
            break

    if not subfatos:
        subfatos = sentencas[:2]

    for i, sf in enumerate(subfatos):
        print(f"Subfato {i+1}: {sf}")

    return subfatos

# =====================================================
# EMBEDDING
# =====================================================

def embed(textos: List[str]) -> np.ndarray:
    if not textos:
        textos = ["texto vazio"]

    emb = encoder.encode(
        textos,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    print(f"[EMBED] shape: {emb.shape}")
    return emb

# =====================================================
# INDEXAÇÃO OFFLINE
# =====================================================

log("Indexando documentos...")

DOC_SUB: List[List[str]] = []
DOC_VEC: List[np.ndarray] = []
DOC_META: List[Dict[str, Any]] = []

docs_validos = 0

for i, item in enumerate(DATA[:MAX_DOCS]):
    texto_limpo = extrair_texto(item)

    if len(texto_limpo) < 30:
        continue

    processo = extrair_numero(item, i)
    texto_doc = montar_texto_documento(item, i)

    print(f"\n📄 DOC {docs_validos} - {processo}")

    sub = gerar_subfatos(texto_limpo)
    emb = embed(sub)
    vec = np.mean(emb, axis=0)

    DOC_SUB.append(sub)
    DOC_VEC.append(vec)
    DOC_META.append({
        "idx_original": i,
        "processo": processo,
        "texto": texto_doc,
        "artigos": extrair_artigos_do_texto(texto_limpo),
    })

    docs_validos += 1

DOC_VEC = np.array(DOC_VEC).astype("float32")

log(f"Documentos válidos indexados: {len(DOC_VEC)}")

# =====================================================
# FAISS
# =====================================================

log("Criando índice FAISS...")
index = faiss.IndexFlatIP(DOC_VEC.shape[1])
index.add(DOC_VEC)
log("FAISS PRONTO")

# =====================================================
# SCORE (MAXSIM + SUM)
# =====================================================

def score(q_emb: np.ndarray, d_emb: np.ndarray) -> float:
    sim = util.cos_sim(torch.tensor(q_emb), torch.tensor(d_emb))
    max_sim = torch.max(sim, dim=1).values
    s = float(torch.sum(max_sim))

    print("[MATRIZ SIMILARIDADE]")
    print(sim)
    print("[MAXSIM]")
    print(max_sim)
    print(f"[SCORE] {s}")

    return s

# =====================================================
# FILTRO DE CONTEXTO
# =====================================================

def doc_relevante_para_query(pergunta: str, subfatos_doc: List[str], artigos_doc: List[str]) -> bool:
    p = pergunta.lower()
    texto_doc = " ".join(subfatos_doc).lower()

    termos_query = [
        token for token in re.findall(r"\w+", p)
        if len(token) >= 4
    ]

    hits = sum(1 for t in termos_query if t in texto_doc)

    if "arma" in p or "fogo" in p:
        if any(x in texto_doc for x in ARM_TERMS):
            # evita deixar tráfico puro subir só por conter art. 33
            if "tráfico" in texto_doc or "trafico" in texto_doc:
                if not any(x in texto_doc for x in ["arma de fogo", "pistola", "revolver", "revólver", "munição", "municao", "disparo"]):
                    return False
            return True

        # se tiver artigo, mas não houver nenhum termo de arma, não deixa passar
        return False

    return hits >= 1

# =====================================================
# ARTIGOS PRIORITÁRIOS PARA ARMA DE FOGO
# =====================================================

def priorizar_artigos(pergunta: str, artigos_encontrados: List[str]) -> List[str]:
    p = pergunta.lower()
    unicos = []
    vistos = set()

    for a in artigos_encontrados:
        if a not in vistos:
            vistos.add(a)
            unicos.append(a)

    if "arma" in p or "fogo" in p:
        prioritarios = []
        resto = []

        # Estatuto do Desarmamento costuma aparecer nos casos como 12/14/16
        foco = {"12", "14", "16"}
        for a in unicos:
            if a in foco:
                prioritarios.append(a)
            else:
                resto.append(a)

        return (prioritarios + resto)[:10]

    return unicos[:10]

# =====================================================
# LLM
# =====================================================

def extrair_resposta_limpa(texto: str) -> str:
    marcadores = [
        "Resposta final:",
        "Resposta:",
        "Conclusão:",
        "Resposta objetiva:",
    ]

    for m in marcadores:
        pos = texto.lower().find(m.lower())
        if pos != -1:
            return texto[pos + len(m):].strip()

    # fallback: remove prompt repetido no começo
    if "Pergunta do usuário:" in texto and "Tarefa:" in texto:
        partes = texto.split("Tarefa:")
        if len(partes) > 1:
            return partes[-1].strip()

    return texto.strip()

def gerar_resposta_llm(pergunta: str, ranking: List[Dict[str, Any]], artigos_cp: List[Dict[str, str]]) -> str:
    print("\n🧠 GERANDO RESPOSTA LLM...")

    contexto_casos = "\n\n".join([r["texto"] for r in ranking[:MAX_PROMPT_CASES]])

    if artigos_cp:
        contexto_cp = "\n".join([
            f"Artigo {a['artigo']}: {a['texto']}"
            for a in artigos_cp[:5]
        ])
    else:
        contexto_cp = "Nenhum artigo específico identificado diretamente nos casos recuperados."

    prompt = f"""
Você é um especialista em direito penal brasileiro.

Pergunta do usuário:
{pergunta}

Casos recuperados:
{contexto_casos}

Código Penal relevante:
{contexto_cp}

Tarefa:
1. Identifique o crime mais provável relacionado à pergunta.
2. Informe o artigo mais provável com base SOMENTE no contexto recuperado.
3. Informe se arma de fogo aparece como elementar, qualificadora ou majorante, quando isso estiver sustentado pelo contexto.
4. Informe pena aproximada REALISTA no Brasil.
5. Se o contexto estiver insuficiente ou misturado, diga explicitamente que há ambiguidade.

Regras:
- NÃO invente artigos.
- NÃO diga prisão perpétua ou reclusão perpétua.
- Responda em português do Brasil.
- Seja técnico e direto.
- Comece sua saída com: "Resposta final:"

Resposta final:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    prompt_len = inputs["input_ids"].shape[1]

    outputs = llm.generate(
        **inputs,
        max_new_tokens=220,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = outputs[0][prompt_len:]
    resposta = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    resposta = extrair_resposta_limpa(resposta)

    print("\n📤 RESPOSTA LLM:")
    print(resposta)

    return resposta

# =====================================================
# PIPELINE
# =====================================================

def pipeline(pergunta: str) -> Dict[str, Any]:
    log_block("PIPELINE INICIADO")
    print(f"Pergunta: {pergunta}")

    # 1. Subfatos da query
    q_sub = gerar_subfatos(pergunta, pergunta=pergunta, is_query=True)

    # 2. Embedding da query
    q_emb = embed(q_sub)
    q_vec = np.mean(q_emb, axis=0).reshape(1, -1)

    # 3. Retrieve inicial
    print("\n[FAISS] RETRIEVE INICIAL")
    D, I = index.search(q_vec, TOP_K_RETRIEVE)

    candidatos = []
    for pos, idx_doc in enumerate(I[0]):
        if idx_doc < 0 or idx_doc >= len(DOC_META):
            continue

        meta = DOC_META[idx_doc]
        sub_doc = DOC_SUB[idx_doc]

        print(f"Candidato {pos+1}: {meta['processo']} | score_faiss={float(D[0][pos])}")

        if not doc_relevante_para_query(pergunta, sub_doc, meta["artigos"]):
            print("   ↳ descartado pelo filtro de contexto")
            continue

        candidatos.append(idx_doc)

    if not candidatos:
        print("⚠️ Filtro removeu tudo, usando candidatos originais do FAISS")
        candidatos = [idx_doc for idx_doc in I[0] if 0 <= idx_doc < len(DOC_META)]

    # 4. Reranking
    resultados = []
    artigos_encontrados = []

    print("\n[RERANK] MAXSIM + SUM")
    for idx_doc in candidatos:
        d_sub = DOC_SUB[idx_doc]
        d_emb = embed(d_sub)
        s = score(q_emb, d_emb)

        meta = DOC_META[idx_doc]
        artigos_encontrados.extend(meta["artigos"])

        resultados.append({
            "id": int(idx_doc),
            "processo": meta["processo"],
            "score": float(s),
            "texto": meta["texto"],
            "subfatos": d_sub,
            "artigos": meta["artigos"]
        })

    ranking = sorted(resultados, key=lambda x: x["score"], reverse=True)

    # 5. Artigos
    artigos_unicos = priorizar_artigos(pergunta, artigos_encontrados)
    artigos_cp = buscar_artigos_codigo_penal(artigos_unicos)

    # 6. Resposta final do LLM
    resposta_llm = gerar_resposta_llm(pergunta, ranking[:TOP_K], artigos_cp)

    resposta = {
        "pergunta": pergunta,
        "subfatos_query": q_sub,
        "artigos_identificados_nos_casos": artigos_unicos[:10],
        "artigos_codigo_penal_usados": artigos_cp[:5],
        "ranking": ranking[:TOP_K],
        "resposta_llm": resposta_llm
    }

    print("\n[RESPOSTA FINAL]")
    print(resposta)

    return resposta

# =====================================================
# ENDPOINT
# =====================================================

@app.post("/api/keller/consulta")
def consultar(req: ConsultaRequest):
    try:
        print("\n================ API CALL ================")
        print(f"Pergunta recebida: {req.pergunta}")
        return pipeline(req.pergunta)
    except Exception as e:
        print("\n❌ ERRO")
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}