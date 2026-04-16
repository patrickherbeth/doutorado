# =====================================================
# 🧠 KELLER-BR FINAL — CORRIGIDO NÍVEL ARTIGO
# =====================================================

import re
import faiss
import torch
import numpy as np

from typing import List, Dict, Any, Tuple

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
MAX_CASE_TEXT_CHARS = 1200
MIN_RELEVANCE_HITS = 1

DEVICE_MAP = "cpu"

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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    device_map=DEVICE_MAP
)

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

# =====================================================
# NORMALIZAÇÃO / EXTRAÇÃO
# =====================================================

def limpar_texto(texto: Any) -> str:
    texto = str(texto or "")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

def normalizar_lower(texto: str) -> str:
    return limpar_texto(texto).lower()

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

def cortar_texto(texto: str, limite: int) -> str:
    texto = limpar_texto(texto)
    if len(texto) <= limite:
        return texto
    return texto[:limite].rsplit(" ", 1)[0] + "..."

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
# EXTRAÇÃO DE REFERÊNCIAS LEGAIS — DIRETO DOS CASOS
# =====================================================

LEI_REGEX = re.compile(
    r"""(?ix)
    art\.?\s*
    (?P<artigo>\d+[A-Za-zº°\-]*)
    (?:\s*,\s*(?:§+\s*\d+º?)?)?
    .*?
    (?:lei\s*n?[ºo.]?\s*(?P<lei>\d{1,6}(?:\.\d{3})*(?:/\d{2,4})?))
    """
)

ARTIGO_SO_REGEX = re.compile(
    r"""(?ix)
    \bart\.?\s*(?P<artigo>\d+[A-Za-zº°\-]*)
    """
)

def normalizar_numero_lei(lei: str) -> str:
    lei = limpar_texto(lei).replace(" ", "")
    return lei

def extrair_referencias_legais(texto: str) -> List[Dict[str, str]]:
    texto = limpar_texto(texto)
    refs = []
    vistos = set()

    for m in LEI_REGEX.finditer(texto):
        artigo = limpar_texto(m.group("artigo"))
        lei = normalizar_numero_lei(m.group("lei"))
        chave = (artigo, lei)
        if chave not in vistos:
            vistos.add(chave)
            refs.append({
                "artigo": artigo,
                "lei": lei,
                "fonte": f"art. {artigo} da Lei {lei}"
            })

    # fallback: se vier apenas artigo no texto, sem lei explícita
    # ainda assim registramos, mas com lei desconhecida
    if not refs:
        for m in ARTIGO_SO_REGEX.finditer(texto):
            artigo = limpar_texto(m.group("artigo"))
            chave = (artigo, "desconhecida")
            if chave not in vistos:
                vistos.add(chave)
                refs.append({
                    "artigo": artigo,
                    "lei": "desconhecida",
                    "fonte": f"art. {artigo}"
                })

    return refs[:20]

def priorizar_referencias_legais(pergunta: str, refs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    p = normalizar_lower(pergunta)
    if not refs:
        return []

    # para "arma de fogo", preferir Estatuto do Desarmamento 10.826/03
    refs_10826 = []
    refs_outros = []

    for r in refs:
        lei = r["lei"]
        if "arma" in p or "fogo" in p:
            if "10.826" in lei or "10826" in lei or "10.826/03" in lei:
                refs_10826.append(r)
            else:
                refs_outros.append(r)
        else:
            refs_outros.append(r)

    ordenado = refs_10826 + refs_outros

    saida = []
    vistos = set()
    for r in ordenado:
        chave = (r["artigo"], r["lei"])
        if chave not in vistos:
            vistos.add(chave)
            saida.append(r)

    return saida[:10]

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
            "estatuto do desarmamento",
        ])

    if "roubo" in p:
        subfatos.extend([
            "roubo com arma de fogo",
            "majorante pelo emprego de arma de fogo",
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
    return emb.astype("float32")

# =====================================================
# SCORE
# =====================================================

def score_maxsim_sum(q_emb: np.ndarray, d_emb: np.ndarray) -> float:
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
# FILTRO LEVE DE CONTEXTO
# =====================================================

def doc_relevante_para_query(pergunta: str, subfatos_doc: List[str], refs_legais: List[Dict[str, str]]) -> bool:
    p = pergunta.lower()
    texto_doc = " ".join(subfatos_doc).lower()

    termos_query = [
        token for token in re.findall(r"\w+", p)
        if len(token) >= 4
    ]

    hits = sum(1 for t in termos_query if t in texto_doc)

    if "arma" in p or "fogo" in p:
        has_arm = any(x in texto_doc for x in ARM_TERMS)
        has_desarmamento = any("10.826" in r["lei"] or "10826" in r["lei"] for r in refs_legais)
        return has_arm or has_desarmamento or hits >= MIN_RELEVANCE_HITS

    return hits >= MIN_RELEVANCE_HITS

# =====================================================
# RESUMO DE CASO PARA LLM
# =====================================================

def montar_resumo_caso_llm(item_ranking: Dict[str, Any]) -> str:
    processo = item_ranking["processo"]
    score = round(float(item_ranking["score"]), 4)
    subfatos = item_ranking["subfatos"][:3]
    refs = item_ranking.get("refs_legais", [])[:3]

    partes = [
        f"Processo: {processo}",
        f"Score: {score}",
        "Trechos-chave:"
    ]

    for sf in subfatos:
        partes.append(f"- {sf}")

    if refs:
        partes.append("Referências legais encontradas:")
        for r in refs:
            if r["lei"] != "desconhecida":
                partes.append(f"- art. {r['artigo']} da Lei {r['lei']}")
            else:
                partes.append(f"- art. {r['artigo']}")

    return "\n".join(partes)

# =====================================================
# FALLBACK DETERMINÍSTICO
# =====================================================

def gerar_resposta_fallback(pergunta: str, ranking: List[Dict[str, Any]], refs: List[Dict[str, str]]) -> str:
    if not ranking:
        return (
            "Não encontrei casos suficientes para responder com segurança. "
            "A consulta ficou ambígua e precisa de mais contexto."
        )

    top1 = ranking[0]
    top_refs = top1.get("refs_legais", [])

    artigo_txt = "não identificado com segurança"
    if top_refs:
        r = top_refs[0]
        if r["lei"] != "desconhecida":
            artigo_txt = f"art. {r['artigo']} da Lei {r['lei']}"
        else:
            artigo_txt = f"art. {r['artigo']}"

    classificacao = "há ambiguidade"
    p = pergunta.lower()
    texto_top = " ".join(top1.get("subfatos", [])).lower()

    if "posse" in texto_top:
        classificacao = "o contexto aponta mais para posse irregular/ilegal de arma de fogo"
    elif "porte" in texto_top:
        classificacao = "o contexto aponta mais para porte ilegal de arma de fogo"
    elif "roubo" in texto_top and "arma" in texto_top:
        classificacao = "o contexto aponta para roubo com emprego de arma de fogo"
    elif "arma" in texto_top:
        classificacao = "o contexto aponta genericamente para delito envolvendo arma de fogo"

    resposta = (
        f"Com base nos casos recuperados, {classificacao}. "
        f"A referência legal mais provável no material recuperado é {artigo_txt}. "
        f"Como a pergunta é genérica ('{pergunta}'), a resposta ainda tem ambiguidade "
        f"entre posse, porte ou uso da arma em outro crime."
    )

    return resposta

# =====================================================
# LLM
# =====================================================

def resposta_llm_valida(texto: str) -> bool:
    t = limpar_texto(texto)
    if len(t) < 40:
        return False

    lixos = [
        "Processo:",
        "Órgão julgador:",
        "Relator:",
        "Data de publicação:",
        "Ementa:"
    ]

    hits_lixo = sum(1 for x in lixos if x in t)
    if hits_lixo >= 3:
        return False

    return True

def gerar_prompt_llm(pergunta: str, ranking: List[Dict[str, Any]], refs: List[Dict[str, str]]) -> str:
    casos = "\n\n".join([montar_resumo_caso_llm(r) for r in ranking[:MAX_PROMPT_CASES]])

    if refs:
        refs_txt = "\n".join([
            f"- art. {r['artigo']} da Lei {r['lei']}" if r["lei"] != "desconhecida" else f"- art. {r['artigo']}"
            for r in refs[:5]
        ])
    else:
        refs_txt = "- nenhuma referência legal explícita identificada"

    return f"""
Você é um especialista em direito penal brasileiro.

Responda SOMENTE com análise jurídica objetiva, sem copiar ementas, sem repetir textos dos casos, sem continuar trechos do contexto.

Pergunta:
{pergunta}

Casos recuperados:
{casos}

Referências legais extraídas dos casos:
{refs_txt}

Responda EXATAMENTE neste formato:

Resposta final:
- Enquadramento provável:
- Base legal provável:
- Papel da arma de fogo:
- Pena em tese:
- Grau de certeza:

Regras:
- Use apenas o contexto recuperado.
- Se houver ambiguidade entre posse, porte ou uso da arma em outro crime, diga isso claramente.
- Não invente artigo.
- Não copie integralmente ementas.
- Máximo de 8 linhas.
""".strip()

def gerar_resposta_llm(pergunta: str, ranking: List[Dict[str, Any]], refs: List[Dict[str, str]]) -> str:
    print("\n🧠 GERANDO RESPOSTA LLM...")

    prompt = gerar_prompt_llm(pergunta, ranking, refs)

    # Tenta usar chat template se existir
    try:
        messages = [{"role": "user", "content": prompt}]
        model_inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        )
        input_ids = model_inputs
        attention_mask = torch.ones_like(input_ids)
    except Exception:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

    prompt_len = input_ids.shape[1]

    outputs = llm.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=180,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.08
    )

    generated_ids = outputs[0][prompt_len:]
    resposta = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("\n📤 RESPOSTA BRUTA LLM:")
    print(resposta)

    if "Resposta final:" in resposta:
        resposta = resposta.split("Resposta final:", 1)[1].strip()
        resposta = "Resposta final:\n" + resposta

    if not resposta_llm_valida(resposta):
        print("⚠️ Saída do LLM inválida. Aplicando fallback determinístico.")
        fallback = gerar_resposta_fallback(pergunta, ranking, refs)
        return f"Resposta final:\n{fallback}"

    return resposta

# =====================================================
# INDEXAÇÃO OFFLINE
# =====================================================

log("Indexando documentos...")

DOC_SUB: List[List[str]] = []
DOC_SUB_EMB: List[np.ndarray] = []
DOC_VEC: List[np.ndarray] = []
DOC_META: List[Dict[str, Any]] = []

docs_validos = 0

for i, item in enumerate(DATA[:MAX_DOCS]):
    texto_limpo = extrair_texto(item)

    if len(texto_limpo) < 30:
        continue

    processo = extrair_numero(item, i)
    texto_doc = montar_texto_documento(item, i)
    refs_legais = extrair_referencias_legais(texto_limpo)

    print(f"\n📄 DOC {docs_validos} - {processo}")

    sub = gerar_subfatos(texto_limpo)
    emb_sub = embed(sub)
    vec = np.mean(emb_sub, axis=0)

    DOC_SUB.append(sub)
    DOC_SUB_EMB.append(emb_sub)
    DOC_VEC.append(vec)
    DOC_META.append({
        "idx_original": i,
        "processo": processo,
        "texto": texto_doc,
        "texto_curto": cortar_texto(texto_doc, MAX_CASE_TEXT_CHARS),
        "refs_legais": refs_legais,
    })

    docs_validos += 1

if not DOC_VEC:
    raise RuntimeError("Nenhum documento válido foi indexado.")

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
# PIPELINE
# =====================================================

def pipeline(pergunta: str) -> Dict[str, Any]:
    log_block("PIPELINE INICIADO")
    print(f"Pergunta: {pergunta}")

    # 1. Subfatos da query
    q_sub = gerar_subfatos(pergunta, pergunta=pergunta, is_query=True)

    # 2. Embedding da query
    q_emb = embed(q_sub)
    q_vec = np.mean(q_emb, axis=0).reshape(1, -1).astype("float32")

    # 3. Retrieve inicial
    print("\n[FAISS] RETRIEVE INICIAL")
    D, I = index.search(q_vec, TOP_K_RETRIEVE)

    candidatos = []
    for pos, idx_doc in enumerate(I[0]):
        if idx_doc < 0 or idx_doc >= len(DOC_META):
            continue

        meta = DOC_META[idx_doc]
        sub_doc = DOC_SUB[idx_doc]
        refs_legais = meta["refs_legais"]

        print(f"Candidato {pos+1}: {meta['processo']} | score_faiss={float(D[0][pos])}")

        if not doc_relevante_para_query(pergunta, sub_doc, refs_legais):
            print("   ↳ descartado pelo filtro leve de contexto")
            continue

        candidatos.append(idx_doc)

    if not candidatos:
        print("⚠️ Filtro removeu tudo, usando candidatos originais do FAISS")
        candidatos = [idx_doc for idx_doc in I[0] if 0 <= idx_doc < len(DOC_META)]

    # 4. Reranking MAXSIM + SUM
    resultados = []
    refs_encontradas = []

    print("\n[RERANK] MAXSIM + SUM")
    for idx_doc in candidatos:
        d_sub = DOC_SUB[idx_doc]
        d_emb = DOC_SUB_EMB[idx_doc]
        s = score_maxsim_sum(q_emb, d_emb)

        meta = DOC_META[idx_doc]
        refs_encontradas.extend(meta["refs_legais"])

        resultados.append({
            "id": int(idx_doc),
            "processo": meta["processo"],
            "score": float(s),
            "texto": meta["texto_curto"],
            "subfatos": d_sub,
            "refs_legais": meta["refs_legais"],
        })

    ranking = sorted(resultados, key=lambda x: x["score"], reverse=True)

    # 5. Referências legais
    refs_priorizadas = priorizar_referencias_legais(pergunta, refs_encontradas)

    # 6. Resposta final do LLM
    resposta_llm = gerar_resposta_llm(pergunta, ranking[:TOP_K], refs_priorizadas)

    resposta = {
        "pergunta": pergunta,
        "subfatos_query": q_sub,
        "referencias_legais_identificadas": refs_priorizadas[:10],
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