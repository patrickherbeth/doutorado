# =====================================================
# 🧠 KELLER-BR FINAL — VERSÃO NÍVEL PUBLICAÇÃO
# =====================================================

import re
import faiss
import torch
import numpy as np

from typing import List, Dict, Any, Tuple, Optional

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
MAX_CASE_TEXT_CHARS = 1400
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
# NORMALIZAÇÃO
# =====================================================

def limpar_texto(texto: Any) -> str:
    texto = str(texto or "")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

def normalizar_lower(texto: str) -> str:
    return limpar_texto(texto).lower()

def cortar_texto(texto: str, limite: int) -> str:
    texto = limpar_texto(texto)
    if len(texto) <= limite:
        return texto
    return texto[:limite].rsplit(" ", 1)[0] + "..."

def sem_acentos_min(texto: str) -> str:
    mapa = str.maketrans(
        "áàãâäéèêëíìîïóòõôöúùûüç",
        "aaaaaeeeeiiiiooooouuuuc"
    )
    return normalizar_lower(texto).translate(mapa)

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
# EXTRAÇÃO DE CAMPOS
# =====================================================

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
# DOMÍNIO JURÍDICO
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

TRAFICO_TERMS = [
    "tráfico", "trafico", "entorpecente", "entorpecentes", "maconha",
    "cocaína", "cocaina", "lei de drogas", "associação para o tráfico",
    "associacao para o trafico", "balança de precisão", "balanca de precisao"
]

def eh_query_arma(pergunta: str) -> bool:
    p = sem_acentos_min(pergunta)
    return "arma" in p or "fogo" in p or "municao" in p or "munição" in p

# =====================================================
# SENTENÇAS E SUBFATOS
# =====================================================

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
    s_lower = sem_acentos_min(sentenca)
    p_lower = sem_acentos_min(pergunta)

    score = 0

    for termo in KEY_TERMS:
        if sem_acentos_min(termo) in s_lower:
            score += 2

    for token in re.findall(r"\w+", p_lower):
        if len(token) >= 4 and token in s_lower:
            score += 3

    if re.search(r"\bart\.?\s*\d+", s_lower):
        score += 2

    if eh_query_arma(pergunta) and any(sem_acentos_min(t) in s_lower for t in ARM_TERMS):
        score += 6

    return score

def expandir_query_subfatos(pergunta: str) -> List[str]:
    p = sem_acentos_min(pergunta)
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

    pontuadas = [(score_sentenca(s, pergunta), s) for s in sentencas]
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
# SCORE MAXSIM + SUM
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
# EXTRAÇÃO LEGAL ROBUSTA
# =====================================================

LEI_PATTERN = re.compile(
    r"lei\s*(?:n[ºo°.]?\s*)?(?P<lei>\d{1,6}(?:\.\d{3})*(?:/\d{2,4})?)",
    flags=re.IGNORECASE
)

ARTIGO_PATTERN = re.compile(
    r"\bart(?:\.|igo)?\s*(?P<artigo>\d+[A-Za-zº°\-]*)",
    flags=re.IGNORECASE
)

def normalizar_numero_lei(lei: str) -> str:
    lei = limpar_texto(lei).replace(" ", "")
    lei = lei.replace("nº", "").replace("n°", "").replace("no", "")
    return lei

def extrair_referencias_legais(texto: str) -> List[Dict[str, str]]:
    """
    Estratégia robusta:
    - divide em sentenças
    - procura artigo e lei na MESMA sentença
    - só associa artigo à lei se a distância textual for curta
    - fallback só com artigo, sem inventar lei
    """
    sentencas = split_sentencas(texto)
    refs = []
    vistos = set()

    for sent in sentencas:
        sent_limpa = limpar_texto(sent)

        artigos = list(ARTIGO_PATTERN.finditer(sent_limpa))
        leis = list(LEI_PATTERN.finditer(sent_limpa))

        if artigos and leis:
            for a in artigos:
                for l in leis:
                    dist = abs(a.start() - l.start())
                    if dist <= 90:
                        artigo = limpar_texto(a.group("artigo"))
                        lei = normalizar_numero_lei(l.group("lei"))
                        chave = (artigo, lei)
                        if chave not in vistos:
                            vistos.add(chave)
                            refs.append({
                                "artigo": artigo,
                                "lei": lei,
                                "fonte": f"art. {artigo} da Lei {lei}"
                            })

        elif artigos:
            for a in artigos:
                artigo = limpar_texto(a.group("artigo"))
                chave = (artigo, "desconhecida")
                if chave not in vistos:
                    vistos.add(chave)
                    refs.append({
                        "artigo": artigo,
                        "lei": "desconhecida",
                        "fonte": f"art. {artigo}"
                    })

    return refs[:25]

def filtrar_referencias_para_query(pergunta: str, refs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not refs:
        return []

    if not eh_query_arma(pergunta):
        saida = []
        vistos = set()
        for r in refs:
            chave = (r["artigo"], r["lei"])
            if chave not in vistos:
                vistos.add(chave)
                saida.append(r)
        return saida[:10]

    refs_firearm = []
    refs_fallback = []

    for r in refs:
        lei = r["lei"]
        artigo = r["artigo"]

        # Estatuto do Desarmamento
        if "10.826" in lei or "10826" in lei:
            refs_firearm.append(r)
            continue

        # artigos sem lei explícita, mas potencialmente úteis
        if lei == "desconhecida" and artigo in {"12", "14", "16", "17", "18"}:
            refs_fallback.append(r)

    ordenado = refs_firearm + refs_fallback

    saida = []
    vistos = set()
    for r in ordenado:
        chave = (r["artigo"], r["lei"])
        if chave not in vistos:
            vistos.add(chave)
            saida.append(r)

    return saida[:10]

# =====================================================
# FILTRO SEMÂNTICO DE DOMÍNIO
# =====================================================

def contar_hits(texto: str, termos: List[str]) -> int:
    t = sem_acentos_min(texto)
    return sum(1 for termo in termos if sem_acentos_min(termo) in t)

def penalidade_trafico_em_query_arma(texto: str) -> float:
    t = sem_acentos_min(texto)
    hits_trafico = sum(1 for termo in TRAFICO_TERMS if sem_acentos_min(termo) in t)
    hits_arma = sum(1 for termo in ARM_TERMS if sem_acentos_min(termo) in t)

    if hits_trafico >= 2 and hits_arma <= 2:
        return 0.75
    if hits_trafico >= 3:
        return 0.60
    return 1.0

def doc_relevante_para_query(pergunta: str, subfatos_doc: List[str], refs_legais: List[Dict[str, str]]) -> bool:
    p = sem_acentos_min(pergunta)
    texto_doc = " ".join(subfatos_doc)
    texto_doc_norm = sem_acentos_min(texto_doc)

    termos_query = [token for token in re.findall(r"\w+", p) if len(token) >= 4]
    hits_query = sum(1 for t in termos_query if t in texto_doc_norm)

    if eh_query_arma(pergunta):
        has_arm = any(sem_acentos_min(x) in texto_doc_norm for x in ARM_TERMS)
        has_lei_arma = any(("10.826" in r["lei"] or "10826" in r["lei"]) for r in refs_legais)
        return has_arm or has_lei_arma or hits_query >= MIN_RELEVANCE_HITS

    return hits_query >= MIN_RELEVANCE_HITS

def classificar_tipo_documento(subfatos_doc: List[str], refs_legais: List[Dict[str, str]]) -> str:
    texto = " ".join(subfatos_doc)
    t = sem_acentos_min(texto)

    if any(("10.826" in r["lei"] or "10826" in r["lei"]) for r in refs_legais):
        if "porte" in t:
            return "porte_arma"
        if "posse" in t:
            return "posse_arma"
        return "arma"

    if any(sem_acentos_min(x) in t for x in TRAFICO_TERMS):
        return "trafico"

    if "roubo" in t and "arma" in t:
        return "roubo_com_arma"

    return "geral"

def bonus_tipo_documento(pergunta: str, subfatos_doc: List[str], refs_legais: List[Dict[str, str]]) -> float:
    if not eh_query_arma(pergunta):
        return 1.0

    tipo = classificar_tipo_documento(subfatos_doc, refs_legais)

    if tipo in {"posse_arma", "porte_arma", "arma"}:
        return 1.12

    if tipo == "roubo_com_arma":
        return 1.03

    if tipo == "trafico":
        return 0.70

    return 1.0

# =====================================================
# RESUMO DE CASOS
# =====================================================

def montar_resumo_caso_llm(item_ranking: Dict[str, Any]) -> str:
    processo = item_ranking["processo"]
    score = round(float(item_ranking["score_final"]), 4)
    subfatos = item_ranking["subfatos"][:3]
    refs = item_ranking.get("refs_legais", [])[:3]

    partes = [
        f"Processo: {processo}",
        f"Score final: {score}",
        "Trechos-chave:"
    ]

    for sf in subfatos:
        partes.append(f"- {sf}")

    if refs:
        partes.append("Referências legais:")
        for r in refs:
            if r["lei"] != "desconhecida":
                partes.append(f"- art. {r['artigo']} da Lei {r['lei']}")
            else:
                partes.append(f"- art. {r['artigo']}")

    return "\n".join(partes)

# =====================================================
# FALLBACK DETERMINÍSTICO
# =====================================================

def detectar_enquadramento_fallback(pergunta: str, ranking: List[Dict[str, Any]], refs: List[Dict[str, str]]) -> Tuple[str, str, str, str]:
    if not ranking:
        return (
            "ambíguo",
            "não identificado com segurança",
            "indeterminado",
            "baixa"
        )

    top = ranking[0]
    texto = sem_acentos_min(" ".join(top.get("subfatos", [])))

    enquadramento = "delito envolvendo arma de fogo"
    papel = "a arma aparece como objeto material do delito"
    pena = "depende do enquadramento específico"
    certeza = "média"

    if "posse" in texto and ("uso permitido" in texto or "arma de fogo" in texto):
        enquadramento = "posse irregular de arma de fogo"
        papel = "a arma funciona como objeto material do crime"
        pena = "em tese, detenção de 1 a 3 anos e multa"
        certeza = "alta"

    elif "porte" in texto and ("uso permitido" in texto or "arma de fogo" in texto):
        enquadramento = "porte ilegal de arma de fogo"
        papel = "a arma funciona como objeto material do crime"
        pena = "em tese, reclusão de 2 a 4 anos e multa"
        certeza = "alta"

    elif "municao" in texto or "munição" in texto or "acessorio" in texto or "acessório" in texto:
        enquadramento = "posse ilegal de munição/acessório de arma de fogo"
        papel = "arma, munição ou acessório integram o núcleo do tipo"
        pena = "depende se o caso é de uso permitido ou restrito"
        certeza = "média-alta"

    elif "roubo" in texto and "arma" in texto:
        enquadramento = "roubo com emprego de arma de fogo"
        papel = "a arma atua como majorante"
        pena = "depende da forma do roubo e do aumento aplicado"
        certeza = "média"

    base = "não identificada com segurança"
    if refs:
        r = refs[0]
        if r["lei"] != "desconhecida":
            base = f"art. {r['artigo']} da Lei {r['lei']}"
        else:
            base = f"art. {r['artigo']}"

    return enquadramento, base, papel, pena, certeza

def gerar_resposta_fallback(pergunta: str, ranking: List[Dict[str, Any]], refs: List[Dict[str, str]]) -> str:
    enquadramento, base, papel, pena, certeza = detectar_enquadramento_fallback(pergunta, ranking, refs)

    observacao = "Há ambiguidade residual porque a pergunta é genérica."
    if certeza == "alta":
        observacao = "Os casos recuperados convergem para esse enquadramento com boa consistência."

    return (
        "Resposta final:\n"
        f"- Enquadramento provável: {enquadramento}.\n"
        f"- Base legal provável: {base}.\n"
        f"- Papel da arma de fogo: {papel}.\n"
        f"- Pena em tese: {pena}.\n"
        f"- Grau de certeza: {certeza}. {observacao}"
    )

# =====================================================
# LLM
# =====================================================

def resposta_llm_valida(texto: str) -> bool:
    t = limpar_texto(texto)
    if len(t) < 60:
        return False

    lixo = [
        "Órgão julgador:",
        "Relator:",
        "Data de publicação:",
        "Ementa:",
        "Processo:"
    ]
    if sum(1 for x in lixo if x in t) >= 3:
        return False

    campos_esperados = [
        "Enquadramento provável",
        "Base legal provável",
        "Papel da arma de fogo",
        "Pena em tese",
        "Grau de certeza"
    ]
    hits = sum(1 for c in campos_esperados if c in t)
    return hits >= 4

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

Sua tarefa é responder com precisão técnica e sem copiar ementas.
Use apenas os casos e referências abaixo.

Pergunta do usuário:
{pergunta}

Casos recuperados:
{casos}

Referências legais extraídas:
{refs_txt}

Instruções:
1. Identifique o enquadramento mais provável.
2. Diferencie posse, porte e uso de arma em outro crime.
3. Só mencione artigo se houver apoio nos casos.
4. Não misture Lei de Drogas se a pergunta estiver centrada em arma de fogo.
5. Seja técnico, curto e objetivo.
6. Máximo de 8 linhas.

Responda exatamente neste formato:

Resposta final:
- Enquadramento provável:
- Base legal provável:
- Papel da arma de fogo:
- Pena em tese:
- Grau de certeza:
""".strip()

def gerar_resposta_llm(pergunta: str, ranking: List[Dict[str, Any]], refs: List[Dict[str, str]]) -> str:
    print("\n🧠 GERANDO RESPOSTA LLM...")

    prompt = gerar_prompt_llm(pergunta, ranking, refs)

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
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.08
    )

    generated_ids = outputs[0][prompt_len:]
    resposta = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("\n📤 RESPOSTA BRUTA LLM:")
    print(resposta)

    if "Resposta final:" not in resposta and "Resposta:" in resposta:
        resposta = resposta.replace("Resposta:", "Resposta final:", 1)

    if not resposta_llm_valida(resposta):
        print("⚠️ Saída do LLM inválida. Aplicando fallback determinístico.")
        return gerar_resposta_fallback(pergunta, ranking, refs)

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
# PÓS-PROCESSAMENTO DO RANKING
# =====================================================

def score_final_documento(pergunta: str, score_maxsim: float, subfatos_doc: List[str], refs_legais: List[Dict[str, str]]) -> float:
    bonus = bonus_tipo_documento(pergunta, subfatos_doc, refs_legais)
    penal = 1.0

    if eh_query_arma(pergunta):
        penal = penalidade_trafico_em_query_arma(" ".join(subfatos_doc))

    return float(score_maxsim * bonus * penal)

def selecionar_refs_do_ranking(pergunta: str, ranking: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    refs = []
    for item in ranking:
        refs.extend(item.get("refs_legais", []))
    return filtrar_referencias_para_query(pergunta, refs)

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
            print("   ↳ descartado pelo filtro de contexto")
            continue

        candidatos.append(idx_doc)

    if not candidatos:
        print("⚠️ Filtro removeu tudo, usando candidatos originais do FAISS")
        candidatos = [idx_doc for idx_doc in I[0] if 0 <= idx_doc < len(DOC_META)]

    # 4. Reranking MAXSIM + SUM
    resultados = []

    print("\n[RERANK] MAXSIM + SUM")
    for idx_doc in candidatos:
        d_emb = DOC_SUB_EMB[idx_doc]
        d_sub = DOC_SUB[idx_doc]
        meta = DOC_META[idx_doc]

        s_maxsim = score_maxsim_sum(q_emb, d_emb)
        s_final = score_final_documento(pergunta, s_maxsim, d_sub, meta["refs_legais"])
        tipo_doc = classificar_tipo_documento(d_sub, meta["refs_legais"])

        print(f"[AJUSTE DOMÍNIO] tipo={tipo_doc} | score_base={s_maxsim:.4f} | score_final={s_final:.4f}")

        resultados.append({
            "id": int(idx_doc),
            "processo": meta["processo"],
            "score": float(s_maxsim),
            "score_final": float(s_final),
            "tipo_documento": tipo_doc,
            "texto": meta["texto_curto"],
            "subfatos": d_sub,
            "refs_legais": meta["refs_legais"],
        })

    ranking = sorted(resultados, key=lambda x: x["score_final"], reverse=True)

    # 5. Referências legais focadas no ranking final
    refs_priorizadas = selecionar_refs_do_ranking(pergunta, ranking[:TOP_K])

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