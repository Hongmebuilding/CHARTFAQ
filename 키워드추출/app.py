# app.py
import re
import numpy as np
import pandas as pd
import streamlit as st
from langchain_ollama import OllamaEmbeddings

# =========================
# 설정
# =========================
CSV_PATH = "data/차트게시판 FAQ.CSV"
EMBED_MODEL = "bge-m3"

TOP_N = 5
HIGH_CONF = 0.80
MID_CONF  = 0.60
MAX_POOL = 400  # 키워드 없을 때 후보 제한

DOMAIN_TERMS = [
    "전자차트", "종이차트", "스캔", "이관", "백업", "보관", "보존", "기록",
    "업로드", "문서", "차팅", "마이크로필름", "광디스크", "의료법",
    "포스트잇", "진료버튼", "진료 버튼", "버튼", "메모"
]

SYNONYMS = {
    "전자 차트": "전자차트",
    "종이 차트": "종이차트",
    "진료 버튼": "진료버튼",
    "스캔 해야": "스캔해야",
    "스캔 해야하": "스캔해야하",
}

# =========================
# 유틸
# =========================
def canonical(text: str) -> str:
    s = str(text).strip()
    for k, v in SYNONYMS.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s)
    return s

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def extract_keywords(question: str) -> list[str]:
    q = canonical(question)
    return [t for t in DOMAIN_TERMS if t in q]

def split_multi_intent(user_question: str) -> list[str]:
    q = canonical(user_question)
    seps = [" 그리고 ", " 및 ", ",", " & ", " and "]
    for sep in seps:
        if sep in q:
            parts = [p.strip() for p in q.split(sep) if p.strip()]
            if len(parts) >= 2:
                return parts[:2]
    return [q]

def format_no_data_message():
    return (
        "입력하신 내용만으로는 문의 내용을 확인하기 어렵습니다.\n"
        "불편하신 기능(메뉴/버튼 이름)과 상황을 함께 적어 주세요.\n"
        "예) \"진료버튼이 안 보여요(어느 화면에서?)\", \"포스트잇이 여러 개 떠요(언제부터?)\""
    )

# =========================
# 데이터/임베딩 로딩 (캐시)
# =========================
@st.cache_resource
def get_embedder():
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://ollama:11434"
        )

@st.cache_data
def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Q"] = df["Q"].astype(str)
    df["A"] = df["A"].astype(str)
    return df

@st.cache_resource
def build_embeddings(df: pd.DataFrame):
    emb = get_embedder()
    qs = [canonical(x) for x in df["Q"].tolist()]
    vecs = emb.embed_documents(qs)
    return np.array(vecs, dtype=np.float32)

def filter_candidates(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    if not keywords:
        return df.head(MAX_POOL).copy() if len(df) > MAX_POOL else df.copy()
    pat = "|".join(map(re.escape, keywords))
    mask = df["Q"].str.contains(pat, regex=True) | df["A"].str.contains(pat, regex=True)
    cand = df[mask].copy()
    if len(cand) > MAX_POOL:
        cand = cand.head(MAX_POOL).copy()
    return cand

def retrieve_top_candidates(df: pd.DataFrame, q_embs: np.ndarray, user_question: str):
    emb = get_embedder()
    user_q = canonical(user_question)
    keywords = extract_keywords(user_q)

    cand_df = filter_candidates(df, keywords)
    if cand_df.empty:
        return [], keywords

    cand_idx = cand_df.index.to_list()
    cand_vecs = q_embs[cand_idx]
    uvec = np.array(emb.embed_query(user_q), dtype=np.float32)

    scored = []
    for i, vec in zip(cand_idx, cand_vecs):
        sim = cosine(uvec, vec)
        scored.append((i, sim))

    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for i, sim in scored[:TOP_N]:
        row = df.loc[i]
        results.append({
            "sim": float(sim),
            "Q": row["Q"],
            "A": row["A"],
            "No.": row.get("No.", None),
        })
    return results, keywords

def decide_response_for_one_question(df, q_embs, q: str):
    results, keywords = retrieve_top_candidates(df, q_embs, q)
    if not results:
        return {"type": "no_data", "text": format_no_data_message(), "cands": []}

    top_sim = results[0]["sim"]

    if top_sim >= HIGH_CONF:
        return {"type": "answer", "text": results[0]["A"], "cands": results, "top_sim": top_sim, "keywords": keywords}

    if top_sim >= MID_CONF:
        cand_qs = [r["Q"] for r in results[:3]]
        text = (
            "아래 질문 중에 가장 가까운 항목이 있으면 선택해 질문을 해주세요.\n"
            + "\n".join([f"{i+1}) {cq}" for i, cq in enumerate(cand_qs)])
        )
        return {"type": "choose", "text": text, "cands": results, "top_sim": top_sim, "keywords": keywords}

    return {"type": "no_data", "text": format_no_data_message(), "cands": results, "top_sim": top_sim, "keywords": keywords}



def answer_user(df, q_embs, user_question: str) -> str:
    sub_qs = split_multi_intent(user_question)

    blocks = []
    for idx, q in enumerate(sub_qs, start=1):
        r = decide_response_for_one_question(df, q_embs, q)

        # 멀티 질문이면 번호만, 단일이면 번호도 생략
        prefix = f"{idx}) " if len(sub_qs) > 1 else ""
        blocks.append(prefix + r["text"])

    return "\n\n".join(blocks)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="DentiumLink FAQ 챗봇", layout="wide")
st.title("DentiumLink FAQ 챗봇")

df = load_df(CSV_PATH)
q_embs = build_embeddings(df)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요. 궁금한 내용을 입력해 주세요."}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("질문을 입력하세요 (예: 포스트잇 여러 개가 떠요)")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    reply = answer_user(df, q_embs, prompt)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

st.caption("종료하려면 브라우저 탭을 닫으면 됩니다.")
