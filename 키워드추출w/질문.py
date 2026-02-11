import re
import numpy as np
import pandas as pd
from langchain_ollama import OllamaEmbeddings

"""
사용자 질문
  ↓
(선택) multi-intent 분해(간단 규칙 기반)
  ↓
정규화(canonical)
  ↓
도메인 키워드 추출
  ↓
키워드 포함된 Q/A만 후보로 필터링 (키워드 없으면 전체 or 상위 N만)
  ↓
문장 유사도 랭킹
  ↓
확신/애매/불확실 구간에 따라
  - 답변 바로 제공
  - 후보 질문 제시(선택 유도)
  - 질문 다시 입력 유도(예시 포함)
"""

EMBED_MODEL = "bge-m3"
emb = OllamaEmbeddings(model=EMBED_MODEL)

TOP_N = 5

# 3단계 UX 기준(조절)
HIGH_CONF = 0.80   # 바로 답변
MID_CONF  = 0.60   # 후보 제시
# 그 미만이면: 재질문 유도

# 도메인 키워드 사전(계속 누적)
DOMAIN_TERMS = [
    "전자차트", "종이차트", "스캔", "이관", "백업", "보관", "보존", "기록",
    "업로드", "문서", "차팅", "마이크로필름", "광디스크", "의료법",
    # 화면/기능 관련도 추가(지금 테스트 케이스용)
    "포스트잇", "진료버튼", "진료 버튼", "버튼", "메모"
]

SYNONYMS = {
    "전자 차트": "전자차트",
    "종이 차트": "종이차트",
    "진료 버튼": "진료버튼",
    "스캔 해야": "스캔해야",
    "스캔 해야하": "스캔해야하",
}

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

def build_question_embeddings(df: pd.DataFrame) -> np.ndarray:
    # Q만 임베딩 (원하면 Q+A로 바꿔도 됨)
    qs = [canonical(x) for x in df["Q"].astype(str).tolist()]
    vecs = emb.embed_documents(qs)
    return np.array(vecs, dtype=np.float32)

def filter_candidates(df: pd.DataFrame, keywords: list[str], max_pool: int = 400) -> pd.DataFrame:
    """
    - 키워드 있으면: Q 또는 A에 키워드 포함된 row만
    - 키워드 없으면: 전체를 쓰되, 너무 많으면 max_pool로 제한(속도)
    """
    if not keywords:
        return df.head(max_pool).copy() if len(df) > max_pool else df.copy()

    pat = "|".join(map(re.escape, keywords))
    mask = (
        df["Q"].astype(str).str.contains(pat, regex=True) |
        df["A"].astype(str).str.contains(pat, regex=True)
    )
    cand = df[mask].copy()
    if len(cand) > max_pool:
        cand = cand.head(max_pool).copy()
    return cand

def retrieve_top_candidates(df: pd.DataFrame, q_embs: np.ndarray, user_question: str):
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
            "Q": str(row["Q"]),
            "A": str(row["A"]),
            "I": row.get("I", None),
            "No.": row.get("No.", None),
        })
    return results, keywords

def split_multi_intent(user_question: str) -> list[str]:
    """
    LLM 없이 간단 규칙 기반 분해.
    - '그리고', '및', ',' 같은 연결어가 있으면 2개로 쪼개는 수준
    - 너무 과하게 쪼개지 않게 2개까지만
    """
    q = canonical(user_question)
    # 흔한 연결어 기준 (필요시 추가)
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

def decide_response_for_one_question(q: str, df: pd.DataFrame, q_embs: np.ndarray):
    results, keywords = retrieve_top_candidates(df, q_embs, q)
    if not results:
        return {"question": q, "type": "no_data", "text": format_no_data_message(), "cands": []}

    top_sim = results[0]["sim"]

    # 확신 구간
    if top_sim >= HIGH_CONF:
        return {"question": q, "type": "answer", "text": results[0]["A"], "cands": results, "top_sim": top_sim, "keywords": keywords}

    # 애매 구간: 후보 제시
    if top_sim >= MID_CONF:
        cand_qs = [r["Q"] for r in results[:3]]
        text = (
            "아래 질문 중에 가장 가까운 항목이 있으면 번호로 선택해 주세요.\n"
            + "\n".join([f"{i+1}) {cq}" for i, cq in enumerate(cand_qs)])
        )
        return {"question": q, "type": "choose", "text": text, "cands": results, "top_sim": top_sim, "keywords": keywords}

    # 불확실 구간
    return {"question": q, "type": "no_data", "text": format_no_data_message(), "cands": results, "top_sim": top_sim, "keywords": keywords}

def answer_user(user_question: str, df: pd.DataFrame, q_embs: np.ndarray) -> str:
    sub_qs = split_multi_intent(user_question)

    blocks = []
    for idx, q in enumerate(sub_qs, start=1):
        r = decide_response_for_one_question(q, df, q_embs)

        # 보기 좋게 섹션화
        header = f"{idx}) {r['question']}" if len(sub_qs) > 1 else f"▶ {r['question']}"
        blocks.append(header + "\n" + r["text"])

    return "\n\n".join(blocks)

# ---------- 실행 ----------

df = pd.read_csv("QNA/data/차트게시판 FAQ.CSV")
df["Q"] = df["Q"].astype(str)
df["A"] = df["A"].astype(str)

q_embs = build_question_embeddings(df)

while True:
    user_q = input("사용자 질문 입력: ").strip()
    if user_q.lower() in ["exit", "quit"]:
        break
    print("\n최종 응답:\n", answer_user(user_q, df, q_embs))
    print("\n" + "=" * 50 + "\n")
