import re
import numpy as np
import pandas as pd
from langchain_ollama import OllamaEmbeddings
'''
사용자 질문
  ↓
정규화(canonical)
  ↓
도메인 키워드 추출 (전자차트 / 종이차트 / 스캔 …)
  ↓
키워드 포함된 Q/A만 후보로 필터링
  ↓
문장 유사도(+ 키워드 히트 보너스)로 랭킹
  ↓
1등 Q/A 선택
'''

EMBED_MODEL = "bge-m3"
emb = OllamaEmbeddings(model=EMBED_MODEL)

TOP_N = 5                  # 최종 후보 몇 개 보여줄지
SIM_THRESHOLD = 0.70       # cosine similarity 기준(조절 필요)

# 도메인 키워드 사전(필요한 것만 계속 추가)
DOMAIN_TERMS = [
    "전자차트", "종이차트", "스캔", "이관", "백업", "보관", "보존", "기록",
    "업로드", "문서", "차팅", "마이크로필름", "광디스크", "의료법"
]

# 동의어/표기 흔들림 보정(최소만)
SYNONYMS = {
    "전자 차트": "전자차트",
    "종이 차트": "종이차트",
    "스캔 해야": "스캔해야",
    "스캔 해야하": "스캔해야하",
}

def canonical(text: str) -> str:
    s = str(text).strip()
    for k, v in SYNONYMS.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s)
    return s

def extract_keywords(question: str) -> list[str]:
    q = canonical(question)
    kws = [t for t in DOMAIN_TERMS if t in q]
    return kws

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def build_question_embeddings(df: pd.DataFrame) -> np.ndarray:
    # Q 임베딩만 미리 만들어두면 속도/정확도 좋아짐
    qs = [canonical(x) for x in df["Q"].astype(str).tolist()]
    vecs = emb.embed_documents(qs)  # list[list[float]]
    return np.array(vecs, dtype=np.float32)

def filter_candidates(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    if not keywords:
        return df  # 키워드 없으면 전체 대상(또는 intent 필터로 좁혀도 됨)

    pat = "|".join(map(re.escape, keywords))
    mask = df["Q"].astype(str).str.contains(pat, regex=True) | df["A"].astype(str).str.contains(pat, regex=True)
    return df[mask].copy()

def retrieve_by_keyword_then_similarity(df: pd.DataFrame, q_embs: np.ndarray, user_question: str):
    user_q = canonical(user_question)
    keywords = extract_keywords(user_q)

    cand_df = filter_candidates(df, keywords)
    if cand_df.empty:
        return [], keywords

    # 후보 row index로 임베딩 뽑기
    cand_idx = cand_df.index.to_list()
    cand_vecs = q_embs[cand_idx]

    uvec = np.array(emb.embed_query(user_q), dtype=np.float32)

    scored = []
    for i, vec in zip(cand_idx, cand_vecs):
        sim = cosine(uvec, vec)
        scored.append((i, sim))

    scored.sort(key=lambda x: x[1], reverse=True)

    # 상위 TOP_N 반환
    results = []
    for i, sim in scored[:TOP_N]:
        row = df.loc[i]
        results.append({
            "sim": round(sim, 4),
            "Q": row["Q"],
            "A": row["A"],
            "I": row.get("I", None),
            "No.": row.get("No.", None),
        })

    return results, keywords

df = pd.read_csv("QNA/data/차트게시판 FAQ.CSV")
df["Q"] = df["Q"].astype(str)
df["A"] = df["A"].astype(str)

q_embs = build_question_embeddings(df)

user_q = "포스트잇 여러개 뜨는게 불편하고 진료버튼이 안보여요"
results, keywords = retrieve_by_keyword_then_similarity(df, q_embs, user_q)

print("키워드:", keywords)
for r in results:
    print(r["sim"], r["Q"])

if results and results[0]["sim"] >= SIM_THRESHOLD:
    answer = results[0]["A"]
else:
    # 후보 질문 제시(사용자 선택 유도)
    candidate_questions = [r["Q"] for r in results[:3]]