## Chatbot PoC

사내 고객지원 FAQ를 기반으로 한 문서 검색형 챗봇 PoC입니다.

### Overview
- 사용자 질문을 임베딩하여 FAQ 문서에서 유사 항목을 검색
- 검색된 문서만을 기반으로 답변을 구성
- 답변 생성이 아닌 문장 정리 수준으로 LLM을 활용

### Architecture
User Question  
→ Embedding  
→ Vector Search (Top-K)  
→ Retrieved FAQ Context  
→ LLM (Answer Formatting)  
→ Response

### Features
- FAQ 기반 질의 응답
- 검색 실패 시 상담 문의 안내
- Streamlit 기반 간단한 채팅 UI

### Tech Stack
- Python
- FAISS (Vector Search)
- SentenceTransformer (Embedding)
- Ollama (LLM)
- Streamlit (UI)

### Note
본 프로젝트는 PoC 목적이며,  
지식 데이터는 자동 학습하지 않고 검수 후 관리하는 구조를 전제로 합니다.

### Run
```bash
pip install -r requirements.txt
streamlit run app.py

