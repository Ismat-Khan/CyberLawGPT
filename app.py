import io
import os
import re
from typing import List, Dict, Tuple

import numpy as np
import requests
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "CYBERLAWGPT"

PDF_URL = (
    "https://www.pakistancode.gov.pk/"
    "pdffiles/administrator6a061efe0ed5bd153fa8b79b8eb4cba7.pdf"
)

SOURCE_NAME = "Prevention of Electronic Crimes Act, 2016"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SAFE CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(20, 184, 166, 0.12),
                transparent 35%
            ),
            #07111f;
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: #0b1728;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .hero-box {
        padding: 24px 0 18px 0;
    }

    .hero-subtitle {
        color: #a9b7c8;
        font-size: 17px;
        line-height: 1.6;
        margin-top: -8px;
        margin-bottom: 20px;
    }

    .metric-card {
        background: #0d1b2d;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 15px;
        padding: 16px;
        text-align: center;
    }

    .metric-number {
        font-size: 25px;
        font-weight: 800;
        color: #5eead4;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 13px;
    }

    .section-label {
        color: #cbd5e1;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .disclaimer {
        color: #94a3b8;
        font-size: 12px;
        line-height: 1.5;
        padding: 15px;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin-top: 30px;
    }

    div[data-testid="stChatMessage"] {
        background: rgba(13, 27, 45, 0.55);
        border-radius: 16px;
    }

    .stButton > button {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None


# ============================================================
# GROQ API KEY
# ============================================================

def get_groq_api_key() -> str:

    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""

    if key:
        return str(key).strip()

    return os.environ.get("GROQ_API_KEY", "").strip()


# ============================================================
# DOWNLOAD OFFICIAL PDF
# ============================================================

@st.cache_data(show_spinner=False)
def download_pdf() -> bytes:

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/130.0 Safari/537.36"
        )
    }

    response = requests.get(
        PDF_URL,
        headers=headers,
        timeout=60,
    )

    response.raise_for_status()

    content = response.content

    if not content.startswith(b"%PDF"):
        raise ValueError(
            "The downloaded file does not appear to be a valid PDF."
        )

    return content


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pages(pdf_bytes: bytes) -> List[str]:

    reader = PdfReader(io.BytesIO(pdf_bytes))

    pages = []

    for page in reader.pages:

        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = re.sub(r"\s+", " ", text).strip()

        pages.append(text)

    return pages


# ============================================================
# SECTION DETECTION
# ============================================================

def detect_section(text: str) -> str:

    patterns = [
        r"\bSection\s+(\d+[A-Za-z]?)",
        r"\b(\d+[A-Za-z]?)\.\s+[A-Z]",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return f"Section {match.group(1)}"

    return "General"


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    pages: List[str],
    words_per_chunk: int = 180,
    overlap: int = 40,
) -> List[Dict]:

    chunks = []

    for page_number, page_text in enumerate(
        pages,
        start=1,
    ):

        if not page_text.strip():
            continue

        words = page_text.split()

        start = 0

        while start < len(words):

            end = min(
                start + words_per_chunk,
                len(words),
            )

            chunk_text = " ".join(
                words[start:end]
            ).strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number,
                        "section": detect_section(chunk_text),
                    }
                )

            if end >= len(words):
                break

            start = max(
                0,
                end - overlap,
            )

    return chunks


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():

    return SentenceTransformer(
        EMBED_MODEL
    )


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

@st.cache_data(show_spinner=False)
def create_embeddings(
    texts: Tuple[str, ...],
) -> np.ndarray:

    model = load_embedding_model()

    embeddings = model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(
        embeddings,
        dtype=np.float32,
    )


# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

@st.cache_resource(show_spinner=False)
def build_knowledge_base():

    pdf_bytes = download_pdf()

    pages = extract_pages(
        pdf_bytes
    )

    chunks = create_chunks(
        pages
    )

    texts = tuple(
        chunk["text"]
        for chunk in chunks
    )

    embeddings = create_embeddings(
        texts
    )

    return chunks, embeddings


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(
    query: str,
    chunks: List[Dict],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> List[Dict]:

    model = load_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        item = dict(chunks[index])

        item["score"] = float(
            scores[index]
        )

        results.append(item)

    return results


# ============================================================
# RESPONSE LIMIT
# ============================================================

def get_response_limit(
    response_size: str,
) -> int:

    limits = {
        "Concise": 700,
        "Standard": 1200,
        "Detailed": 1800,
        "Very detailed": 2600,
    }

    return limits.get(
        response_size,
        1200,
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt(
    technicality: str,
    response_size: str,
    answer_language: str,
    answer_mode: str,
    citation_style: str,
) -> str:

    return f"""
You are CYBERLAWGPT, a retrieval-augmented legal information assistant.

Your knowledge base is the official Pakistan Code PDF containing:

{SOURCE_NAME}

IMPORTANT RULES:

1. Answer using the retrieved excerpts supplied in the user message.
2. Do not invent sections, clauses, punishments, authorities,
   procedures, dates, case law, or legal requirements.
3. If the retrieved evidence is insufficient, clearly say that
   the available source material does not provide enough information.
4. Distinguish between:
   - what the Act says,
   - plain-language explanation,
   - application to the user's hypothetical scenario.
5. Do not fabricate court cases or legal precedents.
6. If a question is unrelated to Pakistan's cyber laws,
   explain that it is outside the current knowledge base.
7. Do not claim to be a lawyer.
8. Give informational assistance, not personalized legal representation.
9. Where possible, identify the relevant section and page.
10. Never present unsupported information as if it came from the Act.

CURRENT SETTINGS:

Technicality: {technicality}
Response size: {response_size}
Language: {answer_language}
Answer mode: {answer_mode}
Citation style: {citation_style}

Keep the answer clear, readable, and practical.
"""


# ============================================================
# GROQ AI FUNCTION
# ============================================================

def ask_groq(
    question: str,
    context: List[Dict],
    technicality: str,
    response_size: str,
    answer_language: str,
    answer_mode: str,
    citation_style: str,
    model_name: str,
) -> str:

    api_key = get_groq_api_key()

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Please add it to Streamlit Secrets."
        )

    client = Groq(
        api_key=api_key
    )

    context_blocks = []

    for item in context:

        context_blocks.append(
            f"""
SOURCE
Section: {item["section"]}
Page: {item["page"]}
Relevance: {item["score"]:.3f}

TEXT:
{item["text"]}
"""
        )

    retrieved_context = "\n".join(
        context_blocks
    )

    system_prompt = build_system_prompt(
        technicality=technicality,
        response_size=response_size,
        answer_language=answer_language,
        answer_mode=answer_mode,
        citation_style=citation_style,
    )

    user_prompt = f"""
Answer this question using ONLY the retrieved source material.

QUESTION:
{question}

RETRIEVED SOURCE MATERIAL:
{retrieved_context}

If the source material does not adequately answer the question,
clearly state that the available evidence is insufficient.

Do not invent missing legal information.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        max_completion_tokens=get_response_limit(
            response_size
        ),
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError(
            "Groq returned an empty response."
        )

    return answer.strip()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ CYBERLAWGPT")

    st.caption(
        "Pakistan Cyber Law RAG Assistant"
    )

    st.divider()

    technicality = st.selectbox(
        "Technicality",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
            "Legal",
            "Technical",
        ],
        index=1,
    )

    response_size = st.selectbox(
        "Response Size",
        [
            "Concise",
            "Standard",
            "Detailed",
            "Very detailed",
        ],
        index=1,
    )

    answer_language = st.selectbox(
        "Answer Language",
        [
            "English",
            "Roman Urdu",
            "Urdu",
            "English + Roman Urdu",
        ],
        index=0,
    )

    answer_mode = st.selectbox(
        "Answer Mode",
        [
            "Legal explanation",
            "Section lookup",
            "Practical scenario",
            "Study / learning",
        ],
        index=0,
    )

    citation_style = st.selectbox(
        "Citation Style",
        [
            "Section + page",
            "Section only",
            "Detailed",
        ],
        index=0,
    )

    evidence_count = st.slider(
        "Evidence passages",
        min_value=3,
        max_value=8,
        value=5,
    )

    model_name = st.text_input(
        "Groq Model",
        value=DEFAULT_GROQ_MODEL,
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption(
        "Source: Prevention of Electronic Crimes Act, 2016"
    )


# ============================================================
# HERO
# IMPORTANT: NO HTML HERE
# ============================================================

st.caption(
    "🔐 AI-POWERED PAKISTAN CYBER LAW ASSISTANT"
)

st.title(
    "⚖️ CYBERLAWGPT"
)

st.markdown(
    "Ask questions about Pakistan's cyber laws and receive "
    "source-grounded answers using Retrieval-Augmented Generation "
    "and the Prevention of Electronic Crimes Act, 2016."
)


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

try:

    with st.spinner(
        "Loading Pakistan cyber law knowledge base..."
    ):

        chunks, embeddings = build_knowledge_base()

except Exception as error:

    st.error(
        "Could not load the cyber law PDF or create embeddings."
    )

    st.exception(error)

    st.stop()


# ============================================================
# METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Knowledge Chunks",
        len(chunks),
    )

with col2:

    st.metric(
        "Retrieval",
        "RAG",
    )

with col3:

    st.metric(
        "AI Generation",
        "Groq",
    )


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

st.markdown("### 💡 Try asking")

suggested_questions = [
    "What is unauthorized access under the Act?",
    "What does the law say about cyber stalking?",
    "What is electronic fraud?",
    "What does the Act say about false or fake information?",
]

question_columns = st.columns(2)

for index, suggestion in enumerate(
    suggested_questions
):

    with question_columns[index % 2]:

        if st.button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True,
        ):

            st.session_state.selected_question = (
                suggestion
            )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# QUESTION INPUT
# ============================================================

selected_question = st.session_state.selected_question

st.session_state.selected_question = None

question = st.chat_input(
    "Ask a question about Pakistan's cyber law..."
)

if selected_question:

    question = selected_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if not question:
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the cyber law and generating your answer..."
        ):

            try:

                retrieved = retrieve_documents(
                    query=question,
                    chunks=chunks,
                    embeddings=embeddings,
                    top_k=evidence_count,
                )

                answer = ask_groq(
                    question=question,
                    context=retrieved,
                    technicality=technicality,
                    response_size=response_size,
                    answer_language=answer_language,
                    answer_mode=answer_mode,
                    citation_style=citation_style,
                    model_name=model_name.strip(),
                )

                # IMPORTANT:
                # AI answer is rendered directly.
                # It is NOT inserted into custom HTML.

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                with st.expander(
                    "📚 Retrieved source evidence"
                ):

                    for index, item in enumerate(
                        retrieved,
                        start=1,
                    ):

                        st.markdown(
                            f"""
**Passage {index}**

**Section:** {item["section"]}

**Page:** {item["page"]}

**Relevance:** {item["score"]:.3f}

{item["text"]}
"""
                        )

                        if index < len(retrieved):
                            st.divider()

            except Exception as error:

                error_text = str(error)

                if (
                    "GROQ_API_KEY" in error_text
                    or "401" in error_text
                    or "authentication"
                    in error_text.lower()
                ):

                    st.error(
                        "Groq authentication failed. "
                        "Please check your GROQ_API_KEY "
                        "in Streamlit Secrets."
                    )

                    st.caption(
                        f"Technical detail: {error_text}"
                    )

                elif (
                    "model" in error_text.lower()
                    and (
                        "not found"
                        in error_text.lower()
                        or "invalid"
                        in error_text.lower()
                    )
                ):

                    st.error(
                        f"The Groq model '{model_name}' "
                        "could not be used."
                    )

                    st.caption(
                        f"Technical detail: {error_text}"
                    )

                else:

                    st.error(
                        "I couldn't generate the answer."
                    )

                    st.caption(
                        f"Technical detail: {error_text}"
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Important: CYBERLAWGPT provides AI-generated legal "
    "information based on the supplied Prevention of Electronic "
    "Crimes Act, 2016 PDF. It is not a substitute for advice "
    "from a qualified legal professional."
)
