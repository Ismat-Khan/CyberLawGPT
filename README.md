# ⚖️ CYBERLAWGPT

CYBERLAWGPT is a Python + Streamlit Retrieval-Augmented Generation (RAG) application for asking questions about Pakistan's **Prevention of Electronic Crimes Act, 2016**.

The application automatically downloads the official Pakistan Code PDF at startup, extracts its text, creates local semantic embeddings, retrieves the most relevant legal passages for each question, and sends only those retrieved passages to **Groq** for the final explanation.

## ✨ Features

- Official Pakistan Code PDF downloaded automatically at startup
- Local semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Cosine-similarity RAG retrieval
- Section/page-aware evidence
- Groq generation through the official Groq Python SDK
- Technicality control:
  - Beginner
  - Intermediate
  - Advanced
  - Legal / Technical
- Response-size control:
  - Concise
  - Standard
  - Detailed
  - Very detailed
- Answer language:
  - English
  - Urdu
  - Roman Urdu
  - English + Roman Urdu
- Answer mode:
  - Legal explanation
  - Section lookup
  - Practical scenario
  - Study / learning
- Evidence passage slider
- Citation style control
- Suggested cyber-law questions
- Conversation history
- Retrieved evidence viewer
- Streamlit Cloud compatible
- Colab compatible
- No database or paid vector database required

## 📚 Legal source

The application uses this official Pakistan Code PDF:

`https://www.pakistancode.gov.pk/pdffiles/administrator6a061efe0ed5bd153fa8b79b8eb4cba7.pdf`

The app is deliberately grounded in this document instead of relying on the model's general legal knowledge.

## 🧠 How the RAG pipeline works

```text
Official Pakistan Code PDF
          ↓
       Download
          ↓
     PDF extraction
          ↓
   Section-aware chunks
          ↓
MiniLM semantic embeddings
          ↓
      User question
          ↓
Question embedding
          ↓
Cosine similarity retrieval
          ↓
Top legal passages
          ↓
        Groq
          ↓
Grounded legal explanation
```

The model is instructed not to invent a section, punishment, procedure, case law, or legal outcome when the retrieved source does not establish it.

## ⚖️ Important legal limitation

CYBERLAWGPT is a **legal-information and educational application**, not a law firm or lawyer.

A response can explain what the supplied Act says and show the relevant source passage, but it cannot determine the final legal outcome of an individual case. For urgent, disputed, criminal, or otherwise high-stakes matters, consult a qualified Pakistani lawyer or the appropriate authority.

The supplied PDF is treated as the application's legal source. If the law is amended or replaced, update the source URL/document and review the application before relying on it.

## 🔎 Example questions

- What is unauthorized access under the Act?
- Explain electronic fraud in simple language.
- What does the Act say about cyber stalking?
- What is cyberbullying?
- What does the Act say about false and fake information?
- What is unauthorized copying or transmission of data?
- Explain cyber terrorism under the Act.
- What does the Act say about identity information?
- Which section relates to spoofing?
- Explain the relevant law for a hypothetical cyber incident.

## 📌 Source

Pakistan Code, Ministry of Law and Justice:

Prevention of Electronic Crimes Act, 2016.
