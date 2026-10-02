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

## 🔐 API key

CYBERLAWGPT uses the **xAI / Groq API**.

Set:

```text
GROQ_API_KEY=your_xai_api_key
```

Do not put the real API key directly into `app.py` or commit it to GitHub.

For Streamlit Community Cloud, add `GROQ_API_KEY` under the app's Secrets settings.

For Google Colab:

```python
import os
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
```

Then run:

```python
!streamlit run app.py &>/content/logs.txt &
```

If using a Colab tunnel, expose the Streamlit port with your preferred tunneling method.

> Important: Streamlit Community Cloud and Google Colab can be used on their free tiers, but an Groq API key may require an account with available API credits. Free hosting does not mean the Groq API itself is necessarily free.

## 💻 Run locally

Create a folder containing exactly:

```text
CYBERLAWGPT/
├── app.py
├── requirements.txt
└── README.md
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the API key.

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Linux/macOS:

```bash
export GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Run:

```bash
streamlit run app.py
```

The application will download the official PDF and create its embeddings automatically.

## ☁️ Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload only:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Open Streamlit Community Cloud.
4. Create a new app.
5. Select your GitHub repository.
6. Set the entrypoint to `app.py`.
7. In Advanced settings → Secrets, add:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

8. Deploy.

Do not upload a `secrets.toml` file containing your real API key.

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

## 🛠️ Why only three files?

This project intentionally keeps the deployment simple:

```text
app.py
requirements.txt
README.md
```

There is no separate vector database, `.env` file, model file, local PDF, or backend service. The PDF and embedding model are obtained at runtime and cached by Streamlit.

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
