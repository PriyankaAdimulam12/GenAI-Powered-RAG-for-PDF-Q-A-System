# 🤖 GenAI-Powered RAG for PDF Q&A System

## Project Overview
This project implements a **Retrieval-Augmented Generation (RAG) system** capable of performing interactive, context-aware Question & Answer on complex PDF documents. The system operates fully locally, utilizing open-source models for both embedding (retrieval) and language generation, eliminating the need for paid cloud APIs.

**Key Achievements:**
* **Virtually eliminated LLM hallucinations** by strictly grounding all answers in the provided document context.
* **Engineered for local performance** using CPU-optimized embedding and inference models.
* Successfully indexed and queried **"Attention Is All You Need"** (the foundational Transformer paper) as a core knowledge base.

---

## 🛠️ Technology Stack
| Component | Technology Used | Role in Pipeline |
| :--- | :--- | :--- |
| **LLM (Generator)** | **Google's FLAN-T5 Large** (via Hugging Face) | Generates the final, human-readable answers. |
| **Embeddings (Retriever)** | **`sentence-transformers/all-MiniLM-L6-v2`** | Converts text chunks and queries into dense vector representations. |
| **Vector Database** | **ChromaDB** | Stores, indexes, and searches the vector embeddings. |
| **Framework** | **LangChain** | Manages the RAG pipeline (chaining the retriever and LLM). |
| **Language** | Python | Core development language. |

---

## 🚀 Getting Started

### Prerequisites
1.  Python 3.8+
2.  Your RAG project folder containing: `rag_pipeline.py`, `requirements.txt`, and the source document (`attention.pdf`).

### Installation
1.  **Clone or Download:**
    ```bash
    # If you install Git later, you can use this:
    # git clone [YOUR GITHUB REPO URL]
    ```
2.  **Install Dependencies:** Navigate to the project root directory and install all required libraries.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The model weights for FLAN-T5 Large (~1.5GB) and the MiniLM embeddings will be downloaded during the first run.*

---

## 💡 How to Run the System

The main script automatically initializes the vector database (`chroma_db`) if it doesn't exist, and then enters an interactive query loop.

1.  **Execute the script:**
    ```bash
    python rag_pipeline.py
    ```
2.  **Interactive Query Loop:** The system will prompt you to enter a question about the `attention.pdf` document.

    **Example Interaction:**
    ```
    --- RAG Pipeline Ready ---
    Ask questions about the document, or type 'quit' to exit.

    Enter your question: Name the three ways the Transformer model uses multi-head attention.

    --- Query: Name the three ways the Transformer model uses multi-head attention. ---

    ✅ *Answer:*
    The Transformer uses multi-head attention in three distinct ways: 1. In the encoder self-attention layers. 2. In the decoder self-attention layers. 3. In the encoder-decoder attention layer, where queries come from the previous decoder layer and keys and values come from the output of the encoder.

    📚 *Sources:* (Grounding the answer in specific document content)
    --- Source 1 ---
    Page: 5
    Snippet: The Transformer uses multi-head attention in three different ways: 1) In the encoder-decoder attention layer, the Queries come from the previous decoder layer, and the Keys and Values come from the output of the encoder...
    ```

---

## ⚠️ Files and Performance Notes

* **Optimized for CPU:** The embeddings model (`all-MiniLM-L6-v2`) was specifically chosen for its efficient performance on CPU, addressing the high latency typically associated with running LLMs locally.
* **Ignored Files:** The large vector database (`chroma_db`) and model cache files are excluded from this repository via the `.gitignore` file. To run the system, you only need the code files and the PDF.
