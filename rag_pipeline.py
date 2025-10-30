import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# --- CONFIGURATION ---
PDF_PATH = "attention.pdf"
CHROMA_DB_PATH = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_MODEL_NAME = "google/flan-t5-large" # Hugging Face LLM for generation

def index_documents():
    """1. Load, 2. Split, 3. Embed, and 4. Store documents."""
    
# 1. Load Documents
    print(f"Loading document: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    try:
        documents = loader.load()
    except FileNotFoundError:
        print(f"\nFATAL ERROR: The file '{PDF_PATH}' was not found in your directory.")
        print("Please ensure you have placed 'attention.pdf' into C:\\Users\\VISHWATEJA\\Desktop\\Rag_Project\\")
        return None, None # Exit gracefully
        
# 2. Split Text (Chunking)
    print(f"Splitting into chunks (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

# 3. Create Embeddings (Jina Embeddings)
    print(f"Initializing Jina Embeddings model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Use 'cuda' if you have a GPU
    )

# 4. Store in ChromaDB
    print(f"Storing embeddings in ChromaDB at: {CHROMA_DB_PATH}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    vector_store.persist()
    print("Indexing complete. Vector store persisted.")
    return vector_store, embeddings

def get_rag_chain(embeddings):
    """Sets up the RAG chain by integrating the retriever and the LLM."""
# 1. Load the Retriever (from ChromaDB)
    print(f"Loading ChromaDB from {CHROMA_DB_PATH}...")
    vector_store_loaded = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    # k=3 retrieves the top 3 most relevant chunks
    retriever = vector_store_loaded.as_retriever(search_kwargs={"k": 3})

# 2. Setup the Hugging Face LLM Pipeline
    print(f"Loading Hugging Face model: {HF_MODEL_NAME}")
    
    # Configure the transformers pipeline (for local LLM)
    hf_pipe = pipeline(
        "text2text-generation", 
        model=HF_MODEL_NAME,
        tokenizer=HF_MODEL_NAME,
        max_new_tokens=256,
        torch_dtype=torch.bfloat16, 
        device='cpu' 
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)

# 3. Create the Retrieval-Augmented Generation Chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'stuff' puts all retrieved docs into the prompt
        retriever=retriever,
        return_source_documents=True
    )
    return rag_chain

def run_query(rag_chain, query):
    """Executes the RAG query and prints the result and sources."""
    print(f"\n--- Query: {query} ---")
    
    # Use .invoke() for modern LangChain versions
    response = rag_chain.invoke({"query": query})
    
    print("\n✅ *Answer:*")
    # Ensure only the text result is printed
    if isinstance(response['result'], str):
        print(response['result'])
    else:
        # Fallback if the output structure is different
        print(response['result'])
    
    print("\n📚 *Sources:* (Grounding the answer in specific document content)")
    for i, doc in enumerate(response['source_documents']):
        print(f"--- Source {i+1} ---")
        page_num = doc.metadata.get('page', 'N/A')
        print(f"Page: {page_num}")
        print(f"Snippet: {doc.page_content[:200]}...")
    
    return response

if __name__ == "__main__":
    
    # 0. Initialize embeddings once for both indexing and retrieval
    embeddings_func = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    # 1. Check if DB exists, if not, create it (Indexing Phase)
    if not os.path.exists(CHROMA_DB_PATH):
        vector_store, embeddings = index_documents()
        if vector_store is None: # Exit if PDF was missing
            exit() 
    else:
        print(f"ChromaDB already exists at {CHROMA_DB_PATH}. Skipping indexing.")

    # 2. Setup the RAG system (Retrieval & Generation Phase)
    rag_system = get_rag_chain(embeddings_func)
    
    # 3. *--- CONTINUOUS QUERY LOOP ADDED HERE ---*
    print("\n--- RAG Pipeline Ready ---")
    print("Ask questions about the document, or type 'quit' to exit.")
    
    while True:
        # Get input from the user
        user_query = input("\nEnter your question: ")
        
        # Check if the user wants to quit
        if user_query.lower() in ["quit", "exit"]:
            print("Exiting RAG pipeline. Goodbye!")
            break
        
        # Run the RAG query
        try:
            run_query(rag_system, user_query)
        
        except Exception as e:
            print(f"An error occurred during query execution: {e}")