"""
=========================================================================
RAG SCRIPT — LangChain Retrieval-Augmented Generation Prototype
=========================================================================
Assignment role mapping:
  - Student 1 (GenAI & Prompts): edit SYSTEM_PROMPT below and test outputs.
  - Student 2 (Data & RAG Lead): put your domain PDFs in ./sample_docs/
    and adjust CHUNK_SIZE / CHUNK_OVERLAP if needed.
  - Student 3 (Workflow Lead): this script can be triggered by an n8n
    "Execute Command" node — see README.md.
  - Student 4 (Integration Lead): run `python rag_script.py` end-to-end,
    screenshot the terminal output for the report.

HOW IT WORKS (User Input -> RAG Script -> Final Output):
  1. Load all PDFs from ./sample_docs/
  2. Split them into chunks
  3. Embed chunks and store them in a local Chroma vector database
  4. On each question, retrieve the most relevant chunks
  5. Feed the question + retrieved chunks to the LLM using SYSTEM_PROMPT
  6. Print the answer
=========================================================================
"""

import os
import sys
import glob

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -------------------------------------------------------------------
# CONFIG — Student 2 (Data & RAG Lead) edits this section
# -------------------------------------------------------------------
DOCS_FOLDER = "./sample_docs"      # put your domain PDFs here
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
VECTORSTORE_DIR = "./chroma_db"    # local vector database folder
MODEL_NAME = "gpt-4o-mini"         # swap for any model your API key supports

# -------------------------------------------------------------------
# SYSTEM PROMPT — Student 1 (GenAI & Prompts) edits this section
# -------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful AI assistant for a business.
Answer the user's question using ONLY the context provided below.
If the answer is not contained in the context, say "I don't have that
information in the provided documents" — do not make things up.
Be concise and professional.

Context:
{context}

Question: {question}

Answer:"""


def load_and_split_documents(folder_path: str):
    """Load every PDF in folder_path and split into chunks."""
    pdf_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_paths:
        print(f"[WARNING] No PDF files found in {folder_path}. "
              f"Add your domain documents there before running.")
        sys.exit(1)

    all_docs = []
    for path in pdf_paths:
        print(f"Loading: {path}")
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Loaded {len(pdf_paths)} PDF(s) -> split into {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks):
    """Embed chunks and store/persist them in Chroma."""
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )
    return vectorstore


def build_qa_chain(vectorstore):
    """Wire up retriever + LLM + custom prompt into a RetrievalQA chain."""
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain


def ask(qa_chain, question: str):
    """Run one question through the RAG pipeline and print the answer."""
    result = qa_chain.invoke({"query": question})
    print("\n" + "=" * 70)
    print(f"Q: {question}")
    print("-" * 70)
    print(f"A: {result['result']}")
    print("-" * 70)
    print("Sources used:")
    for doc in result["source_documents"]:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "?")
        print(f"  - {source} (page {page})")
    print("=" * 70 + "\n")


def main():
    load_dotenv()  # reads OPENAI_API_KEY from .env

    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not found. Copy .env.example to .env "
              "and add your key.")
        sys.exit(1)

    print("Step 1/3: Loading and splitting documents...")
    chunks = load_and_split_documents(DOCS_FOLDER)

    print("Step 2/3: Building vector store (this calls the embeddings API)...")
    vectorstore = build_vectorstore(chunks)

    print("Step 3/3: Building QA chain. Ready for questions!\n")
    qa_chain = build_qa_chain(vectorstore)

    # ---- Student 3 / Student 4: sample test questions for screenshots ----
    sample_questions = [
        "What is this document about?",
        "Summarize the key points in two sentences.",
    ]

    for q in sample_questions:
        ask(qa_chain, q)

    # ---- Interactive mode: run more questions live for your demo/screenshot ----
    print("Type a question and press Enter (or type 'exit' to quit).")
    while True:
        user_q = input("Your question: ").strip()
        if user_q.lower() in ("exit", "quit", ""):
            break
        ask(qa_chain, user_q)


if __name__ == "__main__":
    main()
