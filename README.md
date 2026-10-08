 LangChain RAG Prototype — Step-by-Step Setup

This project answers questions from your domain documents (PDFs) using
LangChain's Retrieval-Augmented Generation (RAG) pattern. It maps to the
4 group roles in your assignment brief.

## Step 1 — Install Python and set up the project

1. Make sure you have Python 3.10+ installed (`python --version`).
2. Open a terminal in this project folder.
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Mac/Linux
   venv\Scripts\activate         # Windows
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Step 2 — Get an OpenAI API key

1. Go to https://platform.openai.com/api-keys and create a key.
2. Copy `.env.example` to a new file named `.env`.
3. Paste your key in: `OPENAI_API_KEY=sk-...`
4. Never commit `.env` to GitHub or include it in your zip file.

## Step 3 — Add your domain documents (Student 2: Data & RAG Lead)

1. Pick your industry (Healthcare, Retail, Logistics, Streaming, etc.).
2. Gather 2–5 real or sample PDFs relevant to your chosen task
   (e.g., product manuals, FAQs, policy documents, intake forms).
3. Drop them into the `sample_docs/` folder.
4. If your documents are very long or short, adjust `CHUNK_SIZE` and
   `CHUNK_OVERLAP` near the top of `rag_script.py`.

## Step 4 — Design and test your system prompt (Student 1: GenAI & Prompts)

1. Open `rag_script.py` and find the `SYSTEM_PROMPT` variable.
2. Rewrite it to fit your chosen task and tone — for example, a
   customer-support assistant vs. a clinical-summary assistant will
   sound very different.
3. Use NotebookLM or any web GenAI tool first to explore your documents
   and draft/test candidate prompts before pasting the final version in.
4. Re-run the script after each prompt change and compare answers.

## Step 5 — Run the script (Student 4: Integration Lead)

```bash
python rag_script.py
```

What happens:
1. It loads and chunks every PDF in `sample_docs/`.
2. It embeds the chunks and stores them in a local Chroma vector
   database (`chroma_db/` folder, created automatically).
3. It runs two sample questions automatically, then lets you type
   your own questions interactively.
4. Take a screenshot of this terminal output for **Page 4** of your
   report ("Screenshots of Working Execution").

## Step 6 — (Optional) Add an n8n workflow (Student 3: Workflow Lead)

A simple way to wire this into n8n without rewriting code:

1. In n8n, add a **Trigger** node (e.g., Webhook, Schedule, or Email
   Trigger) — this represents "User Input" in your architecture
   diagram.
2. Add an **Execute Command** node that runs:
   ```
   python /full/path/to/rag_script.py
   ```
   (For a real integration you'd modify `rag_script.py` to accept the
   question as a command-line argument instead of typing it
   interactively — see the note at the bottom of this file.)
3. Add a final node (e.g., **Send Email** or **Respond to Webhook**) to
   deliver the answer — this represents "Final Output."
4. Export your workflow: **Workflow menu → Download** → save as
   `n8n_workflow.json` and include it in your code zip.
5. Alternatively, if your group used NotebookLM instead of n8n, take
   screenshots of your NotebookLM source-testing session for Page 3.

## Step 7 — Package your deliverables

- **Report (.docx):** Write up Pages 1–4 as described in the brief,
  using this project as the basis for Pages 2 and 3. Do NOT paste the
  full script into the report — just describe your changes and include
  short snippets if needed.
- **Code zip:** Include `rag_script.py`, `requirements.txt`,
  `.env.example` (NOT your real `.env`), your PDFs in `sample_docs/`,
  and `n8n_workflow.json` if used. Name it per the convention:
  `Group_[GroupNumber]_[DomainName]_Code.zip`.

## Making it accept a question as a command-line argument (for n8n)

If you want n8n to pass a specific question instead of using the
interactive prompt, replace the bottom of `main()` in `rag_script.py`
with:

```python
if len(sys.argv) > 1:
    question = " ".join(sys.argv[1:])
    ask(qa_chain, question)
else:
    # existing interactive loop
    ...
```

Then call it from n8n's Execute Command node as:
```
python rag_script.py "What are your store hours?"
```

## Troubleshooting

- **"No PDF files found"** — check that your PDFs are directly inside
  `sample_docs/`, not a subfolder.
- **Authentication error** — double-check `.env` has the correct key
  and no extra spaces/quotes.
- **Slow first run** — building the vector store calls the embeddings
  API once per chunk; subsequent runs reuse `chroma_db/` unless you
  delete it.
