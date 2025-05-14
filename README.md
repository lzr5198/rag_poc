# Local QA System using LangChain, FAISS, and Mistral 7B

## Problem Interpretation

As the client's employees spend significant time searching through documents, which implies low efficiency, the client wants to enhance the information accessibility by implementing a Question-Answering system. This project serves as a proof-of-concept of the system and demonstrates the feasibility of this solution.

## Proposed Solution & Rationale

This Proof of Concept (PoC) implements a **Retrieval-Augmented Generation (RAG)** system using local tools to ensure **privacy, cost-efficiency, and control**. The system allows users to query internal documents and get contextually grounded, natural language answers.

### Technical Approach

- **Document Processing**: We use `TextLoader` from LangChain to load and convert `.txt` files into `Document` objects for further processing.
- **Embedding Model**: We use `all-MiniLM-L6-v2` from Hugging Face for creating dense vector embeddings. This model offers a good balance of accuracy, speed, and resource efficiency.
- **Vector Store**: FAISS is employed to store and retrieve document embeddings quickly based on similarity.
- **Local LLM**: The `mistral-7b-instruct` model is run locally via `llama.cpp` using the `LlamaCpp` wrapper, which ensures zero dependency on external APIs and maintains full data privacy.
- **QA Chain**: LangChain’s `RetrievalQA` combines the retriever and LLM into a pipeline that first retrieves relevant documents and then generates answers based on those.

### Why These Tools?

| Requirement          | Chosen Tool                     | Reason                                                                 |
|----------------------|----------------------------------|------------------------------------------------------------------------|
| Privacy              | Local Mistral model via `llama.cpp` | No API calls or data exposure to third-party services.                 |
| Speed & Efficiency   | `MiniLM` + FAISS                | Fast embedding and vector search ideal for small/medium corpora.       |

## Limitations

- **Local inference latency** can be higher than hosted APIs, especially on CPU-only machines.
- **MiniLM embeddings** may not capture very fine-grained semantics compared to larger models like BGE or E5.
- **Limited document formats** in current implementation (only `.txt` files are supported).

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/lzr5198/rag_poc.git
cd rag_poc
```

### 2. Create Environment and Install Dependencies
```bash
conda create -n rag python=3.11
conda activate rag
pip install -r requirements.txt
```

### 3. Download the Mistral Model

Download the model file mistral-7b-instruct-v0.1.Q2_K.gguf from [TheBloke’s HuggingFace repo](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF) or another trusted source.

Place the file in a local directory, for example:

```bash
/your/custom/path/to/mistral-7b-instruct-v0.1.Q2_K.gguf
```
> Important: Update the path to your downloaded model in the code:

```python
llm = LlamaCpp(
  model_path="/your/custom/path/to/mistral-7b-instruct-v0.1.Q2_K.gguf",
  ...
)
```

### 4. Prepare Document Files
Put your .txt documents into the docs folder.

**Project Structure**

```bash
rag_poc/
├──docs/
  └── filename1.txt
  └── filename2.txt
  └── filename3.txt
  └── ...
├── rag_poc.py
├── requirements.txt
├── README.md
```


## How to use
Run the script:

```bash
python rag_poc.py "your question"
```
For example:

```bash
python rag_poc.py "What's the strategic objectives that we will prioritize?"
# We will prioritize the following strategic objectives: Enhance Customer Retention, Drive Product Innovation, and Strengthen Operational Efficiency.
```
