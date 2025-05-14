import os
import argparse
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

def load_documents(doc_dir):
  filenames = [f for f in os.listdir(doc_dir) if os.path.isfile(os.path.join(doc_dir, f))]

  # Load all text files into LangChain Document objects
  docs = []
  for file in filenames:
    try:
      loader = TextLoader(os.path.join(doc_dir, file))
      loaded = loader.load()
      print(f"Loaded {len(loaded)} docs from {file}")
      # Add loaded docs to the main list
      docs.extend(loaded)
    except Exception as e:
      print(f"Failed to load {file}: {e}")
  return docs

def main(question):
  # Use local language model: mistral
  llm = LlamaCpp(
    model_path="/Users/kevinlin/.models/mistral-7b-instruct-v0.1.Q2_K.gguf",
    # model_path="/path/to/model",
    n_ctx=2048,
    temperature=0.7
  )

  doc_dir = "./docs"
  docs = load_documents(doc_dir)

  # Convert documents into embeddings using a pre-trained HuggingFace model
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  # Store vector embeddings in FAISS index for fast similarity search
  vectorstore = FAISS.from_documents(docs, embeddings)

  # Create a QA system
  qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

  # Run the question through the QA system
  result = qa.run(question)
  print(result)

if __name__ == "__main__":
  # Set up argument parsing
  parser = argparse.ArgumentParser(description="Run a question-answering system with your document store.")
  parser.add_argument('question', type=str, help='The question to ask the QA system.')
  
  args = parser.parse_args()
  main(args.question)