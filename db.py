from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from chromadb import chromadb
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

PATH = "emb"
embeddings = OpenAIEmbeddings()

def get_documents():
  # Load the document
  loader = TextLoader("facts.txt")
  splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0,
  )

  return loader.load_and_split(text_splitter=splitter)

# Check if the database is populated with facts
def should_populate_db() -> bool:
  client = chromadb.PersistentClient(path=PATH)
  try:
    colls = client.get_collection("langchain")
    data = colls.get()
    return data.get("documents") is None
  except:
    return True

def populate_db():
  print("Populating database...")
  Chroma.from_documents(
    documents=get_documents(),
    embedding=embeddings,
    persist_directory=PATH
  )
  print("Done populating database")

if should_populate_db():
  populate_db()
else:
  print("DB already populated")

db = Chroma(
  persist_directory=PATH,
  embedding_function=embeddings,
)

retriever = db.as_retriever()
