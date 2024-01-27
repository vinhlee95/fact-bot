from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from chromadb import chromadb

# Uncomment to enable debug mode
# import langchain
# langchain.debug = True

from dotenv import load_dotenv

load_dotenv()
chat = ChatOpenAI(verbose=True)

# Load the document
loader = TextLoader("facts.txt")
splitter = CharacterTextSplitter(
  separator="\n",
  chunk_size=200,
  chunk_overlap=0,
)

embeddings = OpenAIEmbeddings()

# Check if the database is populated with facts
def should_populate_db() -> bool:
  client = chromadb.PersistentClient(path="emb")
  try:
    colls = client.get_collection("langchain")
    data = colls.get()
    return data.get("documents") is None
  except:
    return True

if should_populate_db():
  print("Populating database...")
  db = Chroma.from_documents(
    documents=loader.load_and_split(text_splitter=splitter),
    embedding=embeddings,
    persist_directory="emb"
  )
  print("Done populating database")
else:
  print("DB already populated")

db = Chroma(
  persist_directory="emb",
  embedding_function=embeddings,
)

chain = RetrievalQA.from_chain_type(
  llm=chat,
  retriever=db.as_retriever(),
  chain_type="stuff",
  verbose=True,
)


while True:
  question = input(">> ")
  result = chain.run(f"Answer {question}. Plus point out the number of the fact you uses.")

  print(result)

  