from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from db import retriever as chroma_retriever, db as chroma_db
from custom_retriever import CustomRetriever
from langchain.embeddings import OpenAIEmbeddings

# Uncomment to enable debug mode
# import langchain
# langchain.debug = True

def get_retriever():
  if True:
    return CustomRetriever(
      embeddings=OpenAIEmbeddings(),
      chroma=chroma_db,
    )
  else:
    return chroma_retriever

chain = RetrievalQA.from_chain_type(
  llm=ChatOpenAI(verbose=True),
  retriever=get_retriever(),
  chain_type="stuff",
  verbose=True,
)


while True:
  question = input(">> ")
  result = chain.run(f"Answer {question}. Plus point out the number of the fact you uses.")

  print(result)

  