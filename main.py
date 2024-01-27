from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from db import retriever as chroma_retriever

# Uncomment to enable debug mode
# import langchain
# langchain.debug = True

from dotenv import load_dotenv
load_dotenv()

chain = RetrievalQA.from_chain_type(
  llm=ChatOpenAI(verbose=True),
  retriever=chroma_retriever,
  chain_type="stuff",
  verbose=True,
)


while True:
  question = input(">> ")
  result = chain.run(f"Answer {question}. Plus point out the number of the fact you uses.")

  print(result)

  