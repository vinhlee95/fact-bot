

### Inspect ChromaDB
```shell
pipenv shell
python

>>> import chromadb
>>> client = chromadb.PersistentClient(path="emb")
>>> coll = client.get_collection("langchain")
>>> coll
Collection(name=langchain)
>>> coll.get()
{'ids': [], 'embeddings': None, 'metadatas': [], 'documents': [], 'uris': None, 'data': None}
```