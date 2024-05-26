from langchain_community.llms import ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from flask import Flask, request

cached_llm = ollama.Ollama(model="llama3")
db_path = "db"
embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False)
raw_prompt = PromptTemplate.from_template(""" 
<s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s> 
[INST] {input}
     Context: {context}
      Answer:                                                                              
""")

app = Flask(__name__)

@app.post("/pdf")
def loadpdfintodb():
    file = request.files["file"]
    file_name = file.filename
    print(f"Uploaded file={file_name}")
    save_file = "./data/" + file_name
    file.save(save_file)
    print(f"Uploaded file={file_name}")
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len ={len(docs)}")
    chunks = text_splitter.split_documents(docs)
    print(f"chunks len ={len(chunks)}")
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=db_path
    )
    vector_store.persist()
    return {
        "status":"Successfully Loaded",
        "filename": file_name,
        "chunk_size":len(chunks),
        "docs_length":len(docs)
    }
 

@app.post("/ai")
def aipost():
    json_content = request.json
    query = json_content.get("query")
    print(" ::: ", query)
    response = cached_llm.invoke(query)
    return response

@app.post("/ask_pdf")
def ask_pdf():
    json_content = request.json
    query = json_content.get("query")
    print(" ::: ", query)
    vector_store = Chroma(
        persist_directory=db_path, embedding_function=embedding
    )
    retriever= vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k":20,
            "score_threshold":0.1
        }
    )
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input":query})
    sources = []
    for doc in result["context"]:
        sources.append({
            "source": doc.metadata["source"],
            "page_content": doc.page_content
        })
    response_answer = {"answer": result["answer"], "sources":sources}
    return response_answer
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)