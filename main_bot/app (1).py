from flask import Flask, render_template, request,jsonify
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from langchain.chains import RetrievalQA
from flask_cors import CORS
import os
import pathlib
import creds
os.environ["OPENAI_API_KEY"]=creds.api_key
app = Flask(__name__)
CORS(app)
# Load documents
def load_docs(directory):
    loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    return documents

directory = pathlib.Path('C:/main_bot')  # Update with your directory path
documents = load_docs(directory)
# Initialize components
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)
text = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings()
persist_directory = 'db'
vectordb = Chroma.from_documents(documents=text, embedding=embedding, persist_directory=persist_directory)
vectordb.persist()
# vectordb=None
# vectordb=Chroma(persist_directory=persist_directory,
#                 embedding_function=embedding)
retriever=vectordb.as_retriever()
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(batch_size=5), chain_type="stuff", retriever=retriever,
                                       return_source_documents=True)

# Define route for home page
# @app.route('/')
# def home():
#     return render_template('index.html')
def process_llm_response(llm_response):
    return llm_response['result']
def get_response(query):
    llm_response = qa_chain(query)
    output = process_llm_response(llm_response)
    if "don't" in output.lower():
        # Return a predefined link
        return f"Sorry, i am unable to answer your question. Please visit the website <a href='https://www.iiests.ac.in/' target='_blank'>@iiest.official.site</a>"
    return output
@app.post("/predict")
def predict():
    query = request.json.get("message")
    response = get_response(query)
    message= {"answer": response}
    return jsonify(message)

#     response = get_response(query)
#     message= {"answer": response}
#     return jsonify(message)
if __name__ == '__main__':
     app.run(debug=True)