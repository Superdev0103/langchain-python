import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from flasgger import Swagger, LazyJSONEncoder
from flasgger import swag_from
from config import *

from query_data import get_chain

load_dotenv()

app = Flask(__name__)
cors = CORS(app)
app.json_encoder = LazyJSONEncoder

swaager = Swagger(app, template=swagger_template, config=swagger_config)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin()
def home():
    return "Hello!!!!"

@swag_from("question.yml", methods=['POST'])
@app.route("/api/chat", methods=['POST'])
@cross_origin()
def get_response():
    req_data = request.get_json()

    print(req_data)
    query = req_data['text']

    # initialize a connection to the Pinecone service
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"]
    )

    # load variable from .env
    index_name = os.environ["PINECONE_INDEX_NAME"]
    namespace = os.environ["PINECONE_NAME_SPACE"]

    # load the embeddings from OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    
    # load a Pinecone vector store
    doclist = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)

    # load Conversational Retrieval Chain
    custom_chain = get_chain(doclist)
    chat_history=[("My name is Robert", "Hi Robert"), ("Who are you?", "I am an AI chatbot")]
    buffer = ""
    for human_s, ai_s in chat_history:
        human = f"Question: " + human_s
        ai = f"Answer in Markdown: " + ai_s
        buffer += "\n" + "\n".join([human, ai])

    # get answer with question
    result = custom_chain.run(question=query, chat_history=chat_history)
    
    
    return result

if __name__ == "__main__":
    app.run()