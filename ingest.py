from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os
from dotenv import load_dotenv
load_dotenv()

# load unstructured text data from a file
loader = UnstructuredFileLoader("test.txt")
data = loader.load()

# Splitting the loaded text data into chunks of 1000 characters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

# Generating embeddings for each chunk of text
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# initialize pinecone
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT"]
)

# load Pinecone config from .env
index_name = os.environ["PINECONE_INDEX_NAME"]
namespace = os.environ["PINECONE_NAME_SPACE"]

# Add the embeddings of each chunk of text as a document to the Pinecone index
docsearch = Pinecone.from_texts(
    [text for text in texts], embeddings,
    index_name=index_name, namespace=namespace
)
print("docsearch~~~~~~~", docsearch)