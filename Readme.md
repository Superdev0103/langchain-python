# 🦜️🔗 A Chatbot with Langchain and Pinecone

This repo is an implementation of a locally hosted chatbot specifically focused on question answering over the [LangChain documentation](https://langchain.readthedocs.io/en/latest/).
Built with [LangChain](https://github.com/hwchase17/langchain/) and [Flask](https://flask.palletsprojects.com/en/2.2.x/api/).

Tech stack used includes LangChain, Pinecone, Openai, and Python. LangChain is a framework that makes it easier to build scalable AI/LLM apps and chatbots. Pinecone is a vectorstore for storing embeddings in text to later retrieve similar docs.

## ✅ Running locally
### Set environment variables:
Copy .env.example to .env
```
OPENAI_API_KEY=<Your OpenAI Key>
PINECONE_ENVIRONMENT=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
PINECONE_NAME_SPACE=
```
1. Create virtual environment: `python -m venv .venv`
1. Activate the environment: `.venv\Scripts\activate.bat`
1. Install dependencies: `pip install -r requirements.txt`
1. Run `ingest.sh` to ingest LangChain docs data into the vectorstore (only needs to be done once).
   1. You can use other [Document Loaders](https://langchain.readthedocs.io/en/latest/modules/document_loaders.html) to load your own data into the vectorstore.
1. Run the app: `flask run --host=0.0.0.0 port=9000`
   1. `langchain-server` is running locally.
1. API is Deployed in  [localhost:9000](http://localhost:9000).

## 🚀 Important Links

Swagger UI: [Swagger for chatbot API](https://localhost:5000/docs)

## 📚 Technical description

There are two components: ingestion and question-answering.

Ingestion has the following steps:

1. Pull html from documentation site
2. Load html with LangChain's ReadTheDocs Loader
3. Split documents with LangChain's [TextSplitter](https://langchain.readthedocs.io/en/latest/reference/modules/text_splitter.html)
4. Create a vectorstore of embeddings, using LangChain's [vectorstore wrapper](https://langchain.readthedocs.io/en/latest/reference/modules/vectorstore.html) (with OpenAI's embeddings and Pinecone vectorstore).

### 👩‍💻 Code Explanation

```python
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
```

Question-Answering has the following steps, all handled by [ConversationalRetrievalChain](https://python.langchain.com/en/harrison-docs-refactor-3-24/modules/chains/index_examples/chat_vector_db.html?highlight=ConversationalRetrievalChain):

1. Given the chat history and new user input, determine what a standalone question would be (using GPT-3).
2. Given that standalone question, look up relevant documents from the vectorstore.
3. Pass the standalone question and relevant documents to GPT-3 to generate a final answer.

### 👩‍💻 Code Explanation

app.py
#### 
```python
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

# get answer with question
result = custom_chain.run(question=query, chat_history=chat_history)
```

query_data.py
```python
# initialized with an OpenAI language model and a key to store chat history
memory = ConversationSummaryMemory(llm=OpenAI(), memory_key="chat_history")

# make the template for question generator component of the chain
CONDENSE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")

# make template for document retriever component of the chain
QA_PROMPT = PromptTemplate.from_template(
  """You are an AI assistant providing helpful advice. You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided.
If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer.

Question: {question}
=========
{context}
=========
Answer in Markdown:""")

# create ConversationalRetrievalChain with Pinecone vectore store for question and answer
def get_chain(
        vectorstore: Pinecone
) -> ConversationalRetrievalChain:

    # determine OpenAI Language model for Q/A generator 
    llm=OpenAI(temperature=0)
    streaming_llm=ChatOpenAI(temperature=0)

    # create chain for generating standalone questions from follow-up inputs, using the conversation history as context
    question_generator = LLMChain(
        llm=llm,
        memory=memory,
        prompt=CONDENSE_PROMPT
    )

    # create chain for retrieving relevant documents from the vectorstore based on the rephrased question generated by question generator
    doc_chain = load_qa_chain(
        streaming_llm,
        chain_type="stuff",
        prompt=QA_PROMPT
    )

    # return ConversationRetrievalChain that answers user questions based on a given document store
    return ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
    )
```
