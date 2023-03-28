import os
import pinecone
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Dict, List, Tuple
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.base import Chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.chains import ConversationalRetrievalChain
load_dotenv()


class CustomChain(Chain, BaseModel):
    vstore: Pinecone
    chain: BaseCombineDocumentsChain
    key_word_extractor: Chain

    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        question = inputs["question"]
        chat_history_str = _get_chat_history(inputs["chat_history"])
        print(chat_history_str)
        print(question)
        print(inputs)
        if chat_history_str:
            new_question = self.key_word_extractor.run(
                question=question, chat_history=chat_history_str
            )
            # print(new_question)
        else:
            new_question = question
        print(new_question)
        docs = self.vstore.similarity_search(new_question, k=4)
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        new_inputs["chat_history"] = chat_history_str
        answer, _ = self.chain.combine_docs(docs, **new_inputs)
        return {"answer": answer}
    
def get_new_chain1() -> Chain:
    
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"]
    )
    index_name = os.environ["PINECONE_INDEX_NAME"]
    namespace = os.environ["PINECONE_NAME_SPACE"]

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    eg_store = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)

    _eg_template = """## Example:

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question: {answer}"""

    _eg_prompt = PromptTemplate(
        template=_eg_template,
        input_variables=["chat_history", "question", "answer"],
    )

    _prefix = """Given the following conversation and a follow up question, rephrase the follow up question to be a 
    standalone question. You should assume that the question is related to LangChain."""
    _suffix = """## Example:

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    example_selector = SemanticSimilarityExampleSelector(vectorstore=eg_store, k=4)
    prompt = FewShotPromptTemplate(
        prefix=_prefix,
        suffix=_suffix,
        example_selector=example_selector,
        example_prompt=_eg_prompt,
        input_variables=["question", "chat_history"],
    )
    llm = OpenAI(temperature=0, model_name="text-davinci-003")
    key_word_extractor = LLMChain(llm=llm, prompt=prompt)

    EXAMPLE_PROMPT = PromptTemplate(
        template=">Example:\nContent:\n---------\n{page_content}\n----------\nSource: {source}",
        input_variables=["page_content", "source"],
    )
    template = """You are an AI assistant for the open source library LangChain. The documentation is located at https://langchain.readthedocs.io.
You are given the following extracted parts of a long document and a question. Provide a conversational answer with a hyperlink to the documentation.
You should only use hyperlinks that are explicitly listed as a source in the context. Do NOT make up a hyperlink that is not listed.
If the question includes a request for code, provide a code block directly from the documentation.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about LangChain, politely inform them that you are tuned to only answer questions about LangChain.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
    PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])
    doc_chain = load_qa_chain(
        OpenAI(temperature=0, model_name="text-davinci-003"),
        chain_type="stuff",
        prompt=PROMPT,
        document_prompt=EXAMPLE_PROMPT,
    )
    return ConversationalRetrievalChain(
        combine_docs_chain=doc_chain, retriever=eg_store.as_retriever(), question_generator=key_word_extractor
    )


def _get_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = ""
    for human_s, ai_s in chat_history:
        human = f"Human: " + human_s
        ai = f"Assistant: " + ai_s
        buffer += "\n" + "\n".join([human, ai])
    return buffer


chat_history=[("My name is Robert", "Hi Robert"), ("Who are you?", "I am an AI chatbot")]
chain = get_new_chain1()

inputs = {"question": "what is my name?", "chat_history": chat_history}
print(inputs["question"])
print(inputs["chat_history"])

print(chain.run(inputs))