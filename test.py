from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain import OpenAI, LLMChain, PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

conv_memory = ConversationBufferMemory(
    memory_key="chat_history_lines",
    input_key="input"
)

template = """You are a chatbot having a conversation with a human.

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], 
    template=template
)
# memory = ConversationSummaryMemory(llm=OpenAI(),memory_key="chat_history")

llm_chain = LLMChain(
    llm=OpenAI(), 
    prompt=prompt, 
    verbose=True, 
    # memory=memory,
)
chat_history=[("My name is Robert", "Hi Robert"), ("Who are you?", "I am an AI chatbot")]
buffer = ""
for human_s, ai_s in chat_history:
    human = f"Human: " + human_s
    ai = f"Chatbot: " + ai_s
    buffer += "\n" + "\n".join([human, ai])

print("Result: "+llm_chain.run(human_input="Hi there my friend", chat_history=buffer))
print("Result: "+llm_chain.run(human_input="Not to bad - how are you?", chat_history=buffer))
print("Result: "+llm_chain.run(human_input="Not to bad - how are you?", chat_history=buffer))
print("Result: "+llm_chain.run(human_input="What is my name?", chat_history=buffer))