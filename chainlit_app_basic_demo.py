import os
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
import chainlit as cl
from dotenv import load_dotenv
# from langchain_community.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

# Load environment variables
load_dotenv('.env')

YOUR_API_KEY=" "

prompt_template = ChatPromptTemplate.from_messages(
    [
    ("system", 
    """
    You are a friendly assistant that answers user's question about astronomy.
    If the user's question is not about these topics, 
    respond with "Uh-oh! I do not have the information to answer your question. Ask me about Space, Planets and Stars!".
    """
    ),
    ("user", "{question}\n"),
    ]
)

@cl.on_chat_start
def main():
    # Instantiate required classes for the user session
    # llm_chat = AzureChatOpenAI(
    #         deployment_name=os.getenv("OPENAI_CHAT_MODEL"),
    #         temperature=0.0,
    #         max_tokens=100
    # )
    llm_chat = ChatOpenAI(base_url="https://api.openai.com/v1/", temperature=0.0, max_tokens=100, openai_api_key=YOUR_API_KEY, model="gpt-3.5-turbo-0613", streaming=True)
    llm_chain = LLMChain(prompt=prompt_template, llm=llm_chat, verbose=True, memory=ConversationBufferWindowMemory(k=2))

    # Store the chain in the user session for reusability
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")
    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["text"]).send()