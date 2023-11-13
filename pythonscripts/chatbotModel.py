# Python libraries
import chainlit as cl

from config import ConfigLlmModel, ConfigGeneric
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS


custom_prompt_template = '''
Use the following piece of information to anwser the user's question. If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.

Context: {context}
Question: {question}

Only returns the helpful anwser below and nothing else.
Helpful anwser:
'''


def set_custom_prompt():
    return PromptTemplate(template = custom_prompt_template, input_variables = ['context', 'question'])


def load_llm_model(llm_model, model_type):
    return CTransformers(model = llm_model,
                         model_type = model_type,
                         config = {'max_new_tokens' : 512,
                                   'temperature' : 0.5,
                                   'context_length' : 2048})


def create_retrieval_qa_chain(llm, prompt, database):
    return RetrievalQA.from_chain_type(llm = llm,
                                       chain_type = 'stuff',
                                       retriever = database.as_retriever(search_kwargs = {'k' : 2}),
                                       return_source_documents = True,
                                       chain_type_kwargs = {'prompt' : prompt})


def create_embeddings(model_name):
    return HuggingFaceEmbeddings(model_name = model_name, model_kwargs = {'device' : 'cpu'})


def create_qa_chatbot():
    embeddings = create_embeddings(ConfigGeneric.MODEL_NAME)
    database = FAISS.load_local(ConfigGeneric.DATABASE_FAISS_PATH, embeddings)
    llm = load_llm_model(ConfigLlmModel.PRE_TRAINED_LLM_MODEL, ConfigLlmModel.MODEL_TYPE)
    qa_prompt = set_custom_prompt()

    return create_retrieval_qa_chain(llm, qa_prompt, database)


def final_result(query):
    qa_result = create_qa_chatbot()

    return qa_result({'query' : query})



@cl.on_chat_start
async def start():
    chain = create_qa_chatbot()

    msg = cl.Message(content = "Starting the chatbot...")
    await msg.send()

    msg.content = "Hi, Welcome to the LLaMA 2 Chatbot. How can I assist you?"
    await msg.update()

    cl.user_session.set('chain', chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get('chain')
    callback = cl.AsyncLangchainCallbackHandler(stream_final_answer = True,
                                                answer_prefix_tokens = ['FINAL', 'ANSWER'])

    callback.answer_reached = True
    result = await chain.acall(message.content, callbacks = [callback])

    answer = result['result']
    sources = result['source_documents']

    '''finalContent = f"Sources: " + str(sources) if sources else f"No sources found for the answer!"
    msg = cl.Message(content = finalContent)
    await msg.send()'''

