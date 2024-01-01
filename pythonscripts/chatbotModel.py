# Python libraries
import logging

from config import ConfigLlmModel, ConfigGeneric
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers


CUSTOM_PROMPT_TEMPLATE = '''
Use the provided context to anwser the user questions. Read the given context before providing the anwsers for the questions and think step by
step. If you can not answer a user question based on the provided context,inform the user. Do not use any other information for answering user.
Provide a detailed answer to the question.
'''
SYSTEM_PROMPT = f'<<SYS>>\n{CUSTOM_PROMPT_TEMPLATE}\n<</SYS>>\n\n'


class LlmChatbot:
    def __init__(self, history = False):
        self.qa_prompt, self.qa_memory = self.set_custom_prompt(history = history)
        self.qa_chain = self.create_qa_chatbot(history = history)
        self.history = []

    def set_custom_prompt(self, history = False):
        logging.info("Creating custom prompt.")
        template_input_variables = ['history', 'context', 'question'] if history else ['context', 'question']
        instruction = f'''Context: {{history}}\n{{context}}\nQuestion: {{question}}\n''' if history else f'''Context: {{context}}\nQuestion: {{question}}\n'''
        prompt_template = f'[INST]{SYSTEM_PROMPT}{instruction}[/INST]'
        return (PromptTemplate(input_variables = template_input_variables, template = prompt_template),
                ConversationBufferMemory(input_key = 'question', memory_key = 'history'))

    def load_llm_model(self):
        try:
            logging.debug(f"Loading the pretrained LLM model: {ConfigLlmModel.MODEL_TYPE}")
            return CTransformers(model = ConfigLlmModel.PRE_TRAINED_LLM_MODEL,
                                 model_file = ConfigLlmModel.MODEL_FILE,
                                 model_type = ConfigLlmModel.MODEL_TYPE,
                                 config = ConfigLlmModel.MODEL_CONFIG)
        except Exception as e:
            logging.error(f"Error loading LLM model: {e}")

    def create_retrieval_qa_chain(self, llm, prompt, memory, history, database):
        logging.debug(f"Chat history: {history}")
        chain_type_kwargs = {'prompt': prompt, 'memory': memory} if history else {'prompt': prompt}
        return RetrievalQA.from_chain_type(llm = llm,
                                           chain_type = 'stuff',
                                           retriever = database.as_retriever(search_kwargs = {'k': 2}),
                                           return_source_documents = True,
                                           chain_type_kwargs = chain_type_kwargs)

    def create_embeddings(self):
        logging.debug(f"Creating embedding with: {ConfigGeneric.MODEL_NAME}")
        return HuggingFaceEmbeddings(model_name = ConfigGeneric.MODEL_NAME, model_kwargs = {'device': 'cpu'})

    def create_qa_chatbot(self, history = False):
        if not isinstance(history, bool):
            raise ValueError("History parameter must be a boolean.")

        logging.info(f"Creating chatbot.")
        embeddings = self.create_embeddings()
        database = FAISS.load_local(ConfigGeneric.DATABASE_FAISS_PATH,
                                    embeddings)
        llm = self.load_llm_model()
        return self.create_retrieval_qa_chain(llm, self.qa_prompt, self.qa_memory, history, database)

    def final_result(self, query, conversation_history = None):
        logging.debug(f"Conversation history: {'True' if conversation_history else 'False'}")
        return self.qa_chain({'query': query, 'history': conversation_history}) if conversation_history else self.qa_chain({'query': query})
