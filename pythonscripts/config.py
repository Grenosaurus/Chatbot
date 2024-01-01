class ConfigTraining:
    DATA_PATH = '../data_files/'
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    GLOB_PATTERN = '*.pdf'


class ConfigLlmModel:
    PRE_TRAINED_LLM_MODEL = 'TheBloke/Llama-2-13B-Chat-GGUF'
    MODEL_FILE = 'llama-2-13b-chat.Q5_K_M.gguf'
    MODEL_TYPE = 'llama'
    MODEL_CONFIG = {'max_new_tokens': 512,
                    'temperature': 0.5,
                    'context_length': 2048}


class ConfigGeneric:
    DATABASE_FAISS_PATH = '../vectorstores/db_faiss'
    MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


class ConfigUI:
    PAGE_TITLE = 'Janu Chatbot'
    PAGE_ICON = ':robot_face:'
    PAGE_LAYOUT = 'wide'
    TITLE_ICON = ':basketball:'
    BOT_ICON = ':robot_face:'
    HISTORY_ICON = ':books:'


class ConfigLog:
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s: %(lineno)s - %(message)s'
