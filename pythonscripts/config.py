class ConfigTraining:
    DATA_PATH = '../data_files/'
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    GLOB_PATTERN = '*.pdf'


class ConfigLlmModel:
    PRE_TRAINED_LLM_MODEL = '../llm_model/[LLM_MODEL]'
    MODEL_TYPE = 'llama'


class ConfigGeneric:
    DATABASE_FAISS_PATH = '../vectorstores/db_faiss'
    MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

