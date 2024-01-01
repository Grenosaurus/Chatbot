# Python libraries
import time
import logging

from config import ConfigTraining, ConfigGeneric, ConfigLog
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


class DataProcessor:
    def __init__(self, data_path = ConfigTraining.DATA_PATH, glob_pattern = ConfigTraining.GLOB_PATTERN, loader_cls = PyPDFLoader):
        self.data_path = data_path
        self.glob_pattern = glob_pattern
        self.loader_cls = loader_cls

    def load_documents(self):
        try:
            logging.info(f"Loading documents from: {self.data_path}")
            data_loader = DirectoryLoader(self.data_path, glob=self.glob_pattern, loader_cls=self.loader_cls)
            return data_loader.load()
        except Exception as e:
            logging.error(f"Error loading documents: {e}")
            raise

    def split_documents(self, file_documents):
        try:
            logging.info("Splitting text from loaded documents.")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = ConfigTraining.CHUNK_SIZE,
                                                           chunk_overlap = ConfigTraining.CHUNK_OVERLAP)
            return text_splitter.split_documents(file_documents)
        except Exception as e:
            logging.error(f"Error splitting documents: {e}")
            raise


class EmbeddingDatabase:
    def __init__(self, data_processor, model_name = ConfigGeneric.MODEL_NAME):
        self.model_name = model_name
        self.data_processor = data_processor

    def create_embeddings(self):
        try:
            logging.info(f"Creating embeddings with: {self.model_name}")
            return HuggingFaceEmbeddings(model_name = self.model_name, model_kwargs = ConfigGeneric.MODEL_KWARGS)
        except Exception as e:
            logging.error(f"Error creating embeddings: {e}")
            raise

    def create_vector_database(self):
        try:
            logging.info("Creating vector databese from the splitted document text.")
            file_documents = self.data_processor.load_documents()
            chunked_texts = self.data_processor.split_documents(file_documents)
            embeddings = self.create_embeddings()
            logging.info(f"Saving the database in: {ConfigGeneric.DATABASE_FAISS_PATH}.")
            database = FAISS.from_documents(chunked_texts, embeddings)
            database.save_local(ConfigGeneric.DATABASE_FAISS_PATH)
        except Exception as e:
            logging.error(f"Error creating vector database: {e}")
            raise


def configure_logging():
    logging.basicConfig(format = ConfigLog.LOG_FORMAT, level = logging.INFO)


def measure_runtime(start_time):
    delta_time_seconds = round(time.perf_counter() - start_time, 2)
    delta_time_minutes = round(delta_time_seconds / 60, 2)

    logging.info(f"Program took {delta_time_seconds} seconds (~ {delta_time_minutes} minutes).")


if __name__ == '__main__':
    configure_logging()
    start_time = time.perf_counter()
    logging.info("Starting preprocessing of the dataset and initializing the embedding object.")
    data_processor = DataProcessor()
    embedding_database = EmbeddingDatabase(data_processor)
    logging.info("Embedding the text split database and creating a vector.")
    embedding_database.create_vector_database()
    logging.info("Database is ready for the LLM.")
    measure_runtime(start_time)
