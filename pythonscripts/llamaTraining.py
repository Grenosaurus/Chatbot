# Python libraries
import time

from config import ConfigTraining, ConfigGeneric
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS



class DataProcessor:
    def __init__(self, data_path, glob_pattern, loader_cls):
        self.data_path = data_path
        self.glob_pattern = glob_pattern
        self.loader_cls = loader_cls


    def load_documents(self):
        data_loader = DirectoryLoader(self.data_path, glob = self.glob_pattern, loader_cls = self.loader_cls)

        return data_loader.load()


    def split_documents(self, file_documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = ConfigTraining.CHUNK_SIZE, chunk_overlap = ConfigTraining.CHUNK_OVERLAP)

        return text_splitter.split_documents(file_documents)



class EmbeddingDatabase:
    def __init__(self, model_name, data_processor):
        self.model_name = model_name
        self.data_processor = data_processor


    def create_embeddings(self):
        return HuggingFaceEmbeddings(model_name = self.model_name, model_kwargs = {'device' : 'cpu'})


    def create_vector_database(self):
        file_documents = self.data_processor.load_documents()
        chunked_texts = self.data_processor.split_documents(file_documents)
        embeddings = self.create_embeddings()

        database = FAISS.from_documents(chunked_texts, embeddings)
        database.save_local(ConfigGeneric.DATABASE_FAISS_PATH)



def measure_runtime(start_time):
    delta_time_seconds = round(time.time() - start_time, 2)
    delta_time_minutes = round(delta_time_seconds / 60, 2)

    print(f"\nProgram took {delta_time_seconds} seconds (~ {delta_time_minutes} minutes).")


if __name__ == '__main__':
    start_time = time.time()

    data_processor = DataProcessor(ConfigTraining.DATA_PATH, ConfigTraining.GLOB_PATTERN, PyPDFLoader)
    embedding_database = EmbeddingDatabase(ConfigGeneric.MODEL_NAME, data_processor)

    embedding_database.create_vector_database()

    measure_runtime(start_time)

