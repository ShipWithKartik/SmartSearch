from dotenv import load_dotenv
load_dotenv()

import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch

from rag.config_loader import config
from rag.utils.mongodb_helper import get_mongo_collection, create_vector_search_index
from rag.utils.prepare_test_data import get_input_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_data():
    """This method will initialize the MongoDB collection with some sample data """
    database_name = config["database_name"]
    collection_name = config["collection_name"]
    vector_index_name = config["vector_index_name"]
    dimensions = config["embedding_model_dimensions"]
    similarity = config["similarity"]

    docs = get_input_data()

    embeddings = HuggingFaceEmbeddings(model_name=config["embedding_model"])

    collection = get_mongo_collection(db_name=database_name, collection_name=collection_name)

    create_vector_search_index(
        collection=collection,
        index_name=vector_index_name,
        embedded_field_names=["embedding"],
        dimensions=dimensions,
        similarity=similarity,
        filter_fields_with_datatype={
            "rating": "number",
            "release_date": "token",
            "genre": "token",
            "director": "token"
        }
    )

    MongoDBAtlasVectorSearch.from_documents(docs, embeddings, collection=collection)

    logger.info("Initialization completed successfully")


initialize_data()
