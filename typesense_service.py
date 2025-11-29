from typing import List, Optional, Dict, Any
import os
import logging
from dotenv import load_dotenv
import typesense
from typesense.exceptions import ObjectNotFound
from typesense.client import Client
import msgspec

logger = logging.getLogger("typesense_service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
)
logger.addHandler(handler)
load_dotenv()


class DocumentBody(msgspec.Struct):
    id: str
    name: str
    image: str
    slug: str
    brand: str
    price: float
    categories: List[str]
    updated_at: str


class TypesenseService:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        protocol: Optional[str] = None,
        api_key: Optional[str] = None,
        connection_timeout_seconds: int = 60,
    ):
        """
        Crea una instancia del cliente Typesense a partir de variables de entorno o parámetros directos.
        """
        self.host = host or os.getenv(
            "TYPESENSE_HOST", "localhost"
        )
        self.port = port or int(
            os.getenv("TYPESENSE_PORT", 8108)
        )
        self.protocol = protocol or os.getenv(
            "TYPESENSE_PROTOCOL", "http"
        )
        self.api_key = api_key or os.getenv(
            "TYPESENSE_API_KEY", "xyz"
        )
        self.timeout = connection_timeout_seconds

        self.client: Client = typesense.client.Client(
            {
                "nodes": [
                    {
                        "host": self.host,
                        "port": self.port,
                        "protocol": self.protocol,
                    }
                ],
                "api_key": self.api_key,
                "connection_timeout_seconds": self.timeout,
            }
        )

        logger.info(
            f"✅ Typesense conectado en {self.protocol}://{self.host}:{self.port}"
        )
        self.ensure_collection(
            "products", self.default_product_schema()
        )

    # -----------------------------------------------------
    # 📦 Esquemas y aseguramiento de colecciones
    # -----------------------------------------------------
    def default_product_schema(self) -> Dict[str, Any]:
        return {
            "name": "products",
            "fields": [
                {"name": "id", "type": "string"},
                {"name": "name", "type": "string"},
                {"name": "slug", "type": "string"},
                {"name": "image", "type": "string"},
                {"name": "brand", "type": "string"},
                {"name": "price", "type": "float"},
                {
                    "name": "categories",
                    "type": "string[]",
                    "facet": True,
                },
                {"name": "updated_at", "type": "string"},
                {
                    "name": "embedding",
                    "type": "float[]",
                    "embed": {
                        "from": [
                            "name",
                            "brand",
                            "categories",
                        ],
                        "model_config": {
                            "model_name": "ts/all-MiniLM-L12-v2"
                        },
                    },
                },
            ],
            "default_sorting_field": "price",
        }

    def ensure_collection(
        self, collection_name: str, schema: Dict[str, Any]
    ):
        """
        Verifica si la colección existe. Si no, la crea.
        """
        try:
            self.client.collections[
                collection_name
            ].retrieve()
            logger.info(
                f"✅ Collection '{collection_name}' ya existe."
            )
        except ObjectNotFound:
            logger.info(
                f"🆕 Creando collection '{collection_name}'..."
            )
            # Usar tipos específicos para create
            self.client.collections.create(schema)  # type: ignore
        except Exception as e:
            logger.error(
                f"Error asegurando collection '{collection_name}': {e}"
            )
            raise

    # -----------------------------------------------------
    # 📤 Operaciones CRUD / Indexación
    # -----------------------------------------------------
    def upsert_document(
        self, collection: str, doc: DocumentBody
    ):
        """Crea o actualiza un documento."""
        # data = msgspec.json.decode(doc)

        return self.client.collections[
            collection
        ].documents.upsert(msgspec.structs.asdict(doc))

    def add_documents(
        self, collection: str, docs: List[DocumentBody]
    ):
        """Bulk import con newline-delimited JSON."""

        # doc_list = [
        #     msgspec.structs.asdict(doc) for doc in docs
        # ]
        doc_list = msgspec.json.encode(docs)
        return self.client.collections[
            collection
        ].documents.import_(doc_list, {"action": "upsert"})

    def get_document(self, collection: str, doc_id: str):
        return (
            self.client.collections[collection]
            .documents[doc_id]
            .retrieve()
        )

    def delete_document(self, collection: str, doc_id: str):
        return (
            self.client.collections[collection]
            .documents[doc_id]
            .delete()
        )

    def clear_collection(self, collection: str):
        return self.client.collections[
            collection
        ].documents.delete({"filter_by": "id: *"})

    # -----------------------------------------------------
    # 🔍 Búsquedas
    # -----------------------------------------------------
    def search(
        self,
        collection: str,
        q: str,
        query_by: str = "name,brand,categories,embedding",
        sort_by: str = "_text_match:desc",
        limit: int = 6,
        offset: int = 0,
    ):
        return self.client.collections[
            collection
        ].documents.search(
            {
                "q": q,
                "query_by": query_by,
                "sort_by": sort_by,
                "limit": limit,
                "offset": offset,
            }
        )
