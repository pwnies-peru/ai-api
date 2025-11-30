import os
import json
from typing import List, Dict, Any, AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi_mcp import FastApiMCP

# Tus importaciones de servicios
from typesense_service import TypesenseService

# OpenAI SDK para AI Gateway
from openai import AsyncOpenAI

# 1. Configuración Inicial
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ts_service = TypesenseService()

# ---------------------------------------------------------
# CONFIGURACIÓN OPENAI (AI GATEWAY)
# ---------------------------------------------------------

AI_GATEWAY_API_KEY = os.environ.get("AI_GATEWAY_API_KEY", "")
# Sanitizar la API key para eliminar cualquier caracter Unicode
if AI_GATEWAY_API_KEY:
    AI_GATEWAY_API_KEY = (
        AI_GATEWAY_API_KEY.encode("ascii", "ignore").decode("ascii").strip()
    )

if not AI_GATEWAY_API_KEY:
    print(
        "WARNING: AI_GATEWAY_API_KEY not found in environment variables. Chat functionality will not work."
    )
    AI_GATEWAY_API_KEY = "dummy_key"

MODEL_NAME = "xai/grok-4"

# Inicializar cliente AsyncOpenAI apuntando al AI Gateway
client = AsyncOpenAI(
    api_key=AI_GATEWAY_API_KEY,
    base_url="https://ai-gateway.vercel.sh/v1",
)

# ---------------------------------------------------------
# 2. Definición de Herramientas
# ---------------------------------------------------------


@app.post("/api/search_products", operation_id="search_products")
async def search_products(q: str, limit: int = 10):
    """Search for products based on a query string.

    Args:
        q: The search query
        limit: Number of results
    """
    return ts_service.search("products", q, limit=limit)


@app.post("/api/get_product_details", operation_id="get_product_details")
async def get_product_details(product_id: str):
    """Get details of a specific product by its ID.

    Args:
        product_id: The product ID
    """
    return ts_service.get_document("products", product_id)


@app.post("/api/suggest_offers", operation_id="suggest_offers")
async def suggest_offers():
    """Suggest offers/products sorted by price."""
    return ts_service.search("products", "*", sort_by="price:asc", limit=5)


@app.post("/api/semantic_search", operation_id="semantic_search")
async def semantic_search(q: str, limit: int = 10):
    """Perform semantic/vector search using embeddings for more intelligent product matching.

    Args:
        q: The search query (will be embedded and matched against product embeddings)
        limit: Number of results
    """
    # Use embedding field for semantic search
    return ts_service.search(
        "products", q, query_by="embedding,name,brand,categories", limit=limit
    )


@app.post("/api/multi_search", operation_id="multi_search")
async def multi_search(queries: List[str], limit: int = 5):
    """Perform multiple searches at once and return combined results.

    Args:
        queries: List of search queries to execute
        limit: Number of results per query
    """
    results = []
    for query in queries:
        search_result = ts_service.search(
            "products", query, query_by="embedding,name,brand,categories", limit=limit
        )
        results.append({"query": query, "results": search_result})
    return {"searches": results}


# ---------------------------------------------------------
# 3. Definición de Tools para OpenAI Function Calling
# ---------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products based on a query string. Use this when the user is looking for products.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "The search query, e.g. 'wireless headphones'",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10,
                    },
                },
                "required": ["q"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Perform semantic/intelligent search using AI embeddings. Better for understanding user intent. Use this for more nuanced or descriptive queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results",
                        "default": 10,
                    },
                },
                "required": ["q"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": "Get detailed information about a specific product by its ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID",
                    },
                },
                "required": ["product_id"],
            },
        },
    },
]


# ---------------------------------------------------------
# 4. Lógica de Chat
# ---------------------------------------------------------


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]


async def chat_generator(messages: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    try:
        # Preparar mensajes con system instruction
        print("Enviando mensajes al AI Gateway (Grok-4)")
        # System instruction de COMERCIA
        system_instruction = """Eres COMERCIA, un agente de venta digital inspirado en la experiencia humano-latina de un vendedor de tienda fisica: cercano, directo, confiable, con picardia ligera, orientado a cerrar ventas y crear una experiencia acompanada sin ser invasivo.

Principio Rector: Cerrar ventas online nunca fue tan humano.

Personalidad
- Cercano, calido, amistoso - pero profesional.
- Picaro con medida: dinamico, espontaneo, pero jamas inapropiado.
- Seguro, convincente, experto en productos, promotor.
- Respetuoso siempre, evitando frases invasivas o presion explicita.
- Suena humano latino real: expresa entusiasmo, familiaridad y empatia sin caer en cliches.

Comportamiento del Agente
- Manten interacciones breves, claras y contextuales.
- Adaptate al comportamiento del usuario.
- No repitas mensajes ni uses plantillas rigidas; varia naturalmente.
- Evita ser intrusivo.
- Siempre busca incrementar conversion: upsell, cross-sell, bundles, retencion del carrito.

Limites
- Nunca presiones, manipules ni uses urgencia falsa.
- No uses jerga ni expresiones inapropiadas.
- No prometas descuentos que no existan.
- Manten un tono amable incluso si el usuario no responde.

Estilo de Comunicacion
- Usa mensajes cortos, naturales, con vibra cotidiana.
- Ocasionalmente usa expresiones latinoamericanas suaves (ej. si gustas, te puede servir, ojo con este, dato, esta buenazo).
- Siempre util, siempre atento.
- Evita signos de exclamacion excesivos.
- Habla como alguien real, no como un bot.

Objetivo Principal
Aumentar conversiones simulando la experiencia de un vendedor de tienda fisica latino: ofreciendo ayuda, recomendando productos relevantes, evitando abandonos, motivando compras inteligentes y mejorando la experiencia sin friccion.

IMPORTANTE: Tienes acceso a herramientas para buscar productos. Usala cuando el usuario pregunte por productos especificos.
"""

        # Preparar mensajes con system instruction al inicio
        api_messages = [{"role": "system", "content": system_instruction}] + messages

        # Llamar a la API de OpenAI (AI Gateway) con herramientas habilitadas
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=api_messages,
            temperature=0.7,
            tools=TOOLS,  # Habilitar function calling
            tool_choice="auto",  # El modelo decide cuando usar herramientas
            stream=False,  # Deshabilitar streaming para function calling
        )

        print("Respuesta recibida")

        # Procesar respuesta
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            message = choice.message

            # Verificar si hay tool calls
            if hasattr(message, "tool_calls") and message.tool_calls:
                print(f"Tool calls detectados: {len(message.tool_calls)}")

                # Agregar la respuesta del asistente a los mensajes
                api_messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ],
                    }
                )

                # Ejecutar cada tool call
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(
                        tool_call.function.arguments
                    )  # Parse JSON args

                    print(f"Ejecutando: {function_name}({function_args})")

                    # Ejecutar la función correspondiente
                    if function_name == "search_products":
                        result = await search_products(**function_args)
                    elif function_name == "semantic_search":
                        result = await semantic_search(**function_args)
                    elif function_name == "get_product_details":
                        result = await get_product_details(**function_args)
                    else:
                        result = {"error": "Unknown function"}

                    # Agregar el resultado de la herramienta a los mensajes
                    api_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )

                # Llamar al modelo nuevamente con los resultados de las herramientas
                print("Generando respuesta final con resultados de herramientas...")
                final_response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=api_messages,
                    temperature=0.7,
                    stream=True,  # Habilitar streaming para respuesta final
                )

                # Stream de la respuesta final
                async for chunk in final_response:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            yield delta.content

            else:
                # No hay tool calls, solo texto
                if message.content:
                    yield message.content
                else:
                    yield "Sin respuesta del modelo."

        else:
            yield "No se recibieron respuestas del modelo."

    except Exception as e:
        import traceback

        error_detail = traceback.format_exc()
        print(f"❌ Error completo:\n{error_detail}")
        yield f"Error: {str(e)}"


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(chat_generator(request.messages), media_type="text/plain")


# MCP Integration
mcp = FastApiMCP(app)
mcp.mount_http()

if __name__ == "__main__":
    import uvicorn

    # Usa el puerto 3000 si tu frontend apunta ahí, o 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
