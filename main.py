from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
from typesense_service import TypesenseService
from typing import List, Optional, Dict, Any

app = FastAPI()

# Initialize Typesense Service
ts_service = TypesenseService()

@app.get("/search_products",operation_id="search_products", name="Search Products", description="Search for products based on a query string.")
async def search_products(q: str, limit: int = 10) -> Dict[str, Any]:
    return ts_service.search("products", q, limit=limit)

@app.get("/get_product_details",operation_id="get_product_details", name="Get Product Details", description="Get details of a specific product by its ID.")
async def get_product_details(product_id: str) -> Dict[str, Any]:
    return ts_service.get_document("products", product_id)

@app.get("/get_related_products",operation_id="get_related_products", name="Get Related Products", description="Get related products for a given product ID.")
async def get_related_products(product_id: str) -> Dict[str, Any]:
    try:
        product = ts_service.get_document("products", product_id)
        query = f"{product.get('brand', '')} {product.get('name', '')}"
        return ts_service.search("products", query, limit=5)
    except Exception as e:
        return {"error": f"Could not find related products: {str(e)}"}

@app.get("/suggest_offers",operation_id="suggest_offers", name="Suggest Offers", description="Suggest offers. Currently returns products sorted by price ascending.")
async def suggest_offers() -> Dict[str, Any]:
    return ts_service.search("products", "*", sort_by="price:asc", limit=5)

mcp = FastApiMCP(app)
# Mount MCP endpoints
mcp.mount_http()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

