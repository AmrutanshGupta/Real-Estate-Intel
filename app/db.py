from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pymongo.collection import Collection

from app.config import Config
from app.logger import logger

# ── MongoDB Connection Setup ───────────────────────────────────────────────────

try:
    client = MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=5000)
    # Verify connection
    client.admin.command('ping')
    logger.info("MongoDB connection successfully established.")
except ConnectionFailure:
    logger.error("Failed to connect to MongoDB. Is it running on port 27017?")
    client = None

# Database namespace
db = client[Config.PROJECT_NAME.replace(" ", "_")] if client else None

# ── Collections ────────────────────────────────────────────────────────────────
# Using type hints to prevent IDE linting errors
tenants_collection: Collection | None = db["tenants"] if db is not None else None
history_collection: Collection | None = db["query_history"] if db is not None else None

def init_db():
    """
    Called on FastAPI startup to ensure critical database indexes exist.
    """
    if db is None:
        return
        
    try:
        # Ensure org_id is always unique and fast to query
        tenants_collection.create_index("org_id", unique=True)
        # Index history by org_id for fast dashboard retrieval
        history_collection.create_index("org_id")
        
        logger.info("MongoDB collections and indexes verified.")
    except Exception as e:
        logger.error(f"Error initializing MongoDB indexes: {e}")

# ── Database Helper Functions ──────────────────────────────────────────────────

def verify_or_create_tenant(org_id: str, email: str = None) -> bool:
    """
    Checks if a tenant exists. If not, creates one. 
    Used during login and OAuth callbacks.
    """
    if tenants_collection is None:
        return True # Fallback if running without DB locally
        
    tenant = tenants_collection.find_one({"org_id": org_id})
    if not tenant:
        tenants_collection.insert_one({
            "org_id": org_id,
            "created_by": email,
            "status": "active"
        })
    return True