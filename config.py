from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# AWS and OpenSearch Configuration
INDEX_NAME = os.getenv('INDEX_NAME')

# Model Configuration
EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID')
LLM_MODEL_ID = os.getenv('LLM_MODEL_ID')
VECTOR_DIMENSION = os.getenv('VECTOR_DIMENSION')

# Validate required environment variables
required_vars = [
    'INDEX_NAME',
    'EMBEDDING_MODEL_ID',
    'LLM_MODEL_ID',
    'VECTOR_DIMENSION'
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}") 