from agno.knowledge.reader import Reader
from agno.knowledge.chunking.strategy import ChunkingStrategy, FixedSizeChunking
from agno.knowledge.embedder.base import Embedder
from dotenv import load_dotenv

load_dotenv()

# define the chunking strategy
chunking_strategy = FixedSizeChunking(chunk_size=1000)

# create the reader
reader = Reader(chunking_strategy=chunking_strategy)



