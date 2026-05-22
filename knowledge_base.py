from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.chunking.semantic import SemanticChunking
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.vectordb.lancedb import LanceDb
from dotenv import load_dotenv

load_dotenv()

#define the embedder
embedder = OpenAIEmbedder()

#define the chinking strategy
chunking_strategy = SemanticChunking(embedder=embedder,
                                     chunk_size=1000)

# create the pdf reader
reader = PDFReader(chunking_strategy=chunking_strategy)


