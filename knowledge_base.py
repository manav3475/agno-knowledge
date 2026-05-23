from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.chunking.semantic import SemanticChunking
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.vectordb.lancedb import LanceDb
from agno.knowledge.knowledge import Knowledge
from dotenv import load_dotenv

load_dotenv()

#define the embedder
embedder = OpenAIEmbedder()

#define the chinking strategy
chunking_strategy = SemanticChunking(embedder=embedder,
                                     chunk_size=1000)

# create the pdf reader
reader = PDFReader(chunking_strategy=chunking_strategy)


# create the vector db 
vector_db = LanceDb(uri="vector_db/lancedb",
                    embedder=embedder,
                    table_name="knowledge_table")


#create a knowledge base 
Knowledge_base = Knowledge(name="knowledge_base",
                           description="Contains the research paper 'Attention is all you need'",
                           vector_db=vector_db)


if __name__ == "__main__":

    # add content to the knowledge base
    Knowledge_base.add_content(path="Data/Attention_is_all_you_need.pdf",
                           reader=reader)