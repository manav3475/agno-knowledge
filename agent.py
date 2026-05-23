from agno.agent import Agent
from agno.models.openai import OpenAIChat
from knowledge_base import Knowledge_base
from dotenv import load_dotenv

load_dotenv()

# create the model
llm = OpenAIChat(id="gpt-4.1-mini")

#create the agent
agent = Agent(
    model=llm,
    knowledge=Knowledge_base,
    name="Knowledge_Agent",
    search_knowledge=True,
    instructions=["You are a helpful assistant ",
                 "whenever asked about transformer model,",
                 "use the knowledge base to get the context",
                 "try not to hallucinate while responding",
                 "if you don't know the answer to a particular query just say I don't know."],
    stream=True,
    markdown=True
)


agent.print_response("Hello How are you. Can you tell me the capital of India")

agent.print_response("Can you tell me What hardware was used for training the transformer model ?")


