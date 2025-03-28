# import basics
import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain import hub

from supabase.client import Client, create_client
from langchain_core.tools import tool

# load environment variables
load_dotenv()  

# initiate supabase database
# supabase_url = os.environ.get("SUPABASE_URL")
# supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = create_client("https://orlvepdterdawwvivhqi.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9ybHZlcGR0ZXJkYXd3dml2aHFpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MzEzMzg4NiwiZXhwIjoyMDU4NzA5ODg2fQ.Lllq1ohg8m9LXayzscEJKoJ0Irjqu11Hi961tzznB3o")

# initiate embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# initiate vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# initiate large language model (temperature = 0)
llm = ChatOpenAI(temperature=0.6, model="gpt-4o")

# fetch the prompt from the prompt hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# create the tools
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# combine the tools and provide to the llm
tools = [retrieve]
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# invoke the agent
response = agent_executor.invoke({"input": "What is the most common reason startup faill? Give me a report for 500."})

# put the result on the screen
print(response["output"])