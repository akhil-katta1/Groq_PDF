import typer
from typing import Optional, List
from phi.agent import Agent
from phi.model.groq import Groq
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2
#from phi.embedder.mistral import MistralEmbedder
from phi.embedder.huggingface import HuggingfaceCustomEmbedder

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Database configuration
db_url = "postgresql+psycopg2://ai:ai@localhost:5532/ai"

# Knowledge base
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection="recipes", db_url=db_url),
    #embedder=MistralEmbedder(),
    embedder = HuggingfaceCustomEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
)

knowledge_base.load()

# Storage
storage = PgAssistantStorage(
    table_name="agent_runs",  
    db_url=db_url
)


def pdf_agent(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    agent = Agent(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        model=Groq(id="llama3-3-70b-versatile"),  
        description="PDF assistant agent using Llama-3.3-70b-Versatile",
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
    )

    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    agent.cli_app(markdown=True)


if __name__ == "__main__":
    typer.run(pdf_agent)
