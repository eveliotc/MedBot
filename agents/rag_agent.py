import os
from dotenv import load_dotenv
from crewai import Agent, Crew
from crewai import Task
from IPython.display import display, Markdown
from langchain_openai import ChatOpenAI


from crewai_tools import BaseTool
class RagAgent:

  def __init__(self, vector_store=None):
    # Load environment variables from .env file
    load_dotenv()

    # Set the API model name from environment variable
    os.environ["OPENAI_API_KEY"] = "NA"
    self.index = vector_store;

    #Create a list of fake documents. Replace with real from INDEX
    self.fake_docs = [
      "This is a fake document content for testing purposes. as if similar from index",
      "Another fake document with different content."
    ]
    # this is CrewAI special set up
    llm = ChatOpenAI(
        model = "llama3",
        base_url = "http://localhost:11434/v1")

    #Initialize RAG agent
    self.doctor_room_agent = Agent(
        role="Doctor",
        goal="Find the best documets that describe the patient symptoms",
        backstory=("You are a medical doctor in a clinic. "
                   "Your task is to review patient symptoms and provide a preliminary diagnosis. "
                   "You have access to medical documents and literature to assist in your diagnosis."),
        tools=[MyCustomTool()],
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Initialize the peer task
    self.doctor_room_task = Task(
        description="Search for 10 most relevant documents in this my MyCustomTool using this {query}"
                      "Search relevant medical documents, and provide a preliminary diagnosis",

        expected_output="1. Review the patient's basic information and symptoms."
                         "2. Search for relevant medical documents that match the symptoms."
                         "3. Provide a preliminary diagnosis based on the findings."
                         "4. Output format: Patient Info: Symptoms: Preliminary Diagnosis: Relevant Documents:",
        agent=self.doctor_room_agent,
    )

    # Create the crew with the agent and task
    self.crew = Crew(
        agents=[self.doctor_room_agent],
        tasks=[self.doctor_room_task],
        verbose=True,
    )

  def get_crew(self):
    return self.crew

  def display_result(self, result):
    result_formatted = Markdown(result)
    display(result_formatted)
    return result_formatted


class MyCustomTool(BaseTool):
  name: str = "Similar medical documents RAG Searcher"
  description: str = "Uses RAG to retrieve information from the medical documents using this user input {query} as query"

  def _run(self, query: str) -> str:
    # Your RAG logic here
    # plug in your similar faiss doc search here
    # SHOULD LOOK LIKE THIS:
    #docs = self.index.similarity_search(query=query, k=5)
    #return " ".join([doc.page_content for doc in docs])
    print(f"This will be your similar doc search query based on user input => {query}")
    return "High pressure is because you eat unhealthy and work a lot"


