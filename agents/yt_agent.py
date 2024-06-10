import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task
from IPython.display import display, Markdown
from langchain_openai import ChatOpenAI
from googleapiclient.discovery import build
from langchain.tools import tool


class YTAgent:

  def __init__(self, vector_store=None):
    # Load environment variables from .env file
    load_dotenv()

    # Set the API model name from environment variable
    os.environ["OPENAI_API_KEY"] = "NA"

    # this is CrewAI special set up
    llm = ChatOpenAI(
        model = "llama3",
        base_url = "http://localhost:11434/v1")

    #Search query agent
    self.query_creator_agent = Agent(
        role="Query creator",
        goal="Based on given user input {query} create a youtube search query",
        backstory=("You are a helper who can turn long text into a relevant query for search"
                   "Your task is to review user long input an provide a short search query "
                   "You have access to tools"),
        verbose=True,
        max_iter=3,
        allow_delegation=True,
        llm=llm
    )

    # Initialize the peer task
    self.query_creator_task = Task(
        description="Create a simple search query from potentially long text based on user input {query}",
        expected_output="Concise search query to search youtube videos",
        agent=self.query_creator_agent,
    )


    #Download YT videos
    self.yt_loader_agent = Agent(
        role="Youtube searcher",
        goal="To get list of youtube videos using tool and query from previous task",
        backstory=("You are a helper who can use tools for youtube video searching"
                   "You have access to tools"),
        tools=[GetYoutube().get_videos],
        verbose=True,
        max_iter=3,
        allow_delegation=True,
        llm=llm
    )



    # Initialize the YT loader
    self.yt_loader_task = Task(
        description="Search for video using tool",
        expected_output="List of video descriptions and urls",
        agent=self.yt_loader_agent,
    )


    #Download YT videos
    self.yt_summary_agent = Agent(
        role="Summary writer",
        goal="To write a nice summary fro youtube videos found",
        backstory=("You are a helper who can nicely summarize the video data into human frindly format"),
        verbose=True,
        max_iter=4,
        allow_delegation=True,
        llm=llm
    )



    # Initialize the YT loader
    self.yt_summary_task = Task(
        description="Write a nice summary from received yotube videos information. Respond to the human as helpfully and accurately as possible.",
        expected_output="List of videos in the format "
                        "1.Description"
                        "2.Url to video"
                        "3.Short description"
        ,
        agent=self.yt_summary_agent,
    )


    # Create the crew with the agent and task
    self.crew = Crew(
        agents=[self.query_creator_agent, self.yt_loader_agent, self.yt_summary_agent],
        tasks=[self.query_creator_task, self.yt_loader_task, self.yt_summary_task],
        verbose=True,
        process=Process.sequential
    )

  def get_crew(self):
    return self.crew

  def display_result(self, result):
    result_formatted = Markdown(result)
    display(result_formatted)
    return result_formatted

class GetYoutube:
  @tool("Get Youtube videos ")
  def get_videos(query: str) -> []:
    """Uses google api to get information about youtube videos based on created query"""
    youtube = build('youtube', 'v3', developerKey=os.getenv("GOOGLE_API_KEY"))

    # Call the search.list method to retrieve results matching the specified query term
    search_response = youtube.search().list(
        q=query,
        part='id,snippet',
        maxResults=5
    ).execute()

    videos = []

    for search_result in search_response.get('items', []):
      if search_result['id']['kind'] == 'youtube#video':
        video_id = search_result['id']['videoId']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
      videos.append({
        'title': search_result['snippet']['title'],
        'videoId': video_id,
        'url': video_url,
        'description': search_result['snippet']['description'],
        'channelTitle': search_result['snippet']['channelTitle']
      })
    return videos


# class YoutubeSearchTool(BaseTool):
#   name: str = "Searcher for youtube videos"
#   description: str = "Uses google api to get information about youtube videos based on created query"
#
#   def _run(query: str) -> str:
#     youtube = build('youtube', 'v3', developerKey=os.getenv("GOOGLE_API_KEY"))
#
#     # Call the search.list method to retrieve results matching the specified query term
#     search_response = youtube.search().list(
#         q=query,
#         part='id,snippet',
#         maxResults=5
#     ).execute()
#
#     videos = []
#
#     for search_result in search_response.get('items', []):
#       if search_result['id']['kind'] == 'youtube#video':
#         video_id = search_result['id']['videoId']
#         video_url = f"https://www.youtube.com/watch?v={video_id}"
#       videos.append({
#         'title': search_result['snippet']['title'],
#         'videoId': video_id,
#         'url': video_url,
#         'description': search_result['snippet']['description'],
#         'channelTitle': search_result['snippet']['channelTitle']
#       })
#
#     return videos


