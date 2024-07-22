import getpass
import os

os.environ["TAVILY_API_KEY"] = getpass.getpass()

from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults()

tool.invoke({"query": "What happened in the latest burning man floods"})

