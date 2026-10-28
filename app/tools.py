from langchain_tavily import TavilySearch

web_search = TavilySearch(
    max_results=3,
    topic="general",
    include_images=True,               # return images related to the query
    include_image_descriptions=True,   # plus a short description of each image
)