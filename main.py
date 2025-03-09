from flask_cors import CORS
import tempfile
from flask import Flask, request, jsonify
import os
import json
from dotenv import load_dotenv
import requests
import io
from arango import ArangoClient
from langchain_community.graphs import ArangoGraph
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


load_dotenv()

app = Flask(__name__)
#  Add a cors list of domains
# CORS(app, resources={r"/*": {"origins": ["https://voxaudio.nusic.fm", "http://localhost:3000"]}})
CORS(app, resources={r"/*": {"origins": "*"}})
# Configure a temporary directory to store uploaded files


@app.route("/")
def hello_world():
    return jsonify({"text": "Hello world"}), 200

db = ArangoClient(hosts="https://b2eea5c0fdd9.arangodb.cloud:8529").db(username="root", password="8Ykvx9i54OysfIaCz5gk", verify=True)
arango_graph = ArangoGraph(db)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@tool
def tool_extract_emotions(query: str):
    """
    "You are an AI trained to respond to user statements by identifying and expressing the corresponding emotions from 'The Wheel of Emotion' as defined by Robert Plutchik. Do not deviate from the emotions listed in the wheel.

    **Instructions:**

    1.  **Analyze User Input:** Carefully analyze the user's statement to determine the underlying emotions.
    2.  **Refer to Plutchik's Wheel:** Use 'The Wheel of Emotion' as your sole reference for emotion identification.
    3.  **Output Emotion(s):** Respond with the identified emotion(s) directly from the wheel, separated by commas if multiple. Do not provide explanations or additional commentary.

    **Example:**

    **User Input:** 'I just won the lottery!'
    **AI Response:** joy, surprise

    **User Input:** 'My best friend betrayed me.'
    **AI Response:** sadness, disgust

    **User Input:** [User's statement here]"
    """
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    chain = ArangoGraphQAChain.from_llm(
    	llm=llm,
    	graph=arango_graph,
    	verbose=True,
        allow_dangerous_requests=True
    )
    result = chain.invoke(query)
    return str(result["result"])

@tool
def text_to_aql_to_text(query: str):
    """This tool is available to invoke the
    ArangoGraphQAChain object, which enables you to
    translate a Natural Language Query into AQL, execute
    the query, and translate the result back in {emotion: '', audioUrl: ''}.
    """

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

    chain = ArangoGraphQAChain.from_llm(
    	llm=llm,
    	graph=arango_graph,
    	verbose=True,
        allow_dangerous_requests=True
    )

    result = chain.invoke(query)

    return str(result["result"])

def query_graph(query):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    tools = [tool_extract_emotions]
    app = create_react_agent(llm, tools)
    final_state = app.invoke({"messages": [{"role": "user", "content": query}]})
    return final_state["messages"][-1].content

@app.route("/extract-emotions", methods=["POST"])
def extract_emotions():
    query = request.json.get("query")
    content = query_graph(query)
    print(content)
    return jsonify({"content": content}), 200

# @app.route("/extract-emotions", methods=["POST"])
# def extract_emotions():
#     query = request.json.get("query")
#     content = query_graph(query)
#     print(content)
#     return jsonify({"content": content}), 200

# def create_graph():
#     # Graph and Collection Names
#     GRAPH_NAME = "emotionsGraph"
#     VERTEX_COLLECTION = "emotions"
#     EDGE_COLLECTION = "emotion_edges"

#     # Ensure collections exist
#     if not db.has_collection(VERTEX_COLLECTION):
#         db.create_collection(VERTEX_COLLECTION)

#     if not db.has_collection(EDGE_COLLECTION):
#         db.create_collection(EDGE_COLLECTION, edge=True)

#     # Create Graph if not exists
#     if not db.has_graph(GRAPH_NAME):
#         graph = db.create_graph(GRAPH_NAME)
#         graph.create_vertex_collection(VERTEX_COLLECTION)
#         graph.create_edge_definition(
#             edge_collection=EDGE_COLLECTION,
#             from_vertex_collections=[VERTEX_COLLECTION],
#             to_vertex_collections=[VERTEX_COLLECTION]
#         )
#     else:
#         graph = db.graph(GRAPH_NAME)

#     # Example dataset
#     emotion_data = json.load(open("emo.json"))

#     edge_data = [
#         {"_from": "emotions/Happy", "_to": "emotions/Optimistic", "relation": "subcategory_of"},
#         {"_from": "emotions/Optimistic", "_to": "emotions/Inspired", "relation": "subcategory_of"}
#     ]

#     # Insert emotions (vertices)
#     emotion_collection = graph.vertex_collection(VERTEX_COLLECTION)
#     for emotion in emotion_data:
#         if not emotion_collection.has(emotion["_key"]):
#             emotion_collection.insert(emotion)

#     # Insert edges (relationships)
#     edge_collection = graph.edge_collection(EDGE_COLLECTION)
#     for edge in edge_data:
#         edge_collection.insert(edge)

#     print("Graph created successfully!")
@app.route("/qa", methods=["POST"])
def qa():
    query = request.json.get("query")
    # Ask LLM to find the most relevant emotion for the user input
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    # result = llm.invoke("what is the most relevant emotion for the user input: 'I'm in the town, lets roam around'? return some text along with the emotion")

    chain = ArangoGraphQAChain.from_llm(
        llm=llm, graph=arango_graph, verbose=True,
        allow_dangerous_requests=True
    )
    result = chain.invoke(
        f"""Given the user input: '{query}', analyze the emotion and:
        1. First, identify the primary emotion that best matches the input using OR operator
        2. Then, find the corresponding secondary emotion
        3. Finally, return only the tertiary emotion associated with these emotions
        
        If no exact tertiary emotion is found, return the most closely related tertiary emotion.
        Return only the emotion name without any additional explanation. If the emotion is not found, return 'Not found'"""
    )
    return str(result["result"])

if __name__ == "__main__":
    print("Server is running")
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
