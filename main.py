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
    tools = [text_to_aql_to_text]
    app = create_react_agent(llm, tools)
    final_state = app.invoke({"messages": [{"role": "user", "content": query}]})
    return final_state["messages"][-1].content

@app.route("/extract-emotions", methods=["POST"])
def extract_emotions():
    query = request.json.get("query")
    content = query_graph(query)
    print(content)
    return jsonify({"content": content}), 200

if __name__ == "__main__":
    print("Server is running")
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
