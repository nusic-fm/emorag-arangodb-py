FROM python:3.11.11

# Install required system libraries
# RUN apt-get update && apt-get install -y libsndfile1 ffmpeg

WORKDIR /app


# RUN pip install torch
# RUN pip install --upgrade pip
# RUN pip install nx-arangodb
# RUN pip install --upgrade langchain langchain-community langchain-openai langgraph
# RUN pip install --upgrade Flask

RUN pip install --upgrade --quiet  python-arango # The ArangoDB Python Driver
RUN pip install --upgrade --quiet  adb-cloud-connector # The ArangoDB Cloud Instance provisioner
RUN pip install --upgrade --quiet  langchain-openai
RUN pip install --upgrade --quiet  langchain
RUN pip install --upgrade --quiet  langchain-community
RUN pip install --upgrade --quiet  langgraph
RUN pip install --upgrade --quiet  langchain-core

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "main.py"]