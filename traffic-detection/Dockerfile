FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get -y install cmake protobuf-compiler

COPY . .

RUN pip3 install numpy
RUN pip3 install -r requirements.txt

EXPOSE 8080

ENTRYPOINT [ "streamlit", "run", "streamlit_app.py" "--server.port=8080", "--server.address=0.0.0.0"]