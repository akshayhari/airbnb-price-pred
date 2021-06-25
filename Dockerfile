FROM python:3.7-slim
EXPOSE 8080
COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt
COPY . /app
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
