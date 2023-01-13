FROM python:3.7
EXPOSE 443
WORKDIR /app
COPY main.py .
COPY pulldata.py .
COPY getmodel.py .
COPY requirements.txt .
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=443", "--server.address=0.0.0.0"]
