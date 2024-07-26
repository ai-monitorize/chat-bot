FROM python:3.12-slim-bullseye
COPY app app
COPY requirements.txt app/requirements.txt
WORKDIR app
RUN apt-get update  \
    && apt-get install -y gcc libpq-dev  \
    && rm -rf /var/lib/apt/lists/*  \
    && pip install --upgrade pip  \
    && pip install -r requirements.txt
ENTRYPOINT ["python", "main.py"]