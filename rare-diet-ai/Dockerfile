FROM python:3.10-slim

WORKDIR /code

RUN apt-get update && apt-get install -y git

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=7860
EXPOSE 7860

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
