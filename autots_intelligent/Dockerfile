FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r autots_intelligent/requirements.txt

ENV PYTHONPATH=/app

WORKDIR /app/autots_intelligent

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port $PORT"]