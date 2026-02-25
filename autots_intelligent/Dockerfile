# Build from AutoTS directory: docker build -f autots_intelligent/Dockerfile .
# Run: docker run -p 8000:8000 <image>
FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r autots_intelligent/requirements.txt

# Use local autots package when not installed from PyPI
ENV PYTHONPATH=/app

WORKDIR /app/autots_intelligent

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
