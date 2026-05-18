FROM python:3.12-slim

LABEL maintainer="shazad"
LABEL description="Complaints Volume Forecasting Pipeline"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY main.py .

COPY data/ data/

RUN mkdir -p outputs

CMD ["python", "main.py"]
