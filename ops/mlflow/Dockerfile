FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlflow boto3 sqlalchemy psycopg2-binary

EXPOSE 5000

WORKDIR /app

# Ensure the runtime artifact directory exists and is writable.
# We create /mlflow and the artifacts subdirectory and relax permissions for
# development use so the MLflow server can write artifacts into the mounted
# host directory without permission errors.
RUN mkdir -p /mlflow/artifacts \
    && chmod -R 0777 /mlflow

CMD ["sh", "-c", "mlflow server --backend-store-uri $BACKEND_STORE_URI --default-artifact-root $ARTIFACT_ROOT --host 0.0.0.0 --port 5000"]
