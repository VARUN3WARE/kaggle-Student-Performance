# Makefile for common developer tasks
.PHONY: mlflow-up mlflow-down build-trainer train clean

MLFLOW_COMPOSE=docker-compose.mlflow-local.yml

mlflow-up:
	docker compose -f $(MLFLOW_COMPOSE) up -d --build

mlflow-down:
	docker compose -f $(MLFLOW_COMPOSE) down

build-trainer:
	docker compose -f $(MLFLOW_COMPOSE) build --no-cache --progress=plain --build-arg USER_ID=$(shell id -u) --build-arg GROUP_ID=$(shell id -g) trainer

train:
	# Run a training session and log to MLflow on localhost
	docker run --rm -it --network host \
	  -v "$(PWD)":/workspace:cached \
	  -v "$(PWD)/mlflow":/mlflow \
	  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
	  -w /workspace \
	  kaggle-student-performance-trainer \
	  python src/model_training.py --data data/processed/processed_student_data.csv --out-dir models --mlflow

clean:
	rm -rf models/*.pkl models/*.json models/training_summary.csv models/model_comparison.html
