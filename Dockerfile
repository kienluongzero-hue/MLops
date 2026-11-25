FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml ./
COPY requirements-api.txt ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements-api.txt && pip install --no-cache-dir --no-deps .

EXPOSE 8000
CMD ["uvicorn", "predictive_maintenance.api:app", "--host", "0.0.0.0", "--port", "8000"]
