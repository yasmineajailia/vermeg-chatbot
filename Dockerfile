FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories first
RUN mkdir -p /app/static

# Copy static directory with logo
COPY static/ /app/static/

# Copy Python files
COPY *.py .

EXPOSE 8000

CMD ["uvicorn", "vermeg_api:app", "--host", "0.0.0.0", "--port", "8000"]
