# Use official Python 3.12 image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies and Python packages
COPY requirements.txt .

RUN apt-get update && apt-get install -y gcc \
  && pip install --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Copy the rest of the app
COPY . .

# Expose port FastAPI runs on
EXPOSE 8080

# Start the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]