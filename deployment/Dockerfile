
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (important for sklearn)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Hugging Face Spaces use port 7860
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
