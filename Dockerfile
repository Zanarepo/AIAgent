FROM python:3.9.20-slim

# Install build dependencies and update OS to reduce vulnerabilities
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install setuptools>=65.5.0 && pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "api:app"]