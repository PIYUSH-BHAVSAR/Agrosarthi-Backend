# Use a slim Python base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port for Cloud Run
EXPOSE 8080



# Run using Uvicorn (simple and direct)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

