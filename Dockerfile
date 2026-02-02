# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Copy the rest of the application code
COPY . .

# Install dependencies with increased timeout and retries
# We use pip to install the package in editable mode so imports work correctly
RUN pip install --no-cache-dir --timeout=300 --retries=5 -e .

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Expose the port Streamlit runs on
EXPOSE 8501

# Use entrypoint script to initialize DB before starting app
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
