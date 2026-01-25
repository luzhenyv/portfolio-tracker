# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORTFOLIO_DB_PATH=/app/db/portfolio.db

# Set work directory
WORKDIR /app

# Copy the rest of the application code
COPY . .

# Install dependencies
# We use pip to install the package in editable mode so imports work correctly
RUN pip install --no-cache-dir -e .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
# We use streamlit run directly to ensure proper host binding in container
ENTRYPOINT ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
