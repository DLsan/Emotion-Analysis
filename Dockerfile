# -----------------------------
# Base image with Python 3.10
# -----------------------------
  FROM python:3.10-slim

  # Avoid Python buffering & set workdir
  ENV PYTHONUNBUFFERED=1
  WORKDIR /app
  
  # -----------------------------
  # Install system dependencies for OpenCV & TensorFlow
  # -----------------------------
  RUN apt-get update && apt-get install -y \
      build-essential \
      libglib2.0-0 \
      libsm6 \
      libxrender1 \
      libxext6 \
      && rm -rf /var/lib/apt/lists/*
  
  # -----------------------------
  # Install Python packages
  # -----------------------------
  COPY requirements.txt .
  RUN pip install --upgrade pip && pip install -r requirements.txt
  
  # -----------------------------
  # Copy app + model files
  # -----------------------------
  COPY . .
  
  # -----------------------------
  # Expose port
  # -----------------------------
  EXPOSE 8501
  
  # -----------------------------
  # Run Streamlit
  # -----------------------------
  CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
  