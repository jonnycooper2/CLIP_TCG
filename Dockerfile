FROM python:3.10-slim

# Set working directory
WORKDIR /app

# System dependencies for OpenCV and potentially others
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Add libopenblas-dev for numpy/scipy if needed, might help FAISS performance too
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Copy requirements first to leverage Docker cache
COPY requirements.txt .
# Using --no-cache-dir can make the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
# Ensure the data directory is copied correctly
COPY data ./data
COPY serve.py .

# Expose the port the app runs on
EXPOSE 8080

# Set environment variable to ensure CPU usage for PyTorch/Torchvision
ENV CUDA_VISIBLE_DEVICES="-1"

# Run FastAPI app using uvicorn
# Using --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"] 