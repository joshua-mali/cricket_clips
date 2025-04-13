# Use a suitable Python base image
FROM python:3.10-slim

# Install ffmpeg (required by moviepy)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py .
COPY video_utils.py .

# Expose the Streamlit port
EXPOSE 8501

# Set the CMD to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"] 