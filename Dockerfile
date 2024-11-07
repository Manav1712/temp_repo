# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV OPENBLAS_NUM_THREADS 1
ENV OMP_NUM_THREADS 1
ENV MKL_NUM_THREADS 1

# Set the working directory in the container
WORKDIR /app

# Copy all the local files to the container
COPY . /app

# Install any necessary Python packages
# (Assuming you have a requirements.txt file)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

# Command to run the main script by default
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
CMD ["python", "test.py"]