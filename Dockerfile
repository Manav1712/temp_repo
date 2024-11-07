# Use an official Python runtime as a parent image
FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV OPENBLAS_NUM_THREADS 1
ENV OMP_NUM_THREADS 1
ENV MKL_NUM_THREADS 1

# Set the working directory in the container
WORKDIR /code

# Copy all the local files to the container
COPY . /code

# Install any necessary Python packages
# (Assuming you have a requirements.txt file)
# COPY ./requirements.txt /code
RUN pip install --upgrade -r /code/requirements.txt

# Command to run the main script by default
CMD ["python", "test.py"]