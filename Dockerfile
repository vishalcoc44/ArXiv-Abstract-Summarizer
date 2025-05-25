# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to reduce image size
# We also ensure pip is up to date.
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# This will respect the .dockerignore file
COPY . .

# Make port 5001 available to the world outside this container
# This should match FLASK_PORT in your config.py
EXPOSE 5001

# Define environment variable for Flask and Gunicorn
ENV FLASK_APP=app.py 
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5001
# Gunicorn needs to know where the WSGI app is. 
# Assuming your Flask app object is named `app` in your `app.py` file.
CMD ["gunicorn", "--workers", "4", "--timeout", "300", "--bind", "0.0.0.0:5001", "app:app"] 