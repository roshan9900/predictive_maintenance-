# Use Python 3.9 slim as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the image
COPY requirements.txt .


# Copy the application code into the image
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Set the default command to run the application
CMD ["python", "src/model_eval.py"]
