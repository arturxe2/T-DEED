# Use NVIDIA PyTorch base image
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Set the default command - separate script and arguments
CMD ["python", "evaluate_tdeed_challenge.py", "--model", "FineDiving_small"] 