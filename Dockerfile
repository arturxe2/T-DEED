# Use NVIDIA PyTorch base image
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt


COPY checkpoints checkpoints
COPY soccernet soccernet
COPY config config
COPY data data
COPY dataset dataset
COPY model model
COPY util util
COPY .env .env
COPY evaluate_tdeed_challenge.py .
COPY train_tdeed.py .
COPY run_inference.py .

# Set the default command - separate script and arguments
CMD ["python", "run_inference.py", "--model", "SoccerNet_small"] 