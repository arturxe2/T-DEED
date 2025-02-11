# Use NVIDIA PyTorch base image
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY evaluate_tdeed_challenge.py .
COPY train_tdeed.py .
COPY extract_frames_sn.py .

# Set the default command
CMD ["python", "evaluate_tdeed_challenge.py"] 