#!/bin/bash
# Kills the build script in case of failure
set -e

# Ensure Docker is running
sudo systemctl start docker

# Build the image (standard procedure)
sudo docker build -t jupyter_image .

# Stop and remove any old version of the container to prevent conflicts
sudo docker stop jupyter_lab_env || true
sudo docker rm jupyter_lab_env || true

# Run the new container with persistence
sudo docker run -d \
  --name jupyter_lab_env \
  --gpus all \
  --restart=always \
  -p 8888:8888 \
  -v /home/ubuntu/workspace/slm_finetune_healthcare:/workspace \
  -v /home/ubuntu/datasets:/data:ro \
  jupyter_image
