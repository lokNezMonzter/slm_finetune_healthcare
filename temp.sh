#!/bin/bash
set -e

# Configuration
IMAGE_NAME="slm_healthcare_runtime"
CONTAINER_NAME="slm_workstation"

# Automatically use the directory where this script is located as the host workspace
# This makes the script portable across different VMs
HOST_WS=$(pwd)

echo "Initializing SLM Workspace from: $HOST_WS"

# Build the image only if it doesn't already exist to save time on reboots
if [[ "$(sudo docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
  echo "Building Docker image: $IMAGE_NAME..."
  sudo docker build -t $IMAGE_NAME .
else
  echo "Image $IMAGE_NAME already exists. Skipping build."
fi

# Stop and remove any old instances of the container
sudo docker stop $CONTAINER_NAME 2>/dev/null || true
sudo docker rm $CONTAINER_NAME 2>/dev/null || true

echo "Launching container with 32GB Shared Memory for A6000..."

# Run the container with persistence and heavy shared memory
sudo docker run -d \
  --name $CONTAINER_NAME \
  --gpus all \
  --restart=always \
  --shm-size=32g \
  -v "$HOST_WS":/workspace \
  $IMAGE_NAME \
  sleep infinity

echo "----------------------------------------------------------------"
echo "✅ Container is active and mapped to $HOST_WS"
echo "💻 Access your environment via VSCode Remote-SSH -> Attach to Container"
echo "----------------------------------------------------------------"
