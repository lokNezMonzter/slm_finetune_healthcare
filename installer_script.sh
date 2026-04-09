#!/bin/bash

# Detect the Linux distribution
if grep -q "Ubuntu" /etc/os-release; then
  echo "Linux distribution: Ubuntu"
  LINUX_DISTRO="ubuntu"
elif grep -q "Debian" /etc/os-release; then
  echo "Linux distribution: Debian"
  LINUX_DISTRO="debian"

  # check if buster or bullseye
  if grep -q "VERSION_ID=\"11\"" /etc/os-release; then
    echo "Debian version: 11"
    DEBIAN_NAME="bullseye"
  elif grep -q "VERSION_ID=\"10\"" /etc/os-release; then
    echo "Debian version: 10"
    DEBIAN_NAME="buster"
  else
    echo "Unsupported distribution of Debian"
    exit 1
  fi
else
  echo "Unsupported distribution"
  exit 1
fi

# ----------------------------------------------------
# 1. Install NVIDIA drivers
# ----------------------------------------------------
sudo apt -y upgrade
sudo apt-get update -y

sudo apt install -y software-properties-common software-properties-gtk

if [ "$LINUX_DISTRO" = "ubuntu" ]; then
  sudo apt-get -y install locales-all
  sudo apt -y install nvidia-driver-525 nvidia-dkms-525 # works for Ubuntu
elif [ "$LINUX_DISTRO" = "debian" ]; then
  # to avoid user input:
  echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections
  echo 'keyboard-configuration keyboard-configuration/layout select <keyboard_layout>' | sudo debconf-set-selections
  sudo dpkg-reconfigure -f noninteractive keyboard-configuration

  # Works for Debian 11
  echo "Installing drivers for Debian $DEBIAN_NAME"
  sudo apt install -y cron
  sudo apt -y install gcc
  sudo apt-get -y install linux-headers-$(uname -r)

  if [ "$DEBIAN_NAME" = "bullseye" ]; then
    echo "deb http://deb.debian.org/debian/ $DEBIAN_NAME main contrib non-free" | sudo tee -a /etc/apt/sources.list
    sudo apt update -y
    sudo apt install nvidia-driver -y
  elif [ "$DEBIAN_NAME" = "buster" ]; then
    # to get a newer version of the nvidia driver
    # by default install nvidia 4.18 (does not support RTX A4000)
    #   see als: https://download.nvidia.com/XFree86/Linux-x86_64/418.43/README/supportedchips.html
    #   below installs version 4.70
    echo "deb http://deb.debian.org/debian $(lsb_release -cs)-backports main contrib non-free" | sudo tee -a /etc/apt/sources.list.d/backports.list
    sudo apt update -y
    sudo apt -t $(lsb_release -cs)-backports install nvidia-driver -y
  fi
fi

# ----------------------------------------------------
# 2. Install Docker
# ----------------------------------------------------
# Adjusted an parametrized from urls below
# from: https://docs.docker.com/engine/install/debian/
# from: https://docs.docker.com/engine/install/ubuntu/

# Add Docker's official GPG key:
sudo apt-get install ca-certificates curl gnupg -y
sudo install -m 0755 -d /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/"${LINUX_DISTRO}"/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${LINUX_DISTRO} \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" |
  sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update -y
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y

sudo systemctl restart docker

# ----------------------------------------------------
# 3. Install NVIDIA CUDA Toolkit
# ----------------------------------------------------
# from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.1/install-guide.html
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg &&
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update -y
sudo apt-get install -y nvidia-container-toolkit

sudo systemctl restart docker

# To test docker with GPU (ubuntu)
# cuda 11.8 supported for Pytorch: https://pytorch.org/get-started/locally/
#sudo docker run -it --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
