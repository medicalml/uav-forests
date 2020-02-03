#!/bin/bash
echo 'sudo swapoff -a'
sudo swapoff -a
echo "Resize swap"
sudo dd if=/dev/zero of=/swapfile bs=1G count=64
echo "Make the file usable as swap"
sudo mkswap /swapfile
echo "Activate the swap file"
sudo swapon /swapfile
echo "Check the amount of swap available"
grep SwapTotal /proc/meminfo
