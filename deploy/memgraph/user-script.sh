#!/bin/bash

# Update apt
sudo apt-get update

# Install EFS utils
sudo apt-get install -y nfs-common

# Create data directory
sudo mkdir -p /opt/memgraph/data

# Mount EFS
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport EFS_FS_ID.efs.us-east-1.amazonaws.com:/ /opt/memgraph/data

# Adopt ownership and permissions
sudo chown 101:103 /opt/memgraph/data  # memgraph user and group
sudo chmod 755 /opt/memgraph/data

# Install Docker
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Enable docker autostart
sudo systemctl enable docker

# Create docker compose file
cd /opt/memgraph
sudo tee docker-compose.yml > /dev/null <<EOF
services:
  memgraph:
    image: memgraph/memgraph-mage:3.4
    container_name: memgraph
    restart: unless-stopped
    environment:
      - MEMGRAPH="--log-level=TRACE --storage-properties-on-edges=true --storage-snapshot-interval-sec=60 --storage-snapshot-retention-count=3 --storage-mode=IN_MEMORY_TRANSACTIONAL --storage-snapshot-on-exit=true"
    ports:
      - '7444:7444'
      - '7687:7687'
      - '3000:3000'
    volumes:
      - '/opt/memgraph/data:/var/lib/memgraph'
EOF

# Start services
sudo docker compose up -d
