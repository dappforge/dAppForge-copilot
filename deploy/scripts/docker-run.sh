#!/bin/bash

# Function to get public IP from ECS service
get_public_ip() {
    local service_name=$1
    local cluster_name=${2:-DAppSolutionCluster}
    
    local task_id=$(aws ecs list-tasks --cluster "$cluster_name" --service-name "$service_name" --desired-status RUNNING --query 'taskArns[*]' --output text)
    local task_arn=$(aws ecs describe-tasks --cluster "$cluster_name" --tasks "$task_id" --query 'tasks[0].taskArn' --output text)
    local task_eni=$(aws ecs describe-tasks --cluster "$cluster_name" --tasks "$task_arn" --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)
    local public_ip=$(aws ec2 describe-network-interfaces --network-interface-ids "${task_eni}" --query 'NetworkInterfaces[0].Association.PublicIp' --output text)
    
    echo "$public_ip"
}

# Build
echo "Building docker container..."
docker build -t nl-dapp-backend .

# Get public IPs for all services
echo "Getting public IPs for all services..."
ECS_CLUSTER=DAppSolutionCluster
MEMGRAPH_SOLIDITY_IP=$(get_public_ip "DAppMemgraphSolidityService" "$ECS_CLUSTER")
MEMGRAPH_INK_IP=$(get_public_ip "DAppMemgraphInkService" "$ECS_CLUSTER")
MEMGRAPH_RUST_IP=$(get_public_ip "DAppMemgraphRustService" "$ECS_CLUSTER")
MEMGRAPH_SUBSTRATE_IP=$(get_public_ip "DAppMemgraphSubstrateService" "$ECS_CLUSTER")

# Run
echo "Running docker container..."
docker run --rm -it -p 8081:8081 \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e AWS_REGION=us-east-1 \
    -e MEMGRAPH_SOLIDITY_HOST=$MEMGRAPH_SOLIDITY_IP \
    -e MEMGRAPH_INK_HOST=$MEMGRAPH_INK_IP \
    -e MEMGRAPH_RUST_HOST=$MEMGRAPH_RUST_IP \
    -e MEMGRAPH_SUBSTRATE_HOST=$MEMGRAPH_SUBSTRATE_IP \
    -e MEMGRAPH_PORT=7687 \
    -e MEMGRAPH_USER=$MEMGRAPH_USER \
    -e MEMGRAPH_PASSWORD=$MEMGRAPH_PASSWORD \
    nl-dapp-backend
