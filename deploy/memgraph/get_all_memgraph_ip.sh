#!/bin/bash

set -e

ECS_CLUSTER=DAppSolutionCluster

get_ip () {
    ECS_SERVICE=$1
    # echo "$ECS_SERVICE"
    TASK_ID=$(aws ecs list-tasks --cluster "$ECS_CLUSTER" --service-name "$ECS_SERVICE" --desired-status RUNNING --query 'taskArns[*]' --output text)
    # echo "Task ID: $TASK_ID"
    TASK_ARN=$(aws ecs describe-tasks --cluster "$ECS_CLUSTER" --tasks "$TASK_ID" --query 'tasks[0].taskArn' --output text)
    # echo "Task ARN: $TASK_ARN"
    TASK_ENI=$(aws ecs describe-tasks --cluster "$ECS_CLUSTER" --tasks "$TASK_ARN" --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)
    # echo "Task ENI: $TASK_ENI"
    PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids "${TASK_ENI}" --query 'NetworkInterfaces[0].Association.PublicIp' --output text)
    # echo "Public IP: $PUBLIC_IP"
    # echo "-------"
    echo "$ECS_SERVICE: $PUBLIC_IP"
}

echo "====="
get_ip DAppMemgraphSolidityService
get_ip DAppMemgraphInkService
get_ip DAppMemgraphRustService
get_ip DAppMemgraphSubstrateService
echo "====="
