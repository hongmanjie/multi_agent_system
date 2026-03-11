#!/bin/bash

# Startup script for the catalog service
# This script starts both the callback service and the catalog server

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the catalog service directory
cd "$SCRIPT_DIR"

echo "Starting catalog service..."

# Start the callback service for pushing catalog results
echo "Starting task callback service..."

nohup python catalog_server_v3.py > logs/catalog_server_v3.log 2>&1 &
nohup python task_callback_v3.py > logs/task_callback_v3.log 2>&1 &

# Sleep briefly to allow the second service to start
sleep 2

# Show the running processes
echo "Services started. Current Python processes:"
ps aux | grep python | grep -v grep

echo "Startup complete."
echo "Callback service logs are in task_callback.log"
echo "Catalog server logs are in nohup.out"