#!/bin/bash

echo "Syncing environment..."

output=$(uv sync 2>&1)
if [ $? -ne 0 ]
then
    echo "$output"
    echo "Error: Failed to sync the environment."
    exit 1
fi

