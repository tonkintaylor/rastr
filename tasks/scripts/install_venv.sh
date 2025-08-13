#!/bin/bash

if command -v deactivate
then
    deactivate
fi

source ./tasks/shims/recover_corrupt_venv

# Finding Python version
output=$(version=$(cat .python-version))
if [ $? -ne 0 ]
then
    echo "$output"
    echo "Error: No .python-version file found. Please create one and try again."
    exit 1
fi

echo "Creating virtual environment..."
if [ -d .venv ]
then
    output=$(rm -rf .venv 2>&1)
    if [ $? -ne 0 ]
    then
        echo "$output"
        echo "Error: Existing virtual environment directory could not be deleted."
        exit 1
    fi
fi

output=$(uv venv 2>&1)
if [ $? -ne 0 ]
then
    echo "$output"
    echo "Error: Virtual environment could not be created."
    exit 1
fi
