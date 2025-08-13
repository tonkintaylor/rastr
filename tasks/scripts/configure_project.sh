#!/bin/bash

output=$(uv pip show pre-commit &> /dev/null)
if [ $? -eq 0 ]
then
    echo Installing pre-commit hooks...
    output=$(uv run pre-commit install 2>&1)
    if [ $? -ne 0 ]
    then
        echo "$output"
        echo "Error: pre-commit install failed."
        exit 1
    fi
fi

if [ -f ".config.template.yaml" ] && [ ! -f ".config.yaml" ]
then
    echo "Creating .config.yaml file from template..."
    output=$(cp .config.template.yaml .config.yaml)
    if [ $? -ne 0 ]
    then
        echo "$output"
        echo "Error: Failed to create .config.yaml from .config.template.yaml file"
        exit 1
    fi
fi

if [ -f ".vscode/settings.template.json" ] && [ ! -f ".vscode/settings.json" ]
then
    echo "Creating VS Code settings.json..."
    output=$(mkdir -p ".vscode")
    if [ $? -ne 0 ]
    then
        echo "$output"
        echo "Error: Failed to create .vscode directory."
        exit 1
    fi

    output=$(cp ".vscode/settings.template.json" ".vscode/settings.json")
    if [ $? -ne 0 ]
    then
        echo "$output"
        echo "Error: Failed to copy settings.template.json to settings.json."
        exit 1
    fi
fi

output=$(find tasks -type f -exec chmod +x {} \; 2>&1)
if [ $? -ne 0 ]
then
    echo "$output"
    echo "Error: Failed to update file permissions for scripts in tasks/ directory."
    exit 1
fi
