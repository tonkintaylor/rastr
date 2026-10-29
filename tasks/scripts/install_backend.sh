#!/bin/bash

echo "Ensuring project directory has a Git repository initialized..."
# Git is a pre-condition for using pyproject.toml with VCS-based dynamic versioning
if [ ! -d ".git" ]; then
    echo "Error: No Git repository found!"
    exit 1
fi

# Ideally remove this duplication via https://github.com/astral-sh/uv/issues/10547
uv_min_version="0.13.6" # Sync with pyproject.toml

export PATH="$PATH:$HOME/.local/bin"
# Legacy install support < v0.5.0
export PATH="$PATH:$HOME/.cargo/bin"

echo "Ensuring uv is installed at v$uv_min_version or later..."

# Check if the uv command exists and is at least the minimum version, else try to update
need_to_install=true
if command -v uv &> /dev/null
then
    installed_version=$(uv --version | cut -d' ' -f2)
    if [ "$(printf '%s\n%s\n' "$uv_min_version" "$installed_version" | sort -V | head -n1)" = "$uv_min_version" ]
    then
        need_to_install=false
    elif uv self update > /dev/null 2>&1
    then
        need_to_install=false
    fi
fi

if [ "$need_to_install" = true ]
then
    if [ -n "$WINDIR" ]
    then
        # If in Windows
        if command -v pwsh &> /dev/null
        then
            output=$(pwsh -Command "irm https://astral.sh/uv/install.ps1 | iex" 2>&1)
        elif command -v powershell &> /dev/null
        then
            output=$(powershell -Command "irm https://astral.sh/uv/install.ps1 | iex" 2>&1)
        else
            output=$(C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -Command "irm https://astral.sh/uv/install.ps1 | iex" 2>&1)
        fi
    else
        # Otherwise, assume in Linux, and we might be in CI so use backoff
        for i in 1 2 3; do
            curl -LsSf https://astral.sh/uv/install.sh -o install-uv.sh && break 2>&1
            sleep $((2**i))
        done
        chmod +x install-uv.sh
        output=$(./install-uv.sh 2>&1)
        rm -f install-uv.sh
    fi

    if [ $? -ne 0 ]
    then
        echo "$output"
        echo "Error: Failed to install uv!"
        exit 1
    fi
fi
