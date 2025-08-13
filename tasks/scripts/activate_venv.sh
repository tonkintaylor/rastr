#!/bin/bash

echo "Activating virtual environment..."
if [ -f ".venv/Scripts/activate" ]
then
    activate=".venv/Scripts/activate"
elif [ -f ".venv/bin/activate" ]
then
    activate=".venv/bin/activate"
else
    echo "Error: Could not find the activate script for the virtual environment."
    exit 1
fi

if ! source $activate
then
    echo "Error: Virtual environment could not be activated."
    exit 1
fi

# If in Windows
if [ -n "$WINDIR" ]
then
    # When we activate a virtual environment, it checks _OLD_VIRTUAL_PATH for value of
    # PATH at the time of the most recent activation, to revert that previous activation
    # ready for the current activation. But if the previous activation was at the level
    # of PowerShell then _OLD_VIRTUAL_PATH will be missing all our Git Bash executables.
    # So we need to add them back in manually.
    # Normally, when we add environment variables within a venv, we don't want these to
    # persist outside of the venv. But in the case of shell tools, we do want that.
    export PATH="/mingw64/bin:/usr/local/bin:/usr/bin:/bin:/mingw64/bin:/usr/bin:$PATH"

    # Same thing for any uv installation; it should persist.
    export PATH="$PATH:$HOME/.local/bin"
    # Legacy install support < v0.5.0
    export PATH="$PATH:$HOME/.cargo/bin"
fi
