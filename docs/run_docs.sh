#!/bin/bash

# Check if Mintlify CLI is installed
if ! command -v mintlify &> /dev/null
then
    echo "Mintlify CLI not found. Installing..."
    npm i -g mintlify
fi

# Run Mintlify dev server
echo "Starting Mintlify documentation server..."
mintlify dev
