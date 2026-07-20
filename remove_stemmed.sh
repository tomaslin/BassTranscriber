#!/bin/bash

for file in *.mp3; do
    # Remove the .mp3 extension to get the base track name
    base_name="${file%.mp3}"
    stem_dir="stems_${base_name}"
    
    # Check if the directory exists before targeting it
    if [ -d "$stem_dir" ]; then
        echo "Deleting: $stem_dir"
        rm -rf "$stem_dir"
    fi
done
