#!/bin/bash

# Find all .md files in the docs directory and its subdirectories
find . -name "*.md" -not -name "README.md" | while read file; do
    # Get the new filename by replacing .md with .mdx
    new_file="${file%.md}.mdx"

    # Move the file
    mv "$file" "$new_file"

    echo "Converted: $file → $new_file"
done

echo "Conversion complete!"