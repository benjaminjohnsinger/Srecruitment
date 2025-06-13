#!/bin/bash

# Script to convert Python array assignments from x[idx] = y to x = x.at[idx].set(y)
# Usage: ./convert_arrays.sh input_file.py [output_file.py]

if [ $# -eq 0 ]; then
    echo "Usage: $0 input_file.py [output_file.py]"
    echo "If output_file is not specified, changes will be made in-place"
    exit 1
fi

input_file="$1"
output_file="$2"

# Check if input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found"
    exit 1
fi

# Regular expression explanation:
# ^([[:space:]]*)        - Capture leading whitespace (indentation)
# ([a-zA-Z_][a-zA-Z0-9_]*) - Capture variable name (x)
# \[                     - Literal opening bracket
# ([^]]+)                - Capture index expression (idx) - anything except closing bracket
# \]                     - Literal closing bracket
# [[:space:]]*=[[:space:]]* - Match equals sign with optional spaces
# (.+)                   - Capture the value being assigned (y)
# $                      - End of line

if [ -n "$output_file" ]; then
    # Write to output file
    sed -E 's/^([[:space:]]*)([a-zA-Z_][a-zA-Z0-9_]*)\[([^]]+)\][[:space:]]*=[[:space:]]*(.+)$/\1\2 = \2.at[\3].set(\4)/' "$input_file" > "$output_file"
    echo "Converted file saved as: $output_file"
else
    # Modify in-place with backup
    sed -i.bak -E 's/^([[:space:]]*)([a-zA-Z_][a-zA-Z0-9_]*)\[([^]]+)\][[:space:]]*=[[:space:]]*(.+)$/\1\2 = \2.at[\3].set(\4)/' "$input_file"
    echo "File modified in-place. Backup saved as: ${input_file}.bak"
fi

echo "Conversion complete!"