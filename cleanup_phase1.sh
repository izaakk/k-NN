#!/bin/bash
# Phase 1 Automated Cleanup Script
# Removes ticket references from comments

set -e

echo "Phase 1.1: Removing ticket references..."

# Find all Java files with ticket references
FILES=$(find src/main/java/org/opensearch/knn -type f -name "*.java" -exec grep -l "([CWKOS]-[0-9]" {} \;)

for file in $FILES; do
    echo "Processing: $file"
    # Remove ticket references from comments
    # Pattern: (C-\d+ fix), (W-\d+ fix), (K-R\d+-\d+ fix), etc.
    sed -i 's/ (C-[0-9][^ )]*[^)]*)//' "$file"
    sed -i 's/ (W-[0-9][^ )]*[^)]*)//' "$file"
    sed -i 's/ (K-R[0-9][^ )]*[^)]*)//' "$file"
    sed -i 's/ (O-[0-9][^ )]*[^)]*)//' "$file"
    sed -i 's/ (S-[0-9][^ )]*[^)]*)//' "$file"
    sed -i 's/ (L-R[0-9][^ )]*[^)]*)//' "$file"
done

echo "Phase 1.2: Fixing arrow notation..."

# Replace → with -> in comments
find src/main/java/org/opensearch/knn -type f -name "*.java" -exec sed -i 's/→/->/g' {} \;

echo "Phase 1 automated cleanup complete!"
echo "Files modified: $(echo "$FILES" | wc -l)"
