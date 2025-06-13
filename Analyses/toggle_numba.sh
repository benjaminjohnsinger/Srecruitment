# regex script to replace "@jit" with "# @jit" in all .py files in Analyses
find Analyses -name "*.py" -exec sed -i '' 's/@jit/# @jit/g' {} +
find Analyses -name "*.py" -exec sed -i '' 's/# # @jit/@jit/g' {} +