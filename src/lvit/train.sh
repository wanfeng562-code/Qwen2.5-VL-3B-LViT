#!/bin/bash
# Training script for LViT with TextDiff datasets
# Usage: bash train.sh <dataset_name>
# Available datasets: monuseg, mosmeddata, cov19, synapse

DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Error: Please provide a dataset name"
    echo "Usage: bash train.sh <dataset_name>"
    echo "Available datasets: monuseg, mosmeddata, cov19, synapse"
    exit 1
fi

# Validate dataset name
if [[ ! "$DATASET" =~ ^(monuseg|mosmeddata|cov19|synapse)$ ]]; then
    echo "Error: Invalid dataset name: $DATASET"
    echo "Available datasets: monuseg, mosmeddata, cov19, synapse"
    exit 1
fi

echo "========================================"
echo "Training LViT on dataset: $DATASET"
echo "========================================"

# Backup original Config.py
if [ ! -f Config.py.bak ]; then
    cp Config.py Config.py.bak
fi

# Update task_name in Config.py
python -c "
import re
with open('Config.py', 'r', encoding='utf-8') as f:
    content = f.read()
# Update task_name
content = re.sub(
    r\"task_name = '[^']*'\",
    f\"task_name = '$DATASET'\",
    content
)
with open('Config.py', 'w', encoding='utf-8') as f:
    f.write(content)
print(f'Updated Config.py: task_name = {repr(\"$DATASET\")}')
"

# Run training
python train_model.py

echo "Training completed for $DATASET"
