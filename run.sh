#!/bin/sh
# Run all models and output everything

for i in 32 64 128 256; do
    printf "\n\nHidden unit $i\n"
    echo "python train.py --train --model lstm --hidden_unit $i"
    python train.py --train --model lstm --hidden_unit $i
    echo "python train.py --train --model gru --hidden_unit $i"
    python train.py --train --model gru --hidden_unit $i

done