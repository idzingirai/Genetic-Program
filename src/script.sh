#!/bin/bash

#iterate through list with values [3, 6, 88, 100]
for i in 7 21 99 100 101 999 15897 16753 17642 21868 13408
do
    echo "Running seed $i"
    python main.py $RANDOM
done