#!/bin/bash

for i in {1..10}
do
    echo "Running script $i"
    python main.py $RANDOM
done