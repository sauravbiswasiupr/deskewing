#!/bin/bash 
#Script to run a batch of MLPs with different paramaters for deskewing
echo "Running MLPs with different parameters ..."
python MLP.py 25 5 1e-4 3000
python MLP.py 50 5 1e-4 3000
python MLP.py 100 5 1e-4 3000
python MLP.py 200 5 1e-4 3000
python MLP.py 10 7 1e-4 3000
python MLP.py 25 7 1e-4 3000
python MLP.py 50 7 1e-4 3000
python MLP.py 100 7 1e-4 3000
