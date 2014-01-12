#!/bin/bash 
##Script that runs on a combination of parameters for multilayer MLPs
##and stores the results/plots accordingly. I use 25 and 50 hidden neurons in the 
##2 layers 
python MLP_multi.py 5 50 50 1e-4 3000
python MLP_multi.py 10 50 50 1e-4 3000
python MLP_multi.py 30 50 50 1e-4 3000
python MLP_multi.py 75 50 50 1e-4 3000
