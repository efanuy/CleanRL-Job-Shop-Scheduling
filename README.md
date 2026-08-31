For the project RL-Job-Shop-Scheduling(github.com/prosysscience/RL-Job-Shop-Scheduling.git), the agent cannot be successfully trained in python 3.10+, due to the outdated Ray package.

In this CleanRL-Job-Shop-Scheduling, to train the agent more efficiently, I have rewritten the main.py using CleanRL as the deep-RL tools, which based on python 3.10+ as well as pytorch 2.13.0. 

At the same time, I have changed some functions of the gymnasium environment for CleanRL compatibility.

** To upgrade, simply overwrite the original main.py and merge the /env folder into the original project directory.
