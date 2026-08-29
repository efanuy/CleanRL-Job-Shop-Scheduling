For the project RL-Job-Shop-Scheduling(github.com/prosysscience/RL-Job-Shop-Scheduling.git), the agent can not be successfully trained in python3.10+, due to the version of Ray package is old.

In this CleanRL-Job-Shop-Scheduling, to train the agent more efficiently, I have rewrite the main.py using CleanRL as the deep-RL tools, which based on python 3.10+ as well as pytorch 2.13.0. 

At the same time, I have changed some functions of the gymnasium environment adapt to CleanRL.

The file main.py and the folder /env in original RL-Job-Shop-Scheduling can be replaced by the same file/folder in this project.
