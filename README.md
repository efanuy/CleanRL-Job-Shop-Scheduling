For the original project RL-Job-Shop-Scheduling： https://github.com/prosysscience/RL-Job-Shop-Scheduling.git

Due to the outdated Ray package and hardcoded TensorFlow，the project supports neither Python 3.10+ nor PyTorch-based training.

In this CleanRL-Job-Shop-Scheduling, to train the agent more efficiently, I have rewritten main.py using CleanRL, a deep RL library built on Python 3.10+ and PyTorch 2.7.0.

Meanwhile, I modified the signature of the __init__ function inside the Gymnasium environment file (./env/jss_env.py) for CleanRL compatibility.

** To upgrade, simply overwrite the original main.py and merge the ./env folder into the original project directory.
