For the original project RL-Job-Shop-Scheduling： https://github.com/prosysscience/RL-Job-Shop-Scheduling.git

Due to the outdated Ray package and hardcoded TensorFlow，the project supports neither Python 3.10+ nor PyTorch-based training.

In this CleanRL-Job-Shop-Scheduling, to train the agent more efficiently, I have rewritten main.py using CleanRL, a deep RL library built on Python 3.10+ and PyTorch 2.7.0.

Meanwhile, I modified the signature of the __init__ function inside the Gymnasium environment file (./env/jss_env.py) for CleanRL compatibility.

** To upgrade, simply overwrite the original main.py and merge the ./env folder into the original project directory.

The training charts in WandB has been shown below：
<img width="429" height="304" alt="image" src="https://github.com/user-attachments/assets/297d65c1-2d99-450a-aeca-4e6f1e08d372" />

<img width="446" height="299" alt="image" src="https://github.com/user-attachments/assets/4eabf26d-d916-40a0-97f1-8c526337ed24" />
