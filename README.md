For the original project RL-Job-Shop-Scheduling： https://github.com/prosysscience/RL-Job-Shop-Scheduling.git

Due to the outdated Ray package and hardcoded TensorFlow，the project supports neither Python 3.10+ nor PyTorch-based training.

In this CleanRL-Job-Shop-Scheduling, to train the agent more efficiently, I have rewritten main.py using CleanRL, a deep RL library built on Python 3.10+ and PyTorch 2.7.0.

Meanwhile, I modified the signature of the __init__ function inside the Gymnasium environment file (./env/jss_env.py) for CleanRL compatibility.

** To upgrade, simply overwrite the original main.py and merge the ./env folder into the original project directory.

The training results are close to perfect, and the charts in WandB are shown below：

<img width="419" height="287" alt="image" src="https://github.com/user-attachments/assets/d16594b0-a740-4075-a3ad-656b1e5e6d1c" />

<img width="416" height="285" alt="image" src="https://github.com/user-attachments/assets/bccd9aca-a021-4a58-be8d-6367f5fa3cfb" />

<img width="1328" height="627" alt="image" src="https://github.com/user-attachments/assets/392ab679-00e1-4e6c-a84c-facf876ed250" />


