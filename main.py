# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import argparse
import os
import random
import time
from distutils.util import strtobool

from tqdm import tqdm

from env.jss_env import JssEnv
import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from wasabi import Printer

msg = Printer()

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    # 原项目默认 seed 为 0
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="JSS-CleanRL",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")

    # 'env': 'JSSEnv:jss-v1'
    # parser.add_argument("--env-id", type=str, default="JSSEnv:jss-v1",
    #                     help="the id of the environment")

    #'instance_path': 'instances/ta41',
    parser.add_argument("--instance-path", type=str, default="instances/ta41",
                        help="the path to the JSS instance file (e.g., JSS/instances/ta41)")

    # 'layer_nb': 2
    parser.add_argument("--layer-nb", type=int, default=2,
                        help="the number of layers in the neural network")
    # 'layer_size': 319
    parser.add_argument("--layer-size", type=int, default=319,
                        help="the number of nodes per layer")

    # 算法特定参数（已对齐原项目 main.py 最佳超参数）
    #自定义步长
    parser.add_argument("--total-timesteps", type=int, default=15000000,
                        help="total timesteps of the experiments")
    # 'lr': 0.0006861
    parser.add_argument("--learning-rate", type=float, default=0.0006861,
                        help="the learning rate of the optimizer")
    # 'num_envs_per_worker': 4
    parser.add_argument("--num-envs", type=int, default=4,
                        help="the number of parallel game environments")
    # metrics_smoothing_episodes': 2000,
    parser.add_argument("--num-steps", type=int, default=2000, # 配合大 batch_size 调整
                        help="the number of steps to run in each environment per policy rollout")

    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")

    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    # 'gamma': 1.0,
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="the discount factor gamma")
    # 原项目 lambda 为 1.0
    parser.add_argument("--gae-lambda", type=float, default=1.0,
                        help="the lambda for the general advantage estimation")

    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    # 原项目 num_sgd_iter 为 12
    parser.add_argument("--update-epochs", type=int, default=12,
                        help="the K epochs to update the policy")

    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    # 原项目 'clip_param': 0.541
    parser.add_argument("--clip-coef", type=float, default=0.541,
                        help="the surrogate clipping coefficient")

    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    # 原项目 'entropy_start': 0.0002458,
    parser.add_argument("--ent-coef", type=float, default=0.0002458,
                        help="coefficient of the entropy")
    # 原项目 "vf_loss_coeff": 0.7918,
    parser.add_argument("--vf-coef", type=float, default=0.7918, # 原项目 vf_loss_coeff
                        help="coefficient of the value function")

    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    # 原项目  'kl_target': 0.05047
    parser.add_argument("--target-kl", type=float, default=0.05047,
                        help="the target KL divergence threshold")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, layer_nb=2, layer_size=319):
        super().__init__()
        # 获取 JSS 的真实状态维度 (例如: 30 * 7 = 210)
        obs_dim = np.prod(envs.single_observation_space["real_obs"].shape)
        action_dim = envs.single_action_space.n

        # 根据原项目 main.py 动态构建多层感知机 (MLP)
        actor_layers = []
        critic_layers = []

        # 输入层
        actor_layers.append(layer_init(nn.Linear(obs_dim, layer_size)))
        actor_layers.append(nn.Tanh())
        critic_layers.append(layer_init(nn.Linear(obs_dim, layer_size)))
        critic_layers.append(nn.Tanh())

        # 隐藏层
        for _ in range(layer_nb - 1):
            actor_layers.append(layer_init(nn.Linear(layer_size, layer_size)))
            actor_layers.append(nn.Tanh())
            critic_layers.append(layer_init(nn.Linear(layer_size, layer_size)))
            critic_layers.append(nn.Tanh())

        # 输出头
        actor_layers.append(layer_init(nn.Linear(layer_size, action_dim), std=0.01))
        critic_layers.append(layer_init(nn.Linear(layer_size, 1), std=1.0))

        self.actor_net = nn.Sequential(*actor_layers)
        self.critic_net = nn.Sequential(*critic_layers)

    def get_value(self, x):
        # 核心修改：保留 batch 维度 (x.shape[0])，将其余多维特征展平为 obs_dim (210)
        x = x.reshape(x.shape[0], -1)
        return self.critic_net(x)

    def get_action_and_value(self, x, action_mask, action=None):
        # 核心修改：保留 batch 维度，将其余多维特征展平为 obs_dim (210)
        x = x.reshape(x.shape[0], -1)

        logits = self.actor_net(x)

        # 核心：Action Masking 修正 Logits ---
        # 针对未展平的一维张量或 Mini-batch 统一处理：将 mask 为 0 的非法动作 logit 强置为 -1e9
        inf_mask = torch.clamp(torch.log(action_mask), min=-1e9)
        masked_logits = logits + inf_mask
        # --------------------------------------------------

        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()

        # 核心注意：这里的 x 已经展平过了，直接送入 critic_net
        return action, probs.log_prob(action), probs.entropy(), self.critic_net(x)

# *替代原项目的__main__*
def train_func():
    args = parse_args()
    instance_name = os.path.basename(args.instance_path)
    run_name = f"{instance_name}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(device)
    # *环境载入函数适配*
    def make_jss_env(instance_path, seed):
        def thunk():
            env = JssEnv(instance_path)
            return env
        return thunk


    # *初始化向量化环境*
    envs = gym.vector.SyncVectorEnv(
        [make_jss_env(args.instance_path, args.seed + i) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # *实例化神经网络*
    agent = Agent(envs, layer_nb=args.layer_nb, layer_size=args.layer_size).to(device)
    # 实例化优化器
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # *提取 JSS 字典观测空间的形状与动作掩码维度*
    obs_shape = envs.single_observation_space["real_obs"].shape
    action_dim = envs.single_action_space.n

    # ALGO Logic: Storage setup
    # 改为加入obs_shape
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    # 新增：动作掩码
    masks = torch.zeros((args.num_steps, args.num_envs, action_dim)).to(device)
    # 无需改动
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    #加入：envs.reset()["real_obs"]
    next_obs_dict, _ = envs.reset()
    next_obs = torch.Tensor(next_obs_dict["real_obs"]).to(device)

    # 加入：初始动作掩码
    next_mask = torch.Tensor(next_obs_dict["action_mask"]).to(device)
    # 无需改动
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # 最优排程完工时间
    best_makespan = float('inf')

    training_loop = tqdm(range(1, num_updates + 1), desc="🏋️ PPO 训练中", unit="update")
    for update in training_loop:
        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                masks[step] = next_mask  # 加入：当前步动作掩码
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    # 加入动作掩码next_mask
                    action, logprob, _, value = agent.get_action_and_value(next_obs, next_mask)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                #加入解包：next_obs_dict，infos
                next_obs_dict, reward, terminated, truncated, infos = envs.step(action)
                done = terminated | truncated
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                #改为解包：next_obs_dict["real_obs"]
                next_obs = torch.Tensor(next_obs_dict["real_obs"]).to(device)
                next_done = torch.Tensor(done).to(device)
                next_mask = torch.tensor(next_obs_dict["action_mask"], dtype=torch.float32).to(device)
                if "episode" in infos:
                    # 遍历所有并行环境的索引
                    for env_idx in range(envs.num_envs):
                        # 只有当前 env_idx 对应的环境正好结束时，才处理它的指标
                        if infos["_episode"][env_idx]:
                            # 1. 提取基础的 Return 和 Length 数组中对应的元素
                            epi_return = infos["episode"]["r"][env_idx]
                            epi_length = infos["episode"]["l"][env_idx]

                            print(f"global_step={global_step}, episodic_return={epi_return}")
                            writer.add_scalar("charts/episodic_return", epi_return, global_step)
                            writer.add_scalar("charts/episodic_length", epi_length, global_step)

                            # 2. 捕捉车间调度的完工时间 (makespan)
                            # 新版 Gymnasium 会把子环境 info 里的自定义键打包成最外层 infos 的一个数组
                            if 'makespan' in infos:
                                # 提取当前结束的环境对应的 makespan
                                current_makespan = infos['makespan'][env_idx]

                                if current_makespan < best_makespan:
                                    best_makespan = current_makespan
                                    print(f"!!!Step: {global_step} | Found New Best Makespan: {best_makespan}")

                                writer.add_scalar("charts/episodic_makespan", current_makespan, global_step)

            # bootstrap value if not done
            # 无改动
            with torch.no_grad():
                #改为：flatten
                next_value = agent.get_value(next_obs).flatten()
                if args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                    advantages = returns - values

            # flatten the batch
            b_obs = obs.reshape((-1,) + obs_shape) #改为：obs_shape
            b_masks = masks.reshape((-1, action_dim))  # 加入：展平掩码
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    # *将b_masks[mb_inds] 一起传入计算*
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        x=b_obs[mb_inds],
                        action_mask=b_masks[mb_inds],
                        action=b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if update % 20 == 0: print(
                f"Update {update}/{num_updates} | Global Step: {global_step} | SPS: {int(global_step / (time.time() - start_time))}")
            if update % 100 == 0:
                tqdm.write(f"已完成 [{update}/{num_updates}] 轮策略更新 | 当前全局步数: {global_step}")

    print(f"Successfully Finished Training! Best historical Makespan: {best_makespan}")
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    print(f"\n⏱️  程序运行总耗时: {total_duration:.2f} 秒")

    torch.save(agent.state_dict(), f"runs/{run_name}/jss_ppo_pytorch.pt")
    envs.close()
    writer.close()
    if args.track: wandb.finish()


if __name__ == "__main__":
    train_func()

