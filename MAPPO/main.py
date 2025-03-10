import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt

# 定义必要的参数
MODE = "rgb_array"  # render_mode
ACTION_DIM = 3  # 单辆车动作空间大小
OBS_DIM = 7  # 单辆车观察空间大小
NUM_CVS = 3  # 参与协同决策的车辆数目
NUM_OBS_VEH = 4  # 最多观察NUM_OBS_VEH辆车

TRAIN_EPISODES = 100  # 训练的总回合数
EVAL_FREQUENCY = 25  # 每经过EVAL_FREQUENCY个回合进行一次评估
EVAL_EPISODES = 10  # 每一次评估使用EVAL_EPISODES回合的结果求平均

BATCH_SIZE = 128  # 一次训练的批次大小
BUFFER_SIZE = 256  # 缓冲区大小

# 定义强化学习环境
env = gym.make("intersection-v0", render_mode=MODE)
env.unwrapped.config.update({
    "controlled_vehicles": NUM_CVS,  # 有3辆车参与协同决策
    "initial_vehicle_count": 0,  # 不要随机车辆
    "destination": None,  # 注释掉后每次车辆起终点不变
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
            "vehicles_count": NUM_OBS_VEH,  # 可观测车辆数目，不足用0填充
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],  # 观察的要素
            "normalize": True,  # 按范围进行正则化
            "absolute": True  # 绝对值或以ego为中心的相对值
        },
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 4.5, 9],  # 减速、怠速、加速对应的期望速度
            "longitudinal": True,  # 只控制变速，对应0、1、2
            "lateral": False,  # 不控制换道
        }
    },
    "spawn_probability": 0,  # 在step过程中不再产生新的车辆
    "duration": 20,  # [s]，回合最长时长
    "collision_reward": -10,  # 碰撞惩罚
    "high_speed_reward": 1,  # 效率奖励
    "arrived_reward": 1,  # 到达终点奖励
    "reward_speed_range": [7, 9],  # 速度范围内有效率奖励
    "policy_frequency": 5,  # [Hz]决策频率
    "centering_position": [0.5, 0.6],  # ego车位置视角
})


class Critic(nn.Module):
    def __init__(self, num_obs_veh=5, obs_dim=7, num_heads=4):
        super(Critic, self).__init__()
        self.num_obs_veh = num_obs_veh
        self.obs_dim = obs_dim
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # 增加维度
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=num_heads, batch_first=True)
        self.value_head = nn.Linear(32, 1)

    def forward(self, x):
        # 输入x的维度为(bs, num_obs_veh, obs_dim)
        x = x.reshape(x.size(0), self.num_obs_veh, self.obs_dim)
        x = self.fc(x)  # 经过一个全连接层，改变维度,(bs, num_obs_veh, 32)
        attention_out, _ = self.attention(x, x, x)  # 使用自注意力,(bs, num_obs_veh, 32)
        value = self.value_head(attention_out.mean(dim=1))  # 取所有车辆的均值作为Critic的输出(bs, 1)
        return value


class Actor(nn.Module):
    def __init__(self, num_obs_veh=5, obs_dim=7, action_dim=3):
        super(Actor, self).__init__()
        self.num_obs_veh = num_obs_veh
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.fc = nn.Sequential(
            nn.Linear(num_obs_veh * obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        # 输入的维度为(bs, num_obs_veh, obs_dim)
        x = x.reshape(x.size(0), -1)  # 改为(bs, num_obs_veh*obs_dim)
        policy_logits = self.fc(x)  # 变为(bs, action_dim)
        return policy_logits

    def get_action(self, x):
        policy_logits = self.forward(x)
        dist = Categorical(logits=policy_logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), dist  # 返回的维度是(bs)


class MAPP0:
    def __init__(self, num_agents=3, num_obs_veh=5, obs_dim=7, action_dim=3, gamma=0.99, clip_eps=0.05, lr=1e-3,
                 lam=0.95):
        self.num_agents = num_agents
        self.num_obs_veh = num_obs_veh
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.lr = lr
        self.lam = lam
        self.buffer = []
        self.batch_size = BATCH_SIZE
        self.buffer_size = BUFFER_SIZE

        self.critic = Critic(num_obs_veh, obs_dim)
        self.actor = Actor(num_obs_veh, obs_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def _compute_returns(self, rewards, next_values, dones):
        returns = []
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)
        if not dones[-1]:
            R = next_values[-1]
        returns[0] = R
        return returns

    def _process(self, states, actions, log_probs, rewards, values, next_values, dones):
        """ 原始数据中，states,actions,log_probs都对应多个车，需要提取成多组数据，同时计算returns与advantages"""
        returns = self._compute_returns(rewards, next_values, dones)
        advantages = []
        # advantages = self._compute_gae(rewards, values, next_values, dones)  # GAE方法
        for i in range(len(rewards)):
            advantages.append(returns[i] - values[i])  # Monte Carlo方法
            # advantages.append(rewards[i] + (1 - int(dones[i])) * self.gamma * next_values[i] - values[i])  # TD方法
        for t in range(len(states)):
            for i in range(len(states[t])):
                data = {
                    "obs": states[t][i],
                    "old_log_probs": log_probs[t][i],
                    "returns": returns[t],
                    "advantages": advantages[t],
                    "actions": actions[t][i]
                }
                self.buffer.append(data)

    def update(self, states, actions, log_probs, rewards, values, next_values, dones):
        self._process(states, actions, log_probs, rewards, values, next_values, dones)
        if len(self.buffer) >= self.buffer_size:
            self._batch_update()

    def _batch_update(self, ):
        batch = self.buffer[-self.batch_size:]
        # 提取所需数据
        obs_batch = np.array([data['obs'] for data in batch])
        old_log_probs_batch = np.array([data['old_log_probs'] for data in batch])
        returns_batch = np.array([data['returns'] for data in batch])
        advantages_batch = np.array([data['advantages'] for data in batch])
        action_batch = np.array([data['actions'] for data in batch])

        # 转化为tensor
        obs_batch = torch.tensor(obs_batch, dtype=torch.float)
        old_log_probs_batch = torch.tensor(old_log_probs_batch, dtype=torch.float)
        returns_batch = torch.tensor(returns_batch, dtype=torch.float)
        advantages_batch = torch.tensor(advantages_batch, dtype=torch.float)
        action_batch = torch.tensor(action_batch, dtype=torch.float)
        # 更新策略和价值网络
        self._update_networks(obs_batch, action_batch, old_log_probs_batch, returns_batch, advantages_batch)
        # 清空缓冲区或仅清空已处理的部分
        del self.buffer[:self.batch_size]

    def _update_networks(self, obs, action, old_log_probs, returns, advantages):
        # 获取当前策略下的新日志概率
        _, _, entropy, new_dist = self.actor.get_action(obs)
        values = self.critic(obs)
        # 计算PPO的比率
        ratio = (new_dist.log_prob(action) - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()  # 添加熵正则化项
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # 计算Critic的损失
        critic_loss = (returns - values.reshape(len(values))).pow(2).mean()
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def _compute_gae(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + (1 - int(dones[t])) * self.gamma * next_values[t] - values[t]
            gae = delta + (1 - int(dones[t])) * self.gamma * self.lam * gae
            advantages.insert(0, gae)  # 从后往前计算，因此需要插入到列表开头
        return advantages


def evaluate(env, model: MAPP0, eval_eps=10):
    """ 对模型进行评估，采用eval_eps的平均奖励作为评估结果。"""
    # 设置为评估模式
    model.critic.eval()
    model.actor.eval()

    all_reward = []
    with torch.no_grad():
        for _ in range(eval_eps):
            obs, _ = env.reset()
            done = truncated = False  # 提前终止或到时间终止
            episode_reward = 0  # 记录本回合的所有奖励
            while not (done or truncated):
                action = []
                for i in range(len(obs)):
                    obs_i = torch.tensor(obs[i], dtype=torch.float).unsqueeze(0)
                    action_i, _, _, _ = model.actor.get_action(obs_i)
                    action.append(action_i.item())  # 标量列表
                next_obs, reward, done, truncated, _ = env.step(tuple(action))
                episode_reward += reward
                obs = next_obs
            all_reward.append(episode_reward)
        print(f"模型评估：奖励均值为{np.mean(all_reward)}")
    return all_reward


eval_rewards = []
best_reward = -float('inf')
model = MAPP0(NUM_CVS, NUM_OBS_VEH, OBS_DIM, ACTION_DIM)
for episode in range(TRAIN_EPISODES):
    # 设定为训练模式
    model.critic.train()
    model.actor.train()
    obs, _ = env.reset()
    done = truncated = False
    episode_reward = 0  # 记录回合奖励
    # 记录一个回合所有数据
    states, actions, rewards, dones = [], [], [], []
    log_probs, values, next_values = [], [], []
    while not (done or truncated):
        # 记录状态 [(array, array, array),(),()]
        states.append(obs)
        # 记录状态价值 [0,1,2]
        value = model.critic(torch.tensor(obs[0], dtype=torch.float).unsqueeze(0)).detach().item()
        values.append(value)
        # 对每一个智能体选择决策动作
        action, log_prob = [], []
        for i in range(len(obs)):
            obs_i = torch.tensor(obs[i], dtype=torch.float).unsqueeze(0)
            action_i, log_prob_i, _, _ = model.actor.get_action(obs_i)
            action.append(action_i.item())  # 标量列表
            log_prob.append(log_prob_i.item())
        # 记录动作，概率[[0, 1, 2],[],[]]
        actions.append(action)
        log_probs.append(log_prob)
        # 执行动作，步进，播放
        next_obs, reward, done, truncated, _ = env.step(tuple(action))
        if MODE == "human":
            env.render()
        # 记录未来状态价值 [1,2,3]
        next_value = model.critic(torch.tensor(next_obs[0], dtype=torch.float).unsqueeze(0)).detach().item()
        next_values.append(next_value)
        # 记录奖励，[1,2,3]
        rewards.append(reward)
        # 记录完成情况
        dones.append(done)
        # 计算本轮总奖励
        episode_reward += reward
        obs = next_obs

    model.update(states, actions, log_probs, rewards, values, next_values, dones)
    print(f"Episode {episode}: Reward: {episode_reward}")
'''
    if episode % EVAL_FREQUENCY == 0:
        eval_reward = evaluate(env, model, EVAL_EPISODES)
        if np.mean(eval_reward) > best_reward:
            best_reward = np.mean(eval_reward)
            torch.save(model.actor.state_dict(), f'./actor_best.pth')
            torch.save(model.critic.state_dict(), f'./critic_best.pth')
        eval_rewards.append(eval_reward)
torch.save(model.actor.state_dict(), f'./actor_last.pth')
torch.save(model.critic.state_dict(), f'./critic_last.pth')
np.save("eval_rewards.npy", eval_rewards, allow_pickle=True)
'''

model = MAPP0(NUM_CVS, NUM_OBS_VEH, OBS_DIM, ACTION_DIM)
model.actor.load_state_dict(torch.load("actor_best.pth", map_location='cpu'))
model.critic.load_state_dict(torch.load("critic_best.pth", map_location='cpu'))

eval_rewards = np.load("eval_rewards.npy", allow_pickle=True)
# 计算每个episode的平均奖励
avg_rewards = np.mean(eval_rewards, axis=1)
# 计算每个episode的奖励范围
max_rewards = np.max(eval_rewards, axis=1)
min_rewards = np.min(eval_rewards, axis=1)
# 绘制平均奖励曲线和奖励范围的阴影区域
plt.fill_between(range(len(min_rewards)), min_rewards, max_rewards, alpha=0.3, label='Reward Range')
plt.plot(range(len(min_rewards)), avg_rewards, label='奖励均值曲线')
plt.xlabel(f'Episode (x{EVAL_FREQUENCY})')
plt.ylabel('Reward')
plt.legend()
plt.show()
