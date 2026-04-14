import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_tensorboard_data(log_dir):
    # 找到文件夹下最新的 tfevents 文件
    event_file = [f for f in os.listdir(log_dir) if "tfevents" in f][0]
    ea = event_accumulator.EventAccumulator(os.path.join(log_dir, event_file))
    ea.Reload()
    
    # 提取奖励数据 (通常标签为 'train_episode_reward')
    # 如果你的标签不同，可以在 TensorBoard 网页端确认
    tag = 'train_episode_reward' 
    events = ea.Scalars(tag)
    
    df = pd.DataFrame([(e.step, e.value) for e in events], columns=['Step', 'Reward'])
    # 添加平滑处理
    df['Smooth_Reward'] = df['Reward'].rolling(window=20).mean()
    return df

# 2. 定义两个实验的路径
path_base = "/Users/kristin/Desktop/cse5100/onpolicy/scripts/results/MPE/simple_tag/mappo/"
df_no_comm = load_tensorboard_data(path_base + "tag_sep/run1/logs")
df_with_comm = load_tensorboard_data(path_base + "tag_comm_v1/run1/logs")

# 3. 绘图
plt.figure(figsize=(10, 6))
plt.plot(df_no_comm['Step'], df_no_comm['Smooth_Reward'], label='Baseline (No Comm)', color='blue', alpha=0.3)
plt.plot(df_no_comm['Step'], df_no_comm['Smooth_Reward'].rolling(10).mean(), color='blue', linewidth=2)

plt.plot(df_with_comm['Step'], df_with_comm['Smooth_Reward'], label='MAPPO + Comm', color='red', alpha=0.3)
plt.plot(df_with_comm['Step'], df_with_comm['Smooth_Reward'].rolling(10).mean(), color='red', linewidth=2)

plt.title('Training Reward: Communication vs. Baseline')
plt.xlabel('Training Steps')
plt.ylabel('Average Episode Reward')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("reward_comparison.png", dpi=300)
plt.show()