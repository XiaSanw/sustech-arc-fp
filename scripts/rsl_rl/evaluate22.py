"""终极评估脚本 - 完全避免torchvision导入问题"""

import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd

# 手动添加项目路径到sys.path
project_root = "/personal/limxtron1lab-main"
sys.path.insert(0, os.path.join(project_root, "exts/bipedal_locomotion"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--eval_steps", type=int, default=3000)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.device = "cuda:0"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 现在开始导入（AppLauncher之后）
import gymnasium as gym
from rsl_rl.runner import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# 直接导入配置类（绕过包的__init__.py）
from bipedal_locomotion.tasks.locomotion.robots.limx_pointfoot_env_cfg import (
    PFBlindFlatEnvCfg_PLAY
)
from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import (
    PF_TRON1AFlatPPORunnerCfg
)

# 手动注册环境（避免导入整个bipedal_locomotion包）
if "Isaac-Limx-PF-Blind-Flat-Play-v0" not in gym.envs.registry:
    gym.register(
        id="Isaac-Limx-PF-Blind-Flat-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": PFBlindFlatEnvCfg_PLAY,
            "rsl_rl_cfg_entry_point": PF_TRON1AFlatPPORunnerCfg(),
        },
    )

def main():
    checkpoint_path = args_cli.checkpoint

    print(f"\n{'='*80}")
    print(f"🎯 任务2.2量化评估 / Task 2.2 Quantitative Evaluation")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Environments: {args_cli.num_envs}")
    print(f"Evaluation steps: {args_cli.eval_steps} (~{args_cli.eval_steps*0.02:.1f} seconds)")
    print(f"{'='*80}\n")

    # 创建环境配置
    env_cfg = PFBlindFlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs

    # 创建agent配置
    agent_cfg = PF_TRON1AFlatPPORunnerCfg()
    agent_cfg.device = args_cli.device

    # 创建环境
    print("[INFO]: Creating environment...")
    env = gym.make("Isaac-Limx-PF-Blind-Flat-Play-v0", cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    # 加载policy
    print(f"[INFO]: Loading checkpoint from: {checkpoint_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(checkpoint_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    # 初始化记录
    velocity_errors = []
    orientation_errors = []
    termination_counts = 0
    total_steps = 0

    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory").flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands")

    print("开始评估... / Starting evaluation...\n")

    for step in range(args_cli.eval_steps):
        with torch.inference_mode():
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            obs, rewards, dones, infos = env.step(actions)
            obs_history = infos["observations"].get("obsHistory").flatten(start_dim=1)
            commands = infos["observations"].get("commands")

        robot = env.unwrapped.scene["robot"]

        # 1. 速度跟踪误差
        actual_lin_vel = robot.data.root_lin_vel_w[:, :2]
        actual_ang_vel = robot.data.root_ang_vel_w[:, 2:3]

        cmd_lin_vel = commands[:, :2]
        cmd_ang_vel = commands[:, 2:3]

        lin_vel_error = torch.mean((actual_lin_vel - cmd_lin_vel) ** 2, dim=1)
        ang_vel_error = ((actual_ang_vel - cmd_ang_vel) ** 2).squeeze(1)

        velocity_errors.append(torch.cat([lin_vel_error.unsqueeze(1),
                                        ang_vel_error.unsqueeze(1)], dim=1).cpu().numpy())

        # 2. 姿态稳定性
        base_quat = robot.data.root_quat_w

        from isaaclab.utils.math import quat_to_euler_xyz
        euler_angles = quat_to_euler_xyz(base_quat)

        roll = torch.abs(euler_angles[:, 0])
        pitch = torch.abs(euler_angles[:, 1])

        orientation_errors.append(torch.stack([roll, pitch], dim=1).cpu().numpy())

        # 3. 存活率
        termination_counts += torch.sum(dones).item()
        total_steps += args_cli.num_envs

        if (step + 1) % 500 == 0:
            progress = (step + 1) / args_cli.eval_steps * 100
            print(f"进度: {progress:.1f}% ({step+1}/{args_cli.eval_steps})")

    # 计算结果
    velocity_errors = np.concatenate(velocity_errors, axis=0)
    orientation_errors = np.concatenate(orientation_errors, axis=0)

    lin_vel_mse = np.mean(velocity_errors[:, 0])
    ang_vel_mse = np.mean(velocity_errors[:, 1])
    total_vel_mse = np.mean(velocity_errors)

    roll_std = np.std(orientation_errors[:, 0])
    pitch_std = np.std(orientation_errors[:, 1])
    roll_max = np.max(orientation_errors[:, 0])
    pitch_max = np.max(orientation_errors[:, 1])

    survival_rate = 1.0 - (termination_counts / total_steps)

    # 打印结果
    print(f"\n{'='*80}")
    print(f"📊 评估结果 / Evaluation Results")
    print(f"{'='*80}\n")

    print("1. 速度跟踪误差 (MSE) / Velocity Tracking Error (MSE):")
    print(f"   - 线速度MSE / Linear Velocity MSE:   {lin_vel_mse:.6f} m²/s²")
    print(f"   - 角速度MSE / Angular Velocity MSE:  {ang_vel_mse:.6f} rad²/s²")
    print(f"   - 总体MSE / Total MSE:               {total_vel_mse:.6f}")

    print("\n2. 姿态稳定性 / Orientation Stability:")
    print(f"   - Roll震荡标准差 / Roll Std:   {np.rad2deg(roll_std):.3f}° (std)")
    print(f"   - Pitch震荡标准差 / Pitch Std: {np.rad2deg(pitch_std):.3f}° (std)")
    print(f"   - Roll最大偏移 / Roll Max:     {np.rad2deg(roll_max):.3f}°")
    print(f"   - Pitch最大偏移 / Pitch Max:   {np.rad2deg(pitch_max):.3f}°")

    print("\n3. 存活率 / Survival Rate:")
    print(f"   - 存活率 / Survival Rate:      {survival_rate*100:.2f}%")
    print(f"   - 摔倒次数 / Terminations:     {termination_counts}/{total_steps}")

    print(f"\n{'='*80}\n")

    # 保存CSV
    results_df = pd.DataFrame({
        'Metric': [
            'Linear Velocity MSE (m²/s²)',
            'Angular Velocity MSE (rad²/s²)',
            'Total Velocity MSE',
            'Roll Std (deg)',
            'Pitch Std (deg)',
            'Roll Max (deg)',
            'Pitch Max (deg)',
            'Survival Rate (%)',
            'Termination Count'
        ],
        'Value': [
            lin_vel_mse,
            ang_vel_mse,
            total_vel_mse,
            np.rad2deg(roll_std),
            np.rad2deg(pitch_std),
            np.rad2deg(roll_max),
            np.rad2deg(pitch_max),
            survival_rate * 100,
            termination_counts
        ]
    })

    output_dir = os.path.dirname(checkpoint_path)
    output_file = os.path.join(output_dir, "evaluation_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"✅ 结果已保存到 / Results saved to: {output_file}\n")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()