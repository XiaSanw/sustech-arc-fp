"""量化评估脚本 - 计算MSE、姿态稳定性、存活率 / Quantitative evaluation script"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
from isaaclab.app import AppLauncher

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default=None, help="Task name")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint path")
parser.add_argument("--num_envs", type=int, default=100, help="Number of evaluation environments")
parser.add_argument("--eval_steps", type=int, default=3000, help="Evaluation steps (~60 seconds @ 50Hz)")

import cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # 强制headless模式

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runner import OnPolicyRunner
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import bipedal_locomotion

def main():
    # 解析环境配置
    env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # 加载checkpoint
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path

    print(f"\n{'='*80}")
    print(f"🎯 任务2.2量化评估 / Task 2.2 Quantitative Evaluation")
    print(f"{'='*80}")
    print(f"Checkpoint: {resume_path}")
    print(f"Environments: {args_cli.num_envs}")
    print(f"Evaluation steps: {args_cli.eval_steps} (~{args_cli.eval_steps*0.02:.1f} seconds)")
    print(f"{'='*80}\n")

    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    # 加载policy
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    # 初始化记录数组
    velocity_errors = []
    orientation_errors = []
    termination_counts = 0
    total_steps = 0

    # 重置环境
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory").flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands")

    print("开始评估... / Starting evaluation...")

    # 评估循环
    for step in range(args_cli.eval_steps):
        with torch.inference_mode():
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            obs, rewards, dones, infos = env.step(actions)
            obs_history = infos["observations"].get("obsHistory").flatten(start_dim=1)
            commands = infos["observations"].get("commands")

        # 获取机器人状态
        robot = env.unwrapped.scene["robot"]

        # 1. 速度跟踪误差 (MSE)
        actual_lin_vel = robot.data.root_lin_vel_w[:, :2]  # (num_envs, 2) - vx, vy
        actual_ang_vel = robot.data.root_ang_vel_w[:, 2:3]  # (num_envs, 1) - omega_z

        cmd_lin_vel = commands[:, :2]  # 前两维是线速度命令
        cmd_ang_vel = commands[:, 2:3]  # 第三维是角速度命令

        lin_vel_error = torch.mean((actual_lin_vel - cmd_lin_vel) ** 2, dim=1)  # (num_envs,)
        ang_vel_error = ((actual_ang_vel - cmd_ang_vel) ** 2).squeeze(1)  # (num_envs,)

        velocity_errors.append(torch.cat([lin_vel_error.unsqueeze(1),
                                        ang_vel_error.unsqueeze(1)], dim=1).cpu().numpy())

        # 2. 姿态稳定性 (Roll/Pitch震荡)
        base_quat = robot.data.root_quat_w  # (num_envs, 4) - [x, y, z, w]

        # 将四元数转换为欧拉角 (roll, pitch, yaw)
        # 使用Isaac Lab的工具函数
        from isaaclab.utils.math import quat_to_euler_xyz
        euler_angles = quat_to_euler_xyz(base_quat)  # (num_envs, 3) - [roll, pitch, yaw]

        roll = torch.abs(euler_angles[:, 0])  # Roll绝对值
        pitch = torch.abs(euler_angles[:, 1])  # Pitch绝对值

        orientation_errors.append(torch.stack([roll, pitch], dim=1).cpu().numpy())

        # 3. 存活率 (摔倒检测)
        termination_counts += torch.sum(dones).item()
        total_steps += args_cli.num_envs

        # 每500步打印进度
        if (step + 1) % 500 == 0:
            progress = (step + 1) / args_cli.eval_steps * 100
            print(f"进度: {progress:.1f}% ({step+1}/{args_cli.eval_steps})")

    # 计算最终指标
    velocity_errors = np.concatenate(velocity_errors, axis=0)  # (total_steps, 2)
    orientation_errors = np.concatenate(orientation_errors, axis=0)  # (total_steps, 2)

    # 统计结果
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

    # 保存结果到CSV
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

    output_dir = os.path.dirname(resume_path)
    output_file = os.path.join(output_dir, "evaluation_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"✅ 结果已保存到 / Results saved to: {output_file}\n")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()