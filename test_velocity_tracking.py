import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Limx-PF-Blind-Flat-Play-v0")
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--test_steps", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runner import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg
import bipedal_locomotion
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
import scripts.rsl_rl.cli_args as cli_args

def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)
    print("[INFO] Loading model...")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    ppo_runner.load(args_cli.checkpoint_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory").flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands")
    vel_errors = []
    ang_errors = []
    orientations = []
    print("[INFO] Testing...")
    for step in range(args_cli.test_steps):
        with torch.inference_mode():
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            obs, _, _, infos = env.step(actions)
            obs_history = infos["observations"].get("obsHistory").flatten(start_dim=1)
            commands_new = infos["observations"].get("commands")
            robot = env.unwrapped.scene["robot"]
            cmd_vel_xy = commands[:, :2]
            actual_vel_xy = robot.data.root_lin_vel_w[:, :2]
            vel_error = torch.norm(actual_vel_xy - cmd_vel_xy, dim=1)
            cmd_ang_z = commands[:, 2]
            actual_ang_z = robot.data.root_ang_vel_w[:, 2]
            ang_error = torch.abs(actual_ang_z - cmd_ang_z)
            quat = robot.data.root_quat_w
            roll = torch.atan2(2*(quat[:,0]*quat[:,1] + quat[:,2]*quat[:,3]), 1 - 2*(quat[:,1]**2 + quat[:,2]**2))
            pitch = torch.asin(2*(quat[:,0]*quat[:,2] - quat[:,3]*quat[:,1]))
            vel_errors.append(vel_error.cpu().numpy())
            ang_errors.append(ang_error.cpu().numpy())
            orientations.append(np.stack([roll.cpu().numpy(), pitch.cpu().numpy()], axis=1))
            commands = commands_new
            if (step + 1) % 200 == 0:
                print(f"Progress: {step+1}/{args_cli.test_steps}")
    vel_errors = np.concatenate(vel_errors)
    ang_errors = np.concatenate(ang_errors)
    orientations = np.concatenate(orientations)
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nLinear Velocity Error: {vel_errors.mean():.4f} m/s")
    print(f"Angular Velocity Error: {ang_errors.mean():.4f} rad/s")
    roll_deg = np.rad2deg(orientations[:, 0])
    pitch_deg = np.rad2deg(orientations[:, 1])
    print(f"Roll: {np.abs(roll_deg).mean():.2f} deg")
    print(f"Pitch: {np.abs(pitch_deg).mean():.2f} deg")
    if vel_errors.mean() < 0.3:
        print("\nVelocity tracking: GOOD")
    else:
        print("\nVelocity tracking: Need optimization")
        print("Suggestion: Increase rew_lin_vel_xy.weight to 8.0")
    print("="*60)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
