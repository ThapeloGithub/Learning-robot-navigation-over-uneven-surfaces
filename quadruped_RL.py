from ctypes import pointer
from click import pass_context
import pybullet as p
import pybullet_envs
import pybullet_data
import torch
import gym
from gym import spaces
import time
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import numpy as np
import math
import os
import inv_kine.inv_kine as ik
import matplotlib.pyplot as plt

# For plotting
posx_list = []
posy_list = []
posz_list = []
velx_list = []
rot_list = []
pow_list = []

class TestudogEnv(gym.Env):
    """Custom Environment that follows the gym interface"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(TestudogEnv, self).__init__()
        # Initialize state variables
        self.x_global = 0
        self.y_global = 0
        self.z_global = 0.15
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        
        # Initialize the environment
        self.state = self.init_state()
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(45,), dtype=np.float32)
        
    def init_state(self):
        self.count = 0
        p.connect(p.GUI)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
        self.testudogid = p.loadURDF("C:/Users/thapelo/Downloads/Quadruped_Robot/Quadruped_Robot/urdf/testudog.urdf", [0, 0, 0.25], [0, 0, 0, 1])
        
        # Set the camera view
        focus, _ = p.getBasePositionAndOrientation(self.testudogid)
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-90, cameraPitch=0, cameraTargetPosition=focus)
        
        # Get initial observations
        body_pos = p.getLinkState(self.testudogid, 0)[0]
        body_rot = p.getLinkState(self.testudogid, 0)[1]
        body_rot_rpy = p.getEulerFromQuaternion(body_rot)
        body_lin_vel = p.getLinkState(self.testudogid, 0, computeLinkVelocity=1)[6]
        joint_pos = [p.getJointState(self.testudogid, i)[0] for i in range(12)]
        joint_vel = [p.getJointState(self.testudogid, i)[1] for i in range(12)]
        joint_torque = [p.getJointState(self.testudogid, i)[3] for i in range(12)]
        
        obs = list(body_pos) + list(body_lin_vel) + list(body_rot_rpy) + joint_pos + joint_vel + joint_torque
        obs = np.array(obs).astype(np.float32)
        return obs
    
    def reset(self):
        p.disconnect()
        obs = self.init_state()
        self.state = obs
        return obs
        
    def step(self, action):
        action_legpos = np.array([
            [(action[0] * 0) + 0.1373, (action[0] * 0) - 0.1373, (action[3] * 0) + 0.1373, (action[3] * 0) - 0.1373],
            [(action[1] * 0.15) - 0.102, (action[1] * 0.15) - 0.102, (action[4] * 0.15) + 0.252, (action[4] * 0.15) + 0.252],
            [(action[2] * 0.05) + 0.05, (action[2] * 0.05) + 0.05, (action[5] * 0.05) + 0.05, (action[5] * 0.05) + 0.05]
        ])
        
        joint_angle = ik.inv_kine(ik.global2local_legpos(action_legpos, self.x_global, self.y_global, self.z_global, self.roll, self.pitch, self.yaw))
        joint_angle = np.reshape(np.transpose(joint_angle), [1, 12])[0]
        
        vel1 = action[6:9]
        vel2 = action[9:12]
        p.setJointMotorControlArray(
            self.testudogid, list(range(12)), p.POSITION_CONTROL,
            targetPositions=joint_angle, targetVelocities=np.block([vel1, vel1, vel2, vel2]),
            positionGains=4 * [0.02, 0.02, 0.02], velocityGains=4 * [0.1, 0.1, 0.1]
        )
        
        focus, _ = p.getBasePositionAndOrientation(self.testudogid)
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-90, cameraPitch=0, cameraTargetPosition=focus)
        p.stepSimulation()
        
        # Collect observations
        body_pos = p.getLinkState(self.testudogid, 0)[0]
        body_rot = p.getLinkState(self.testudogid, 0)[1]
        body_rot_rpy = p.getEulerFromQuaternion(body_rot)
        body_lin_vel = p.getLinkState(self.testudogid, 0, computeLinkVelocity=1)[6]
        joint_pos = [p.getJointState(self.testudogid, i)[0] for i in range(12)]
        joint_vel = [p.getJointState(self.testudogid, i)[1] for i in range(12)]
        joint_torque = [p.getJointState(self.testudogid, i)[3] for i in range(12)]
        joint_pow = [joint_vel[i] * joint_torque[i] for i in range(12)]
        
        obs = list(body_pos) + list(body_lin_vel) + list(body_rot_rpy) + joint_pos + joint_vel + joint_torque
        obs = np.array(obs).astype(np.float32)
        
        # Reward calculation
        w1, w2, w3, w4, w5, w6, w7 = 2, 0.1, 0.5, 2, 0.5, 0.2, 0.5
        dt = 1 / 240
        reward = -w1 * body_pos[1] - w2 * sum(np.abs(joint_pow)) * dt - w3 * abs(body_pos[0]) - w4 * abs((math.pi / 2) - body_rot_rpy[1]) \
                 - w5 * body_lin_vel[1] - w6 * abs(body_lin_vel[0]) - w7 * abs(body_pos[2] - 0.16) + 0.5
        
        # Terminal conditions
        done = False
        if body_rot_rpy[1] < 0 or self.count > 5000:
            done = True
            reward = -20 if body_rot_rpy[1] < 0 else 10
        
        # Collect data for plotting
        global posx_list, posy_list, posz_list, velx_list, rot_list, pow_list
        posx_list.append(-body_pos[1])
        posy_list.append(body_pos[0])
        posz_list.append(body_pos[2])
        velx_list.append(-body_lin_vel[1])
        rot_list.append((math.pi / 2) - body_rot_rpy[1])
        pow_list.append(sum(np.abs(joint_pow)))
        
        self.state = obs
        self.count += 1
        info = {}
        return obs, reward, done, info

if __name__ == '__main__':
    # Set save directory
    model_dir = "C:/Users/thapelo/Downloads/Quadruped_Robot/Quadruped_Robot/models/PPO"
    log_dir = "C:/Users/thapelo/Downloads/Quadruped_Robot/Quadruped_Robot/log"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # testudog initial state
    x_global = 0
    y_global = 0
    z_global = 0.15
    roll = 0
    pitch = 0
    yaw = 0
    
    # Create model
    env = TestudogEnv()
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    TIMESTEPS = 50000
    count = 1

    # load model
    model_path = f"{model_dir}/15450000.zip"
    model = PPO.load(model_path,env=env)
    count = int(15450000/TIMESTEPS)
    
    """# Train loop
    while True:
        print(f"Training iteration: {count}")
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{model_dir}/{TIMESTEPS*count}")
        count += 1
        if True == False:
            break"""
    
    # Run trained model
    episodes = 1
    for ep in range(episodes):
       obs = env.reset()
       done = False
       while not done and env.count < 2000:
           action, _ = model.predict(obs)
           print(f"Predicted action: {action}")
           obs, reward, done, info = env.step(action)
           time.sleep(1 / 240)
    
    # Plotting
    size = len(posx_list)
    time_sim = np.arange(0, size, 1) / 240
    fig, axes = plt.subplots(3, 2)
    axes[0, 0].plot(time_sim, posx_list)
    axes[1, 0].plot(time_sim, posy_list)
    axes[2, 0].plot(time_sim, posz_list)
    axes[0, 1].plot(time_sim, velx_list)
    axes[1, 1].plot(time_sim, rot_list)
    axes[2, 1].plot(time_sim, pow_list)
    plt.show()
