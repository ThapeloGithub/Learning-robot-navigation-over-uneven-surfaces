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

# Terrain creation function
def create_terrain_1():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    terrain_shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[1, 1, 0.05],
        heightfieldTextureScaling=128,
        heightfieldData=np.random.uniform(-1, 1, 256*256),
        numHeightfieldRows=256,
        numHeightfieldColumns=256
    )
    terrain = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=terrain_shape
    )
    p.changeDynamics(terrain, -1, lateralFriction=1)
    p.setGravity(0, 0, -9.81)
    return terrain

# Custom Gym Environment
class TestudogEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(TestudogEnv, self).__init__()
        self.state = self.init_state()
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(45,), dtype=np.float32)
        self.jump_burst_enabled = False  # Initially, no jump burst
        self.stuck_count = 0  # Count how long the robot stays stuck
        self.back_leg_boost_enabled = False  # For boosting back leg movement when stuck

    def init_state(self):
        self.count = 0
        self.stuck_count = 0
        self.jump_burst_enabled = False
        p.connect(p.GUI)
        p.resetSimulation()
        heightfield=True
        create_terrain_1()  # Create uneven terrain
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.testudogid = p.loadURDF("C:/Users/thapelo/Downloads/Quadruped_Robot/Quadruped_Robot/urdf/testudog.urdf", [0, 0, 0.25], [0, 0, 0, 1])
        focus, _ = p.getBasePositionAndOrientation(self.testudogid)
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-90, cameraPitch=0, cameraTargetPosition=focus)
        
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
        action_legpos = np.array([[(action[0] * 0) + 0.1373, (action[0] * 0) - 0.1373, (action[3] * 0) + 0.1373, (action[3] * 0) - 0.1373],
                                  [(action[1] * 0.15) - 0.102, (action[1] * 0.15) - 0.102, (action[4] * 0.15) + 0.252, (action[4] * 0.15) + 0.252],
                                  [(action[2] * 0.05) + 0.05, (action[2] * 0.05) + 0.05, (action[5] * 0.05) + 0.05, (action[5] * 0.05) + 0.05]])

        joint_angle = ik.inv_kine(ik.global2local_legpos(action_legpos, x_global, y_global, z_global, roll, pitch, yaw))
        joint_angle = np.reshape(np.transpose(joint_angle), [1, 12])[0]
        vel1 = action[6:9]
        vel2 = action[9:12]
        
        p.setJointMotorControlArray(self.testudogid, list(range(12)), p.POSITION_CONTROL,
                                    targetPositions=joint_angle, targetVelocities=np.block([vel1, vel1, vel2, vel2]),
                                    positionGains=4 * [0.03, 0.03, 0.03], velocityGains=4 * [0.2, 0.2, 0.2])
        focus, _ = p.getBasePositionAndOrientation(self.testudogid)
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-90, cameraPitch=0, cameraTargetPosition=focus)
        p.stepSimulation()
        
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
        info = {}
        self.count += 1
        
        # Define a "top" height for the climb
        top_height = 1.0  # Adjust this value based on your terrain

        # Reward for forward movement
        reward = 15 * body_lin_vel[0]  # Increased reward for moving forward

        # Reward for forward velocity
        reward += 10 * max(0, body_lin_vel[0])  # Only reward forward velocity, penalize backward movement
    
        # Reward for reaching the top
        if body_pos[2] >= top_height:
            reward += 100  # Significant reward for reaching the top

        # Penalize backward movement (negative velocity)
        if body_lin_vel[0] < 0:
            reward -= 50  # Penalty for moving backward
        # Penalize unwanted movements and energy usage
        reward -= 40 * abs(body_pos[0])  # Stronger penalty for sideways movement
        reward += 5 * (1 - abs(body_pos[0]))  # Reward for staying centered
        reward -= 25 * abs(body_rot_rpy[0])  # Heavier penalty for rolling
        reward -= 20 * abs(body_rot_rpy[1])  # Penalize pitching
        reward -= 0.05 * sum(np.abs(joint_pow))  # Penalize based on power usage

        # Penalize erratic velocity changes
        reward -= 10 * abs(body_lin_vel[0] - body_lin_vel[1])

        # Encourage any movement
        reward += 2  # Base reward for movement

        # Handle stuck scenarios
        if abs(body_pos[1]) < 0.01 and abs(body_lin_vel[1]) < 0.01 and not self.jump_burst_enabled:
            self.stuck_count += 1
            if self.stuck_count > 2:  # Penalize being stuck after 3 steps
                self.jump_burst_enabled = True
                reward -= 50
        else:
            self.stuck_count = 0  # Reset if robot moves

        if self.jump_burst_enabled:
            reward += 80  # Reward for escaping being stuck
            action[0:6] = np.random.uniform(0.95, 1.0, 6)  # Random burst to help move out
            action[6:12] = np.random.uniform(0.95, 1.0, 6)
            self.jump_burst_enabled = False
        
        # Penalize flipping
        if abs(body_rot_rpy[0]) > 0.5 or abs(body_rot_rpy[1]) > 0.5 or self.stuck_count > 10:
            reward -= 100
            done = True
        
        done = False
        if body_rot_rpy[1] < 0 or self.count > 15000:
            done = True
            reward = -20 if body_rot_rpy[1] < 0 else 10
        
        global posx_list, posy_list, posz_list, velx_list, rot_list, pow_list
        posx_list.append(-body_pos[1])
        posy_list.append(body_pos[0])
        posz_list.append(body_pos[2])
        velx_list.append(-body_lin_vel[1])
        rot_list.append((math.pi / 2) - body_rot_rpy[1])
        pow_list.append(sum(np.abs(joint_pow)))
        
        self.state = obs
        return obs, reward, done, info

if __name__ == '__main__':
    # Set save directory
    model_dir = "C:/Users/thapelo/Downloads/Quadruped_Robot/Quadruped_Robot/models/PPO"
    log_dir = "C:/Users/thapelo/Downloads/Quadruped_Robot/Quadruped_Robot/log"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Testudog initial state
    x_global = 0
    y_global = 0
    z_global = 0.15
    roll = 0
    pitch = 0
    yaw = 0
        
    # Create model
    env = TestudogEnv()
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    # Load model
    model_path = f"{model_dir}/PPO_terrainONE_final_309"
    model = PPO.load(model_path, env=env)
    
    # Run the trained model
    episodes = 1
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done and env.count < 20000:
            action, _ = model.predict(obs)
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
