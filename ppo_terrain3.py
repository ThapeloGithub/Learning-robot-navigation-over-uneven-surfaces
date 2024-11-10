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

# For plot
posx_list = []
posy_list = []
posz_list = []
velx_list = []
rot_list = []
pow_list = []

# Terrain creation function
def create_terrain_3():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load a flat base plane
    #plane_id = p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
    terrain_shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[0.1, 0.1, 0.05],
        heightfieldTextureScaling=128,
        heightfieldData=np.random.uniform(-1, 1, 512*512),
        numHeightfieldRows=512,
        numHeightfieldColumns=512
    )
    terrain = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=terrain_shape,
        basePosition=[0, 0, 0]
    )
    p.changeDynamics(terrain, -1, lateralFriction=1.15)
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
        create_terrain_3()  # Create uneven terrain
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
        vel1 = action[6:9] * 1.5
        vel2 = action[9:12] * 1.5
        
        p.setJointMotorControlArray(self.testudogid, list(range(12)), p.POSITION_CONTROL,
                                    targetPositions=joint_angle, targetVelocities=np.block([vel1, vel1, vel2, vel2]),
                                    positionGains=[0.086, 0.086, 0.086, # Higher for front-left leg (Leg 0)
                                                   0.086, 0.086, 0.086, # Higher for front-right leg (Leg 1)
                                                   0.053, 0.053, 0.053, # Lower for back-left leg (Leg 2)
                                                   0.053, 0.053, 0.053],  # Lower for back-right leg (Leg 3)
                                    velocityGains=[0.5325, 0.5325, 0.5325, # Higher velocity gain for front-left (Leg 0)
                                                   0.5325, 0.5325, 0.5325, # Higher velocity gain for front-right (Leg 1)
                                                   0.5325, 0.5325, 0.5325, # Lower velocity gain for back-left (Leg 2)
                                                   0.5325, 0.5325, 0.5325])  # Lower velocity gain for back-right (Leg 3)
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
        
        """"reward = -2 * body_pos[1] - 0.1 * sum(np.abs(joint_pow)) / 240 - 0.5 * abs(body_pos[0]) - 2 * abs((math.pi / 2) - body_rot_rpy[1]) \
                 - 0.5 * body_lin_vel[1] - 0.2 * abs(body_lin_vel[0]) - 0.5 * abs(body_pos[2] - 0.16) + 0.5"""

        # Reward for forward movement
        reward = 55 * body_lin_vel[0]  # Increased reward for moving forward
        reward += 20 * abs(body_lin_vel[0])  # Increased reward for forward velocity
        # Penalize unwanted movements and energy usage
        reward -= 60 * abs(body_pos[0])  # Stronger penalty for sideways movement
        reward += 5 * (1 - abs(body_pos[0]))  # Reward for staying centered
        reward -= 25 * abs(body_rot_rpy[0])  # Heavier penalty for rolling
        reward -= 20 * abs(body_rot_rpy[1])  # Penalize pitching
        reward -= 0.1 * sum(np.abs(joint_pow))  # Penalize based on power usage

        # Penalize erratic velocity changes
        reward -= 10 * abs(body_lin_vel[0] - body_lin_vel[1])

        # Encourage any movement
        reward += 2  # Base reward for movement

        # Handle stuck scenarios
        if abs(body_pos[1]) < 0.01 and abs(body_lin_vel[1]) < 0.01 and not self.jump_burst_enabled:
            self.stuck_count += 1
            if self.stuck_count > 2:  # Penalize being stuck after 3 steps
                self.jump_burst_enabled = True
                reward -= 100
        else:
            self.stuck_count = 0  # Reset if robot moves

        if self.jump_burst_enabled:
            reward += 100  # Reward for escaping being stuck
            action[0:6] = np.random.uniform(0.9, 1.0, 6)  # Random burst to help move out
            action[6:12] = np.random.uniform(0.9, 1.0, 6)
            self.jump_burst_enabled = False
        
        # Penalize flipping
        if abs(body_rot_rpy[0]) > 0.5 or abs(body_rot_rpy[1]) > 0.5 or self.stuck_count > 10:
            reward -= 150
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
    model_dir = r"C:/Users/thapelo/Downloads/Quadruped_Robot/Quadruped_Robot/models/PPO"
    log_dir = r"C:/Users/thapelo/Downloads/Quadruped_Robot/Quadruped_Robot/log"
    
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

    # Create environment
    env = TestudogEnv()
    
    TIMESTEPS = 50000  # Number of timesteps per training iteration
    learning_rate = 0.0005  # New learning rate to apply

    # Load pre-trained model if available
    model_path = f"{model_dir}/PPO_terrainSIX_IV_310.zip"  # Adjust the path topre-trained model
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model = PPO.load(model_path, env=env)  # Load the pre-trained model
        # Override the learning rate in the loaded model
        model.learning_rate = learning_rate  # Change the learning rate after loading
        model.policy.optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)  # Update the optimizer with new learning rate

        count = int(15450000 / TIMESTEPS )  # Set the count based on the loaded model's timesteps
        print(f"Resuming training from iteration {count}")
    else:
        print("No pre-trained model found. Starting fresh training.")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=learning_rate)
        count = 1

    # Training loop
    try:
        while True:
            print(f"Training iteration: {count}")
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")

            # Save the model after each iteration
            model_save_path = f"{model_dir}/PPO_terrainTHREE_FINAL_II_{count}.zip"
            print(f"Saving model to: {model_save_path}")
            model.save(model_save_path)
            print(f"Model saved successfully at iteration {count}")

            count += 1

    except KeyboardInterrupt:
        # Save the model on interruption
        print(f"Training interrupted. Saving the model as PPO_terrain_level_training_{count}.zip")
        model_save_path = f"{model_dir}/PPO_terrainTHREE_FINAL_II_{count}.zip"
        model.save(model_save_path)
        print(f"Model saved successfully at interruption: {model_save_path}")
        
    finally:
        # PyBullet disconnect
        print("Disconnecting PyBullet...")
        p.disconnect()
        print("PyBullet disconnected.")

    # Plot the collected data
    size = len(posx_list)
    time_sim = np.arange(0, size, 1) / 240
    fig, axes = plt.subplots(3, 2)

    # Titles and labels for the plots
    axes[0, 0].plot(time_sim, posx_list)
    axes[0, 0].set_title('Position X vs Time')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position X (m)')

    axes[1, 0].plot(time_sim, posy_list)
    axes[1, 0].set_title('Position Y vs Time')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Position Y (m)')

    axes[2, 0].plot(time_sim, posz_list)
    axes[2, 0].set_title('Position Z vs Time')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Position Z (m)')

    axes[0, 1].plot(time_sim, velx_list)
    axes[0, 1].set_title('Velocity X vs Time')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity X (m/s)')

    axes[1, 1].plot(time_sim, rot_list)
    axes[1, 1].set_title('Rotation vs Time')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Rotation (rad)')

    axes[2, 1].plot(time_sim, pow_list)
    axes[2, 1].set_title('Power vs Time')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Power (W)')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()