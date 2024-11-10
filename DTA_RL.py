import torch
from torchvision import transforms, models
from PIL import Image
import pybullet as p
import pybullet_data
import gym
from gym import spaces
import time
from stable_baselines3 import PPO
import numpy as np
import os
import inv_kine.inv_kine as ik
from datetime import datetime

# For plot
posx_list = []
posy_list = []
posz_list = []
velx_list = []
rot_list = []
pow_list = []

"""def create_terrain():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    terrain_shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        #meshScale=[1, 1, 0.05],  # comment out for terrain1
        #meshScale=[0.1, 0.1, 0.02],   # comment out for terrain2
        #meshScale=[0.1, 0.1, 0.05],    # comment out for terrain3
        #meshScale=[0.1, 0.1, 0.15],    # comment out for terrain4
        #meshScale=[1.15, 1.15, 0.2],    # comment out for terrain5
        #meshScale=[1.15, 1.15, 0.3],    # comment out for terrain6
        heightfieldTextureScaling=128,
        heightfieldData=np.random.uniform(-1, 1, 256*256),
        numHeightfieldRows=256,
        numHeightfieldColumns=256
    )
    terrain = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=terrain_shape
    )
    #p.changeDynamics(terrain, -1, lateralFriction=1)  # comment out for terrain1
    #p.changeDynamics(terrain, -1, lateralFriction=1.75)   # comment out for terrain2
    #p.changeDynamics(terrain, -1, lateralFriction=1.15)   # comment out for terrain3
    #p.changeDynamics(terrain, -1, lateralFriction=1.5)    # comment out for terrain4
    #p.changeDynamics(terrain, -1, lateralFriction=0.8)    # comment out for terrain5
    #p.changeDynamics(terrain, -1, lateralFriction=0.85)    # comment out for terrain6
    p.setGravity(0, 0, -9.81)
    return terrain
"""

# Set paths
terrain_classifier_path = r"C:/Users/thapelo/Downloads/Quadruped_Robot/Quadruped_Robot/classifier/terrain_classifier.pth"
policy_models_dir = r"C:/Users/thapelo/Downloads/Quadruped_Robot/Quadruped_Robot/models/PPO"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Map terrain types to their respective pretrained models
pretrained_model_paths = {
    "terrain1": os.path.join(policy_models_dir, "PPO_terrainONE_final_309.zip"),
    "terrain2": os.path.join(policy_models_dir, "PPO_terrainTwo_final_II_309.zip"),
    "terrain3": os.path.join(policy_models_dir, "PPO_terrainTHREE_FINAL_II_309.zip"),
    "terrain4": os.path.join(policy_models_dir, "PPO_terrainFOUR_final_309.zip"),
    "terrain5": os.path.join(policy_models_dir, "PPO_terrainSIX_IV_314.zip"),
    "terrain6": os.path.join(policy_models_dir, "PPO_terrainSIX_IV_314.zip"),
}

# Load the trained terrain classifier
def load_terrain_classifier():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(256, len(pretrained_model_paths)),
        torch.nn.LogSoftmax(dim=1),
    )
    model.load_state_dict(torch.load(terrain_classifier_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

class TestudogEnv(gym.Env): 
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
        self.state = self.init_state()
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(45,), dtype=np.float32)
        #self.load_terrain_func = load_terrain_func
        self.jump_burst_enabled = False  # Initially, no jump burst
        self.stuck_count = 0  # Count how long the robot stays stuck
        self.back_leg_boost_enabled = False  # For boosting back leg movement when stuck
        self.last_z_position = None

    def init_state(self):
        self.count = 0
        self.stuck_count = 0
        self.jump_burst_enabled = False
        if not p.isConnected():
            p.connect(p.GUI)
        p.resetSimulation()

        # Comment out for uneven terrain
        #heightfield=True
        #create_terrain()  

        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
        self.testudogid = p.loadURDF("C:/Users/thapelo/Downloads/Quadruped_Robot/Quadruped_Robot/urdf/testudog.urdf", [0, 0, 0.5], [0, 0, 0, 1])
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
        self.last_z_position = body_pos[2]
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
        vel1 = action[6:9] * 1.2
        vel2 = action[9:12] * 1.2
        
        p.setJointMotorControlArray(self.testudogid, list(range(12)), p.POSITION_CONTROL,
                                    targetPositions=joint_angle, targetVelocities=np.block([vel1, vel1, vel2, vel2]),
                                    positionGains=[0.03, 0.03, 0.03, # Higher for front-left leg (Leg 0)
                                                   0.03, 0.03, 0.03, # Higher for front-right leg (Leg 1)
                                                   0.03, 0.03, 0.03, # Lower for back-left leg (Leg 2)
                                                   0.03, 0.03, 0.03],  # Lower for back-right leg (Leg 3)
                                    velocityGains=[0.2, 0.2, 0.2, # Higher velocity gain for front-left (Leg 0)
                                                   0.2, 0.2, 0.2, # Higher velocity gain for front-right (Leg 1)
                                                   0.2, 0.2, 0.2, # Lower velocity gain for back-left (Leg 2)
                                                   0.2, 0.2, 0.2])  # Lower velocity gain for back-right (Leg 3)

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

        
        top_height = 1.15  # Adjust this value based on your terrain
        
        upward_movement = body_pos[2] - self.last_z_position
        
        if upward_movement > 0:
            vel_multiplier = 1.5  # Increase speed while climbing
        else:
            vel_multiplier = 0.8  # Decrease speed while going downhill
        
        vel1 = action[6:9] * vel_multiplier
        vel2 = action[9:12] * vel_multiplier

        # Reward for forward movement
        reward = 15 * body_lin_vel[0]  # Increased reward for moving forward
        reward += 10 * abs(body_lin_vel[0])  # Increased reward for forward velocity
        # Penalize unwanted movements and energy usage
        reward -= 50 * abs(body_pos[0])  # Stronger penalty for sideways movement
        reward += 5 * (1 - abs(body_pos[0]))  # Reward for staying centered
        reward -= 25 * abs(body_rot_rpy[0])  # Heavier penalty for rolling
        reward -= 100 * abs(body_rot_rpy[1])  # Penalize pitching
        reward -= 0.05 * sum(np.abs(joint_pow))  # Penalize based on power usage
        
        upward_movement = body_pos[2] - self.last_z_position

        reward += 100 * max(0, upward_movement)  # Reward for positive (upward) movement
        reward -= 50 * max(0, -upward_movement)  # Penalize negative (downward) movement
        if body_lin_vel[0] > 0 and upward_movement > 0:
            reward += 100  # Significant reward for forward and upward movement together
        
        # Penalize backward movement (negative velocity)
        if body_lin_vel[0] < 0:
            reward -= 100
        # Reward for forward movement
        reward = 25 * body_lin_vel[0]  # Increased reward for moving forward

        # Reward for forward velocity
        reward += 10 * max(0, body_lin_vel[0])  # Only reward forward velocity, penalize backward movement
    
        # Reward for reaching the top
        if body_pos[2] >= top_height:
            reward += 100  # Significant reward for reaching the top

        # Penalize for pushing too much with back legs (based on joint velocities)
        #back_leg_effort = sum([abs(vel) for vel in joint_vel[6:12]])
        #front_leg_effort = sum([abs(vel) for vel in joint_vel[0:6]])
        #reward -= 10 * (back_leg_effort - front_leg_effort)  # Penalize imbalance
        
        # Reward for balanced movement between front and back legs
        #leg_balance = 1 - abs(front_leg_effort - back_leg_effort) / (front_leg_effort + back_leg_effort + 1e-5)
        #reward += 10 * leg_balance 

        
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
            action[0:6] = np.random.uniform(0.5, 1.0, 6)  # Random burst to help move out
            action[6:12] = np.random.uniform(0.95, 1.0, 6)
            self.jump_burst_enabled = False
        
        # Penalize flipping
        if abs(body_rot_rpy[0]) > 0.5 or abs(body_rot_rpy[1]) > 0.5 or self.stuck_count > 10:
            reward -= 200
            done = True
        
        done = False
        if body_rot_rpy[1] < 0 or self.count > 15000:
            done = True
            reward = -20 if body_rot_rpy[1] < 0 else 10

        if body_lin_vel[0] > 0 and upward_movement > 0:
            reward += 200  # Increased reward for forward and upward movement together
        
        self.last_z_position = body_pos[2]

        # Store the data for plotting (assuming these lists are defined globally elsewhere)
        global posx_list, posy_list, posz_list, velx_list, rot_list, pow_list
        posx_list.append(-body_pos[1])
        posy_list.append(body_pos[0])
        posz_list.append(body_pos[2])
        velx_list.append(-body_lin_vel[1])
        rot_list.append((np.pi / 2) - body_rot_rpy[1])
        pow_list.append(sum(np.abs(joint_pow)))

        self.state = obs
        return obs, reward, done, info

# Initialize the terrain classifier
terrain_classifier = load_terrain_classifier()

# Define transformation for terrain images
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to classify terrain
def classify_terrain(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        terrain_type = list(pretrained_model_paths.keys())[pred.item()]
    return terrain_type

# Function to load the corresponding policy model
def load_policy_model(terrain_type):
    """
    Load the policy model for the specified terrain type.
    """
    if terrain_type in pretrained_model_paths:
        policy_model_path = pretrained_model_paths[terrain_type]
        if os.path.exists(policy_model_path):
            print(f"Loading model for {terrain_type} from {policy_model_path}")
            
            # Dynamically create a new environment for this model
            env = TestudogEnv()
            model = PPO.load(policy_model_path, env=env)
            
            return model, env  # Return both the model and its associated environment
        else:
            print(f"Model not found for terrain: {terrain_type}")
            return None, None
    else:
        print(f"Invalid terrain type: {terrain_type}")
        return None, None

# Function to capture terrain image dynamically
def capture_image(step_count, robot_id):
    # Get the robot's position
    robot_position, _ = p.getBasePositionAndOrientation(robot_id)
    
    # Set the camera to look at the robot's current position
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[robot_position[0] + 1.5, robot_position[1], robot_position[2] + 1.5],
        cameraTargetPosition=robot_position,
        cameraUpVector=[0, 0, 1],
    )
    projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100.0)
    
    # Capture the camera image
    width, height, rgb_image, _, _ = p.getCameraImage(
        width=640, height=480,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix
    )
    
    # Convert the image data from bytes to a NumPy array
    rgb_array = np.array(rgb_image).reshape(height, width, 4)[:, :, :3]  # Extract RGB channels
    img = Image.fromarray(rgb_array, 'RGB')
    
    # Save the image
    image_dir = r"C:/Users/thapelo/Downloads/Quadruped_Robot/Quadruped_Robot/captured"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    filename = os.path.join(image_dir, f"terrain_step_{step_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    img.save(filename)
    print(f"Captured and saved image at step {step_count}: {filename}")
    return filename

# Main operational loop with dynamic classification and switching
if __name__ == "__main__":
    # Initialize PyBullet simulation
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(
        "C:/Users/thapelo/Downloads/Quadruped_Robot/Quadruped_Robot/urdf/testudog.urdf",
        [0, 0, 0.25],
        p.getQuaternionFromEuler([0, 0, 0]),
    )
    p.setGravity(0, 0, -9.81)

    # Initial setup
    current_policy_model = None
    current_terrain_type = None
    env = None
    obs = None

    # Testudog initial state
    x_global = 0
    y_global = 0
    z_global = 0.15
    roll = 0
    pitch = 0
    yaw = 0

    # Simulation loop
    for step in range(100000):
        # Capture terrain image every 50 steps
        if step % 1000 == 0 or current_policy_model is None:
            image_path = capture_image(step, robot_id)
            detected_terrain_type = classify_terrain(image_path, terrain_classifier)

            if detected_terrain_type != current_terrain_type:
                print(f"Switching to policy model for terrain: {detected_terrain_type}")
                current_terrain_type = detected_terrain_type
                current_policy_model, env = load_policy_model(current_terrain_type)

                # Check for successful model load
                if current_policy_model is None or env is None:
                    print(f"Failed to load model or environment for terrain: {current_terrain_type}")
                    continue

                # Reset the environment after loading a new policy
                obs = env.reset()

        # Take action with the current policy model
        if current_policy_model and obs is not None:
            action, _ = current_policy_model.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                print("Episode finished, resetting environment.")
                obs = env.reset()

        # Step the simulation
        p.stepSimulation()
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
    # Disconnect PyBullet simulation
    p.disconnect()
