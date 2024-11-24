# Learning Robot Navigation Over Uneven Surfaces

This repository contains the implementation for training and testing quadrupedal and two-wheel robots to navigate uneven terrains. The project is organized into sections for training, testing, and terrain classification.

---

## Quadrupedal Robot

### Training and Running Models
1. **Start Training:**
   - Use `ppo_terrain1.py` to train the model on the first terrain.
   
2. **Testing Saved Models:**
   - After training, test the saved model:
     - Run `ppo_terrain5.py` to evaluate the model on Terrain 5.
     - Run `ppo_terrain6.py` to evaluate the model on Terrain 6.

3. **New Training:**
   - Start training a new model with `ppo_terrain2.py`.

4. **Testing Additional Terrains:**
   - After training, test the saved model:
     - Run `ppo_terrain3.py` to evaluate the model on Terrain 3.
     - Run `ppo_terrain4.py` to evaluate the model on Terrain 4.

### Terrain Image Capture and Classification
1. **Capture Terrain Images:**
   - Use `capture_terrain1.py`, etc., to capture images of all terrains.

2. **Train Terrain Classification Model:**
   - Run `classification.py` to train a model for classifying terrains.

### Running the Robot on Any Terrain
1. **Deploying the Robot:**
   - Use `dta_rl.py` to run the robot on any terrain.

2. **Using Saved Models:**
   - Use saved models for testing:
     - Run `saved_ppo1.py`, etc., to deploy saved models across terrains.

---

## Two-Wheel Robot

### Folder Structure
1. **Object Models:**
   - Contains the two-wheel robot XML file created in Gazebo.
   - Includes object models for the slope.

2. **Configs:**
   - Stores configuration files (`.yaml`) for training and testing experiments.

3. **Results:**
   - Contains training and testing results, including saved models.

### Source Code
1. **robot_environment.py:**
   - Defines the robot environment for PyBullet.
   - Loads the Two-Wheel Robot model and sets up the terrain.
   - Retrieves state observations and controls the robot based on model actions.
   - Implements reward functions for agent training.

2. **robot_move.py:**
   - Implements the `MoveRobot` class for managing robot actions during training and testing.
   - Reads configuration file parameters to initialize other classes.
   - Handles training episodes and logs metrics for visualization.

3. **robot_agent.py:**
   - Manages TensorFlow model operations, including updating weights and saving trained models.

4. **robot_neural_network.py:**
   - Defines neural network structures for both PyTorch and TensorFlow models.
   - Includes Fully-Connected Networks (FCNs) and model classes for DQN and SAC.
   - Implements action selection strategies for balancing exploration and exploitation.

5. **robot_train.py:**
   - The main training script for balancing and moving the Two-Wheel Robot.
   - Reads configuration files to set training parameters.
   - Supports `DIRECT` mode (no GUI) for faster training.

6. **robot_test.py:**
   - The main testing script for validating trained agents.
   - Visualizes the performance of trained models.

### Training Robot Agents
1. **Define Training Configurations:**
   - Create a new `config.yaml` under the `configs` directory.
   - Assign a unique `run_name` for each new run to generate a corresponding subdirectory under `results/train/<run_name>` for storing results and model weights.

2. **Start Training:**
   - Run the training script with the configuration file:
     ```bash
     python robot_train_hyperparameter_tuning.py --config-file ./config/main.yaml
     ```

---

## Usage Notes
- Ensure dependencies are installed using the `requirements.txt` file.
- For Two-Wheel Robots, navigate to the directory:
  `Learning-robot-navigation-over-uneven-surfaces/tree/main/Two-Wheel.Folders`.

---

### Dependencies
Refer to `requirements.txt` for a complete list of required libraries.

---

## Contributors
This project was developed to explore reinforcement learning techniques for robot navigation over uneven surfaces.

