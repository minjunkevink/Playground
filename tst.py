import gym

# Create the Half Cheetah environment
env = gym.make('HalfCheetah-v4')

# Initialize the environment
observation = env.reset()

# Run a simulation for 1000 steps
for _ in range(1000):
    env.render()  # Render the environment on screen
    
    # Sample a random action from the action space
    action = env.action_space.sample()
    
    # Apply the action and get the observation, reward, done, and info
    observation, reward, done, info = env.step(action)
    
    # If the episode is done, reset the environment
    if done:
        observation = env.reset()

env.close()  # Close the environment
