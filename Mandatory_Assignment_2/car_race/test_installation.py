import gym

env = gym.make('CarRacing-v2', render_mode="human")
observation = env.reset()
for t in range(1500):
    #print(t)
    env.render()
    action = env.action_space.sample()
    observation, reward, ter, trunc, info = env.step(action)
    #print(done)
    if ter or trunc:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()
