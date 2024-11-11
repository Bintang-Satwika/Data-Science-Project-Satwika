import gymnasium as gym
from tqdm import tqdm
import pygame
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque
tf.keras.backend.set_floatx('float32')
env = gym.make(
    "LunarLander-v3",
    continuous=True,
    gravity=-10,
    enable_wind=True,
    wind_power=5.0,
    turbulence_power=0,
    render_mode='human'
)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print(state_dim, action_dim)
queue_length = 15
Quen = deque(maxlen=queue_length)
buffer_length=1e4
memory_B=deque(maxlen=int(buffer_length))
memory_B_human=deque(maxlen=int(buffer_length))


def create_actor_network(state_dim, action_dim):
    state_input = layers.Input(shape=(state_dim,))
    x = layers.Dense(256, activation='relu')(state_input)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(action_dim, activation='tanh')(x)  # Actions dalam rentang [-1, 1]
    return models.Model(inputs=state_input, outputs=output)

actor = create_actor_network(state_dim, action_dim)
#print("actor")
#actor.summary()



def create_critic_network(state_dim, action_dim):
    inputs_critic = [layers.Input(shape=(state_dim,)),layers.Input(shape=(action_dim,))]

    x = layers.Concatenate(axis=-1)(inputs_critic)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(1)(x)
    return models.Model(inputs= inputs_critic, outputs=output)

#print("critic")
critic_1=create_critic_network(state_dim, action_dim)
#critic_1.summary()
critic_2=create_critic_network(state_dim, action_dim)
#critic_2.summary()

learning_rate=0.0001
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# TIDAK WORK
# def update_target_weights(target_model,model, tau=0.005):
#     weights = model.get_weights()
#     target_weights = target_model.get_weights()
#     # for i in range(len(target_weights)):  # set tau% of target model to be new weights
#     #     target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
#     target_model.set_weights(model.get_weights())
    # return mungkin

def print_last_layer_weights(model, model_name):
    print(f"Weights of the last layer in {model_name}:")
    last_layer = model.layers[-1]
    weights = last_layer.get_weights()
    if weights:  # Jika layer memiliki bobot
        print(f"Layer {last_layer.name}:")
        print(weights)

target_actor = create_actor_network(state_dim, action_dim)
target_actor.set_weights(actor.get_weights())

target_critic_1 = create_critic_network(state_dim, action_dim)
target_critic_1.set_weights(critic_1.get_weights())
target_critic_2 = create_critic_network(state_dim, action_dim)
target_critic_2.set_weights(critic_2.get_weights())

# print_last_layer_weights(critic_2, "critic_2")
# print("\nKKKKKKKKKK\n")
# print_last_layer_weights(target_critic_2, "target_critic_2")

def select_action(state, sigma=0.4, step=0):
    if step%2000==0:
        sigma=sigma*0.998
    state = np.expand_dims(state, axis=0).astype(np.float32)
    action = actor(state, training=False).numpy()[0]
    noise = np.random.normal(scale= sigma, size=action_dim)
    action = action + noise
    return action

def select_action_target_network(next_state, sigma=0.2):
    next_state = np.expand_dims(next_state, axis=0).astype(np.float32)
    action = actor(next_state, training=True).numpy()[0]
    noise = np.random.normal(scale= sigma, size=action_dim)
    noise=np.clip(noise, -0.5, 0.5)
    action = action + noise
    return action


#dummy_state=np.array([1,2,3,4,5,6,7,8])
dummy_state, info = env.reset()
print(dummy_state)
print("dummy")
dummy_action=select_action(dummy_state, sigma=0.4, step=0)
print(dummy_action)

def compute_I(critic_1, critic_2, state, action):
    global Quen
    """Menghitung perbedaan nilai Q dari dua critic networks."""
    state = np.expand_dims(state, axis=0).astype(np.float32) # Tambahkan batch dimensi
    print("state.shape: ",state.shape)
    action = np.expand_dims(action, axis=0).astype(np.float32)  # Tambahkan batch dimensi
    print("action.shape: ",action.shape)
    q1 = critic_1([state, action], training=False).numpy()[0][0]
    print("q1: ",q1)
    q2 = critic_2([state, action], training=False).numpy()[0][0]
    Is=abs(q1-q2)
    Quen.append(Is)
    return Is

def apakah_tanya_manusia(Is, Quen, reward_satu_episode, actor_action, reward_max=200, th=5*2.718**(-3)):
    print("rewardmax/th: ",reward_max/th)
    print("reward_accumulated: ",reward_satu_episode)
    print("np.max(Quen): ",np.max(Quen))
    if Is> np.max(Quen) and reward_satu_episode < reward_max/th:
        print("milih action manusia")
        human_action= np.random.uniform(-1, 1, size=action_dim)  # Random action
        return human_action, True
    else:
        print("milih actor")
        return actor_action, False


Is= compute_I(critic_1, critic_2, dummy_state, dummy_action)
print("Quen: ",Quen)
action, apakah_manusia=apakah_tanya_manusia(Is, Quen, 0, dummy_action)
print("action: ",action)
next_state, reward, terminated, truncated, info = env.step(action)
print("\nnext_state: ",next_state)
print("reward: ",reward)
print("terminated: ",terminated)
print("truncated: ",truncated)

def update_memory(state, action, reward, next_state, apakah_manusia):
    global memory_B, memory_B_human
    if apakah_manusia:
        memory_B_human.append((state, action))
    else:
        memory_B.append((state, action, reward, next_state))

update_memory(state=dummy_state, action=dummy_action, reward=reward, 
              next_state=next_state, apakah_manusia=apakah_manusia)

print("memory_B: ",memory_B)
print("memory_B_human: ",memory_B_human)

def TD_error(reward, discount_factor, next_state, state, critic_1, critic_2, action):
    y= reward+discount_factor*np.min(critic_1([next_state, action], training=False).numpy()[0][0],
                                        critic_2([next_state, action], training=False).numpy()[0][0])
# running = True
# n_episodes = 2
# action=[0.00,0] # [(roket mati <=0, roket naik>0), (-1 : belok kiri, 0 : lurus, 1 : belok kanan)]

# for episode in tqdm(range(n_episodes)):

#     state, info = env.reset()
#     iterasi=0
#     done = False
#     print(episode)
#    # print(state)

#     # play episode
#     while not done:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
#                 running = False
#                 break
#         if not running:
#             break
#         env.render()
#         time.sleep(0.01)
#         next_state, reward, terminated, truncated, info = env.step(action)
#         # update if the environment is done and the current obs
#         done = terminated or truncated
#         state= next_state
#         iterasi +=1

#     print(f"Episode: {episode}, Total Reward:")

# pygame.display.quit()
# pygame.quit()
# env.close()
