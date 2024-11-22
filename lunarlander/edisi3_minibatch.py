import gymnasium as gym
from tqdm import tqdm
import pygame
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque
tf.keras.backend.set_floatx('float32')
import random
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
    state_input = layers.Input(shape=(state_dim))
    x = layers.Dense(256, activation='relu')(state_input)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(action_dim, activation='tanh')(x)  # Actions dalam rentang [-1, 1]
    return models.Model(inputs=state_input, outputs=output)

actor = create_actor_network(state_dim, action_dim)
#print("actor")
actor.summary()



def create_critic_network(state_dim, action_dim):
    inputs_critic = [layers.Input(shape=(state_dim)),layers.Input(shape=(action_dim))]
    x = layers.Concatenate(axis=-1)(inputs_critic)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(1)(x)
    return models.Model(inputs= inputs_critic, outputs=output)

#print("critic")
critic_1=create_critic_network(state_dim, action_dim)
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
    #state = np.expand_dims(state, axis=0).astype(np.float32)
    print("selct_action state.shape: ",state.shape)
    action = actor(state, training=False).numpy()[0]
    noise = np.random.normal(scale= sigma, size=action_dim)
    action = action + noise
    return action

#dummy_state=np.array([1,2,3,4,5,6,7,8])
dummy_state, info = env.reset()
print("dummy_state:", dummy_state.shape)
dummy_state=np.reshape(dummy_state, (-1, 8))
print("dummy_state.shape:", dummy_state.shape)
dummy_action=select_action(dummy_state, sigma=0.4, step=0)
print("dummy_action: ",dummy_action.shape)
dummy_action=np.reshape(dummy_action, (-1, 2))
print("dummy_action: ",dummy_action.shape)
dummy_con=np.concatenate([dummy_state, dummy_action], axis=-1)
print("dummy_con: ",dummy_con.shape)
print("\n")



def compute_I(critic_1, critic_2, state, action):
    print("COMPUTE I")
    global Quen
    """Menghitung perbedaan nilai Q dari dua critic networks."""
    #state = np.expand_dims(state, axis=0).astype(np.float32) # Tambahkan batch dimensi
    print("state.shape: ",state.shape)
   # action = np.expand_dims(action, axis=0).astype(np.float32)  # Tambahkan batch dimensi
    print("action.shape: ",action.shape)
    q1 = critic_1([dummy_state, dummy_action], training=False).numpy()[0][0]
    print("q1: ",q1)
    q2 = critic_2([dummy_state, dummy_action], training=False).numpy()[0][0]
    Is=abs(q1-q2)
    Quen.append(Is)
    return Is

Is= compute_I(critic_1, critic_2, dummy_state, dummy_action)
print("Quen: ",Quen)
print("\n")



def apakah_tanya_manusia(Is, Quen, reward_satu_episode, actor_action, reward_max=200, th=5*2.718**(-3)):
    print("APAKAH TANYA MANUSIA")
    print("Is: ",Is)
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
    
action, apakah_manusia=apakah_tanya_manusia(Is, Quen, 0, dummy_action)

print("action: ",action)
next_state, reward, terminated, truncated, info = env.step(action[0])
print("\nnext_state: ",next_state)
print("reward: ",reward)
print("terminated: ",terminated)
print("truncated: ",truncated)
print("\n")



def update_memory(state, action, reward, next_state, apakah_manusia):
    global memory_B, memory_B_human
    if apakah_manusia:
        memory_B_human.append((state[0], action[0]))
    else:
        memory_B.append((state[0], action[0], reward, next_state))

update_memory(state=dummy_state, action=dummy_action, reward=reward, 
              next_state=next_state, apakah_manusia=apakah_manusia)
update_memory(state=dummy_state, action=dummy_action, reward=reward, 
              next_state=next_state, apakah_manusia=apakah_manusia)

print("memory_B: ",memory_B[0])
print("memory_B: ", memory_B[-1])
print("memory_B_human: ",memory_B_human)



def select_action_target_network(next_state, sigma=0.2):
    #next_state = np.expand_dims(next_state, axis=0).astype(np.float32)
    action = target_actor(next_state, training=True)
    print("action target_actor:", action)
    noise = np.random.normal(scale= sigma, size=action_dim)
    noise=np.clip(noise, -0.4, 0.4)
    action = action + noise
    return action

# minibatch = random.sample(memory_B, 1)

# # Unpack komponen-komponen dari minibatch
# states, actions, rewards, next_states = zip(*minibatch)
# print("state: ",tf.convert_to_tensor(states, dtype=tf.float32))
# print("action: ",actions)
# print("reward: ",rewards)
# print("next_state: ",next_states)

print("\n\ntraining")



def TD_error(discount_factor, memory_B, batch_size=256):
    print("TD ERROR")
    minibatch = random.sample(memory_B, batch_size)
    
    # Unzip komponen-komponen dari minibatch
    mb_states, mb_actions, mb_rewards, mb_next_states = zip(*minibatch)
    
    # Konversi ke tensor
    mb_states = tf.convert_to_tensor(mb_states, dtype=tf.float32)
    mb_actions = tf.convert_to_tensor(mb_actions, dtype=tf.float32)
    mb_rewards = tf.convert_to_tensor(mb_rewards, dtype=tf.float32)
    mb_next_states = tf.convert_to_tensor(mb_next_states, dtype=tf.float32)
    
    mb_next_actions = select_action_target_network(mb_next_states, sigma=0.2)
    mb_next_actions= tf.reshape(mb_next_actions, (-1,2))

    print("mb_state: ", mb_states.shape)
    print("mb_action: ", mb_actions.shape)
    print("mb_next_action: ", mb_next_actions.shape)
    print("mb_next_state: ", mb_next_states.shape)

    # Komputasi Q target
    Q1_target = target_critic_1([mb_next_states, mb_next_actions], training=True)
    Q2_target = target_critic_2([mb_next_states, mb_next_actions], training=True)
    y1 = mb_rewards + discount_factor * Q1_target
    y2 = mb_rewards + discount_factor * Q2_target
    y_min = tf.minimum(y1, y2)
    y_min = tf.stop_gradient(y_min)
    
    # gradient  descent critic_1
    with tf.GradientTape() as tape1:
        Q1 = critic_1([mb_states, mb_actions], training=True)
        loss_1 = tf.reduce_mean(tf.square(y_min - Q1))
    grads_critic_1 = tape1.gradient(loss_1, critic_1.trainable_variables)
    critic_1_optimizer.apply_gradients(zip(grads_critic_1, critic_1.trainable_variables))
    
    # gradient  descent critic_2
    with tf.GradientTape() as tape2:
        Q2 = critic_2([mb_states, mb_actions], training=True)
        loss_2 = tf.reduce_mean(tf.square(y_min - Q2))
    grads_critic_2 = tape2.gradient(loss_2, critic_2.trainable_variables)
    critic_2_optimizer.apply_gradients(zip(grads_critic_2, critic_2.trainable_variables))
    
    print("mini batch Loss critic_1:", loss_1.numpy(), "mini batch Loss critic_2:", loss_2.numpy())
    
    return loss_1.numpy(), loss_2.numpy()


if len(memory_B) > 0:
    TD_error(discount_factor=0.99, memory_B=memory_B, batch_size=2) 

def advantage_loss(Q1, Q2):
    - Q1
    Q2



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
