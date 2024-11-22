import gymnasium as gym
from tqdm import tqdm
import random
import pygame
import time
import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras import layers, models
from collections import deque


tf.keras.backend.set_floatx('float32')

# Direktori untuk menyimpan model dan buffer
save_dir = 'saved_models_and_buffers_part2_xxx'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

env = gym.make(
    "LunarLander-v3", 
    continuous=True,
    gravity=-10,
    enable_wind=True,
    wind_power=5,
    turbulence_power=0,
    render_mode='human'
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print("Dimensi State:", state_dim, "Dimensi Aksi:", action_dim)

buffer_length = 10000
memory_B = deque(maxlen=int(buffer_length))

running = True
n_episodes = 300
iterasi = 0
batch_size = 256
learning_rate = 0.0003  # Biasanya learning rate untuk aktor lebih kecil
sigma = 0.4  # Standar deviasi untuk eksplorasi awal
sigma_aksen= 0.2
tau = 0.005  # Untuk soft update target networks
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

def create_actor_network(state_dim, action_dim):
    state_input = layers.Input(shape=(state_dim,))
    x = layers.Dense(256, activation='relu')(state_input)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(action_dim, activation='tanh')(x)  # Actions dalam rentang [-1, 1]
    return models.Model(inputs=state_input, outputs=output)

actor = create_actor_network(state_dim, action_dim)

def create_critic_network(state_dim, action_dim):
    state_input = layers.Input(shape=(state_dim,))
    action_input = layers.Input(shape=(action_dim,))
    x = layers.Concatenate()([state_input, action_input])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(1)(x)
    return models.Model(inputs=[state_input, action_input], outputs=output)

critic_1 = create_critic_network(state_dim, action_dim)
critic_2 = create_critic_network(state_dim, action_dim)

# Optimizers
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Target Networks
target_actor = create_actor_network(state_dim, action_dim)
target_actor.set_weights(actor.get_weights())

target_critic_1 = create_critic_network(state_dim, action_dim)
target_critic_1.set_weights(critic_1.get_weights())
target_critic_2 = create_critic_network(state_dim, action_dim)
target_critic_2.set_weights(critic_2.get_weights())

def select_action(state):
    global sigma
    if iterasi%2000==0:
        sigma=sigma*0.998
    state = tf.convert_to_tensor(state, dtype=tf.float32)
    action = actor(tf.expand_dims(state, axis=0), training=False)[0]
    noise = np.random.normal(0, sigma, size=action_dim)
    action = action.numpy() + noise
    action = np.clip(action, -1, 1)
  #  print("action.shape", action.shape)
    return action

def select_action_target_network(next_state):
    action = target_actor(next_state, training=False)
    noise = tf.random.normal(shape=tf.shape(action), mean=0.0, stddev=0.2, dtype=tf.float32)
    noise = tf.clip_by_value(noise, clip_value_min=-0.5, clip_value_max=0.5)
    action = action + noise
    action = tf.clip_by_value(action, clip_value_min=-1, clip_value_max=1)
   # tf.print("target action.shape", action.shape)
    return action

def update_memory(state, action, reward, next_state, done):
    memory_B.append((state, action, reward, next_state, done))

def take_minibatch(batch_size=256):
    minibatch = random.sample(memory_B, batch_size)
    mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = zip(*minibatch)
    mb_states = tf.convert_to_tensor(mb_states, dtype=tf.float32)
    mb_actions = tf.convert_to_tensor(mb_actions, dtype=tf.float32)
    mb_rewards = tf.convert_to_tensor(mb_rewards, dtype=tf.float32)
    mb_next_states = tf.convert_to_tensor(mb_next_states, dtype=tf.float32)
    mb_dones = tf.convert_to_tensor(mb_dones, dtype=tf.float32)
    return mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones


def train_step(mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones):
    # Update Critics
    with tf.GradientTape(persistent=True) as tape:
        # Target actions
        mb_next_actions = select_action_target_network(mb_next_states)
        mb_next_actions = tf.reshape(mb_next_actions, (-1, action_dim))
        # Target Q-values
        target_Q1 = target_critic_1([mb_next_states, mb_next_actions], training=False)
        target_Q2 = target_critic_2([mb_next_states, mb_next_actions], training=False)
        target_Q = tf.minimum(target_Q1, target_Q2)
        target_Q = tf.reshape(target_Q, (-1,))
        assert target_Q.shape == (batch_size,), f"target_Q.shape: {target_Q.shape}"
        y = mb_rewards + (1 - mb_dones) * 0.99 * target_Q
        y = tf.stop_gradient(y)
        y= tf.reshape(y, (-1,))
        assert y.shape == (batch_size,), f"y.shape: {y.shape}"
        # Current Q-values
        current_Q1 = critic_1([mb_states, mb_actions], training=True)
        current_Q2 = critic_2([mb_states, mb_actions], training=True)
        current_Q1 = tf.reshape(current_Q1, (-1,))
        current_Q2 = tf.reshape(current_Q2, (-1,))
        assert current_Q1.shape == (batch_size,), f"current_Q1.shape: {current_Q1.shape}"
        # Compute critic loss
      #  tf.print("square:", tf.square(y - current_Q1).shape)
        critic_1_loss = tf.reduce_mean(tf.square(y - current_Q1))
       # tf.print("critic_1_loss:", critic_1_loss.shape)
        critic_2_loss = tf.reduce_mean(tf.square(y - current_Q2))
        #tf.print(np.shape(mb_rewards), np.shape(mb_dones), np.shape(y), np.shape(target_Q), np.shape(current_Q1), np.shape(current_Q2))
    # Update critics
    critic_1_grad = tape.gradient(critic_1_loss, critic_1.trainable_variables)
    critic_2_grad = tape.gradient(critic_2_loss, critic_2.trainable_variables)
    critic_1_optimizer.apply_gradients(zip(critic_1_grad, critic_1.trainable_variables))
    critic_2_optimizer.apply_gradients(zip(critic_2_grad, critic_2.trainable_variables))
    del tape

    # Delayed Policy Updates
    if iterasi % 20 == 0:
        with tf.GradientTape() as tape:
            actions = actor(mb_states, training=True)
            actor_loss = -tf.reduce_mean(critic_1([mb_states, actions], training=True))
        actor_grad = tape.gradient(actor_loss, actor.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor.trainable_variables))
        del tape
        # Soft update target networks
        update_target_weights()

def update_target_weights():
    for target_param, param in zip(target_actor.variables, actor.variables):
        target_param.assign(tau * param + (1 - tau) * target_param)
    for target_param, param in zip(target_critic_1.variables, critic_1.variables):
        target_param.assign(tau * param + (1 - tau) * target_param)
    for target_param, param in zip(target_critic_2.variables, critic_2.variables):
        target_param.assign(tau * param + (1 - tau) * target_param)

# Fungsi untuk menyimpan model
def save_models(episode):
    actor_save_path = os.path.join(save_dir, f'actor_episode_{episode}')
    critic_1_save_path = os.path.join(save_dir, f'critic_1_episode_{episode}')
    critic_2_save_path = os.path.join(save_dir, f'critic_2_episode_{episode}')
    target_actor_save_path = os.path.join(save_dir, f'target_actor_episode_{episode}')
    target_critic_1_save_path = os.path.join(save_dir, f'target_critic_1_episode_{episode}')
    target_critic_2_save_path = os.path.join(save_dir, f'target_critic_2_episode_{episode}')

    actor.save(actor_save_path)
    critic_1.save(critic_1_save_path)
    critic_2.save(critic_2_save_path)
    target_actor.save(target_actor_save_path)
    target_critic_1.save(target_critic_1_save_path)
    target_critic_2.save(target_critic_2_save_path)

    print(f'Models saved at episode {episode}')

# Fungsi untuk menyimpan replay buffer
def save_replay_buffer(memory_B, filename):
    with open(filename, 'wb') as f:
        pickle.dump(memory_B, f)
    print(f'Replay buffer saved to {filename}')




# Loop Pelatihan
for episode in tqdm(range(1, n_episodes + 1)):
    state, info = env.reset()
    done = False
    reward_satu_episode = 0

    while not done:
        iterasi += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        if not running:
            break

        action = select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        reward_satu_episode += reward
        done = terminated or truncated

        update_memory(state, action, reward, next_state, done)

        state = next_state

        # Update Networks
        if len(memory_B) > batch_size:
            banyak_batch=int(len(memory_B)/batch_size)+1
            #if iterasi % 10 == 0:
                #print("banyak_batch", banyak_batch)
            mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = take_minibatch(batch_size)
            train_step(mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones)
            # Buat dask bag dan proses paralel

    print(f"\nEpisode: {episode}, Reward: {reward_satu_episode}, Iterasi: {iterasi}")

    # Menyimpan model dan buffer setiap 20 episode
    if episode % 20 == 0:
        save_models(episode)
        save_replay_buffer(memory_B, filename=os.path.join(save_dir, f'replay_buffer_episode_{episode}.pkl'))

pygame.display.quit()
pygame.quit()
env.close()

from matplotlib import pyplot as plt
# Plotting Episode Statistics
episode_rewards = np.array(env.return_queue)
episode_lengths = np.array(env.length_queue)

fig, axs = plt.subplots(1, 2, figsize=(20, 8))

# Plot Episode Rewards
axs[0].plot(episode_rewards)
axs[0].set_title("Episode Rewards")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")

# Plot Episode Lengths
axs[1].plot(episode_lengths)
axs[1].set_title("Episode Lengths")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Length")

plt.tight_layout()
plt.show()