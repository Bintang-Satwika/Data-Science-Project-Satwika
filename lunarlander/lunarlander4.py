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
    wind_power=10,
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
#actor.summary()



def create_critic_network(state_dim, action_dim):
    inputs_critic = [layers.Input(shape=(state_dim)),layers.Input(shape=(action_dim))]
    x = layers.Concatenate(axis=-1)(inputs_critic)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(1)(x)
    return models.Model(inputs= inputs_critic, outputs=output)


critic_1=create_critic_network(state_dim, action_dim)
critic_2=create_critic_network(state_dim, action_dim)


learning_rate=0.0001
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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
    print("selct_action state.shape: ",state.shape)
    action = actor(state, training=False)
    noise = np.random.normal(scale= sigma, size=np.shape(action))
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

# pemisalan buffer  mesin sudah di isi
update_memory(state=dummy_state, action=dummy_action, reward=reward, 
              next_state=next_state, apakah_manusia=apakah_manusia)
update_memory(state=dummy_state, action=dummy_action, reward=reward, 
              next_state=next_state, apakah_manusia=apakah_manusia)
update_memory(state=dummy_state, action=dummy_action, reward=reward, 
              next_state=next_state, apakah_manusia=apakah_manusia)

print("memory_B: ", memory_B)
print("memory_B_human: ",memory_B_human)


def select_action_target_network(next_state, sigma=0.2):
    action = target_actor(next_state, training=False)
    print("action target_actor:", action)
    noise = np.random.normal(scale= sigma, size=action_dim)
    noise=np.clip(noise, -0.4, 0.4)
    action = action + noise
    return action



print("\n\ntraining")

def take_minibatch_machine(memory_B, batch_size=256):
    print("Take Minibatch")
    minibatch = random.sample(memory_B, batch_size)
    
    mb_states, mb_actions, mb_rewards, mb_next_states = zip(*minibatch)
    
    # Konversi ke tensor
    mb_states = tf.convert_to_tensor(mb_states, dtype=tf.float32)
    mb_actions = tf.convert_to_tensor(mb_actions, dtype=tf.float32)
    mb_rewards = tf.convert_to_tensor(mb_rewards, dtype=tf.float32)
    mb_next_states = tf.convert_to_tensor(mb_next_states, dtype=tf.float32)
    


    return mb_states, mb_actions, mb_rewards, mb_next_states


mb_states, mb_actions, mb_rewards, mb_next_states=take_minibatch_machine(memory_B, batch_size=2)
print("mb_state: ", mb_states.shape)
print("mb_action: ", mb_actions.shape)
print("mb_rewards: ", mb_rewards.shape)
print("mb_next_state: ", mb_next_states.shape)


@tf.function
def TD_error(discount_factor, mb_states, mb_actions, mb_rewards, mb_next_states):
   
    global critic_1, critic_1_optimizer, critic_2, critic_2_optimizer
    mb_next_actions = select_action_target_network(mb_next_states, sigma=0.2)
    mb_next_actions= tf.reshape(mb_next_actions, (-1,2))
  
    # Komputasi Q target
    Q1_target = target_critic_1([mb_next_states, mb_next_actions], training=False)
    Q2_target = target_critic_2([mb_next_states, mb_next_actions], training=False)
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
    
    print("mini batch Loss critic_1:", loss_1, "mini batch Loss critic_2:", loss_2)
    del tape1
    del tape2

if len(memory_B) > 0:
    TD_error(discount_factor=0.99, mb_states=mb_states, mb_actions=mb_actions, mb_rewards=mb_rewards, mb_next_states=mb_next_states) 

def take_minibatch_human(memory_B_human, batch_size=256):
    print("\n\nTake Minibatch Human")
    minibatch = random.sample(memory_B_human, batch_size)
    #print("mini batch: ",minibatch)
    
    mb_states_human, mb_actions_human= zip(*minibatch)
    
    # Konversi ke tensor
    mb_states_human = tf.convert_to_tensor(mb_states_human, dtype=tf.float32)
    mb_actions_human = tf.convert_to_tensor(mb_actions_human, dtype=tf.float32)
   

    return mb_states_human, mb_actions_human


for _ in range(0,4):# pemisalan action manusia sudah di isi
    print(_)
    update_memory(state=dummy_state, action=dummy_action, reward=reward, 
                next_state=next_state, apakah_manusia=True)


mb_states_human, mb_actions_human=take_minibatch_human(memory_B_human, batch_size=4)
print("mb_states_human: ", mb_states_human.shape)
print("mb_actions_human: ", mb_actions_human.shape)

@tf.function
def advantage_loss(mb_states_human, mb_actions_human):
    print("\n\nAdvantage Loss")
    global critic_1, critic_1_optimizer, critic_2, critic_2_optimizer

    # Gradient descent untuk critic_1
    
    with tf.GradientTape() as tape1:
        mb_actions_policy = select_action(mb_states_human, sigma=0.4, step=0)
        Q1_machine = critic_1([mb_states_human, mb_actions_policy], training=True)
        Q1_human = critic_1([mb_states_human, mb_actions_human], training=True)
        loss_1 = tf.reduce_mean(Q1_human - Q1_machine)

    grads_critic_1 = tape1.gradient(loss_1, critic_1.trainable_variables)
    critic_1_optimizer.apply_gradients(zip(grads_critic_1, critic_1.trainable_variables))
    print("Loss Critic 1:", loss_1)
    del tape1

    # Gradient descent untuk critic_2
    with tf.GradientTape() as tape2:
        mb_actions_policy = select_action(mb_states_human, sigma=0.4, step=0)
        Q2_machine = critic_2([mb_states_human, mb_actions_policy], training=True)
        Q2_human = critic_2([mb_states_human, mb_actions_human], training=True)
        loss_2 = tf.reduce_mean(Q2_human - Q2_machine)

    grads_critic_2 = tape2.gradient(loss_2, critic_2.trainable_variables)
    critic_2_optimizer.apply_gradients(zip(grads_critic_2, critic_2.trainable_variables))
    print("Loss Critic 2:", loss_2)
    del tape2


if len(memory_B_human) > 0  and 2%1==0:
    advantage_loss(mb_states_human, mb_actions_human)


def gradient_descent_actor(mb_states):
    global actor, actor_optimizer
    print("\n\nGradient Descent Actor")
    with tf.GradientTape() as tape:
        actions = actor(mb_states, training=True)
        Q_machine = critic_1([mb_states, actions], training=False)
        loss = tf.reduce_mean(Q_machine)
    grads_actor = tape.gradient(loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(grads_actor, actor.trainable_variables))
    print("Loss Actor:", loss)
    del tape

gradient_descent_actor(mb_states)


def update_target_weights(tau=0.005):
    print("\n\nUpdate Target Weights")
    global target_actor, target_critic_1, target_critic_2

    # Update target actor weights
    for target_weights, weights in zip(target_actor.trainable_weights, actor.trainable_weights):
        target_weights.assign(tau * weights + (1 - tau) * target_weights)

    # Update target critic_1 weights
    for target_weights, weights in zip(target_critic_1.trainable_weights, critic_1.trainable_weights):
        target_weights.assign(tau * weights + (1 - tau) * target_weights)

    # Update target critic_2 weights
    for target_weights, weights in zip(target_critic_2.trainable_weights, critic_2.trainable_weights):
        target_weights.assign(tau * weights + (1 - tau) * target_weights)
    print("updated target actor dan critic weights")


# Memperbarui weights
#print(target_actor.get_weights())
if 2%1==0:
    update_target_weights(tau=0.005)
print("\n\n")
#print(target_actor.get_weights())



# def update_target_weights( tau=0.005):
#     global target_actor, target_critic_1, target_critic_2
#     weights = actor.get_weights()
#     target_weights = target_actor.get_weights()
#     for i in range(len(target_weights)):  # set tau% of target model to be new weights
#         target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
#     target_actor.set_weights(target_weights)

#     weights = critic_1.get_weights()
#     target_weights = target_critic_1.get_weights()
#     for i in range(len(target_weights)):  # set tau% of target model to be new weights
#         target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
#     target_critic_1.set_weights(target_weights)

#     weights = critic_2.get_weights()
#     target_weights = target_critic_2.get_weights()
#     for i in range(len(target_weights)):  # set tau% of target model to be new weights
#         target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
#     target_critic_2.set_weights(target_weights)
