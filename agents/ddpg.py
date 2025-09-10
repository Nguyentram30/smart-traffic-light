import numpy as np
import tensorflow as tf
from keras.src.layers import Dense
from tensorflow.keras import Model, layers
from agents.replay_buffer import ReplayBuffer
from tensorflow.keras.optimizers import Adam
from utils import soft_update

class Actor(Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = Dense(256, activation='relu')
        self.fc2 = Dense(256, activation='relu')
        self.out = Dense(action_dim, activation='tanh')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)


class Critic(Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = Dense(256, activation='relu')
        self.fc2 = Dense(256, activation='relu')
        self.out = Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)


class DDPGAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.005, lr=1e-4, buffer_size=100000):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.set_weights(self.actor.get_weights())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.set_weights(self.critic.get_weights())

        self.actor_opt = Adam(lr)
        self.critic_opt = Adam(lr)

        self.replay = ReplayBuffer(buffer_size)
        self.gamma, self.tau = gamma, tau
        self.action_dim = action_dim

    def select_action(self, state, noise=0.1):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = self.actor(state)[0].numpy()
        action += noise * np.random.randn(self.action_dim)
        return np.clip(action, -1, 1)

    def optimize(self, batch_size=64):
        if len(self.replay) < batch_size:
            return
        batch = self.replay.sample(batch_size)
        state = np.vstack(batch.state).astype(np.float32)
        action = np.vstack(batch.action).astype(np.float32)
        reward = np.array(batch.reward, dtype=np.float32).reshape(-1, 1)
        next_state = np.vstack(batch.next_state).astype(np.float32)
        done = np.array(batch.done, dtype=np.float32).reshape(-1, 1)

        #Critic update
        next_action = self.actor_target(next_state)
        target_q = self.critic_target(next_state, next_action)
        y = reward + self.gamma * target_q * (1 - done)

        with tf.GradientTape() as tape:
            q_val = self.critic(state, action)
            critic_loss = tf.reduce_mean(tf.square(y - q_val))

        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

        #Actor update
        with tf.GradientTape() as tape:
            actions_pred = self.actor(state)
            actor_loss = -tf.reduce_mean(self.critic(state, actions_pred))

        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)