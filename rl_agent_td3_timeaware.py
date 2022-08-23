import numpy as np
import pandas as pd
import gym
import math
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import scipy.signal
from joblib import Parallel, delayed
import itertools


# TD3 -- continuous action environments

def flat_vars(xs):
    return tf.concat([tf.reshape(x, [-1, ]) for x in xs], axis=0)


def unflat_vars(x, params):
    flat_size = lambda p: int(np.prod(p.shape.as_list()))  # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params], axis=0)
    new_params = [tf.reshape(p_new, p.shape).numpy() for p, p_new in zip(params, splits)]
    return new_params


EPS = 1e-8


class mlp(tf.keras.layers.Layer):
    def __init__(self, hidden_units, activation='relu', output_activation=None):
        super(mlp, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(i, activation=activation) for i in hidden_units[:-1]]
        self.dense_out = tf.keras.layers.Dense(hidden_units[-1], activation=output_activation)

    def call(self, obs, training):
        o = obs
        for layer in self.dense_layers:
            o = layer(o)
        out = self.dense_out(o)
        return out


# policy

class policy(tf.keras.Model):
    def __init__(self, hidden_units, act_dim=1, activation='tanh', output_activation=None):
        super(policy, self).__init__()
        self.act_dim = act_dim
        self.pi = mlp(hidden_units + [act_dim], activation=activation, output_activation=output_activation)

    def call(self, obs):
        pi = self.pi(obs)
        return pi


# Q-value

class Q(tf.keras.Model):
    def __init__(self, hidden_units, activation='relu', output_activation=None):
        super(Q, self).__init__()
        self.q = mlp(hidden_units + [1], activation=activation, output_activation=output_activation)

    def call(self, inp):
        obs, act = inp
        x = tf.concat([obs, act], axis=-1)
        q = self.q(x)
        return q


# td3 model

class mlp_td3_model(tf.keras.Model):
    def __init__(self, pi_hidden_units, act_dim, q_hidden_units, activation, pi_output_activation, q_output_activation):
        super(mlp_td3_model, self).__init__()

        self.pi = policy(pi_hidden_units, act_dim, activation, pi_output_activation)
        self.q1 = Q(q_hidden_units, activation, q_output_activation)
        self.q2 = Q(q_hidden_units, activation, q_output_activation)

    def call(self, inp):
        obs, act = inp
        pi = self.pi(obs)
        q1 = self.q1(inp)
        q2 = self.q2(inp)
        q1_pi = self.q1([obs, pi])
        return pi, q1, q2, q1_pi

    def get_action(self, obs):
        a = self.pi(obs)
        return a


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.priority_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, priority):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.priority_buf[self.ptr] = priority
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def update(self, idxs, priority):
        self.priority_buf[idxs] = priority

    def normalize_priority(self):
        self.priority_buf[:self.size] = self.priority_buf[:self.size] / np.sum(self.priority_buf[:self.size])

    def sample_batch(self, batch_size=32):
        p = self.priority_buf[:self.size] / np.sum(self.priority_buf[:self.size])
        # print(self.size, len(self.priority_buf[:self.size]), np.sum(p))
        idxs = np.random.choice(self.size, size=batch_size, p=p, replace=False)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    priority=self.priority_buf[idxs],
                    batch_indices=idxs)


class TD3Agent:
    def __init__(self,
                 env_fn,
                 max_ep_len,
                 actor_hidden_units=[128, 128],
                 critic_hidden_units=[128, 128],
                 activation='selu',
                 policy_output_activation='sigmoid',
                 q_output_activation=None,
                 pi_lr=0.0001,
                 q_lr=0.0001,
                 gradient_descents_per_update=1,
                 update_every=200,
                 start_timesteps=1e4,
                 total_timesteps=1e6,
                 replay_buffer_size=1e6,
                 alpha=0.7,
                 beta=0.5,
                 polyak=0.995,
                 discount=0.99,
                 expl_noise=0.2,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 noise_decay_rate=0.98,
                 noise_decay_steps=1000,
                 policy_freq=2,
                 eval_freq=1000,
                 save_freq=10000,
                 num_eval_episodes=10,
                 train_batch_size=512,
                 model_prefix='D:\\DeepRL_SergeyLevine\\TRPO_TF2\\orgym_td3agent'):

        self.env_fn = env_fn
        self.env = env_fn
        self.test_env = env_fn
        self.act_shape = self.env.action_space.shape
        self.act_dim = int(math.prod(self.env.action_space.shape))
        self.obs_dim = int(math.prod(self.env.observation_space.shape))
        self.obs_scaling_constant = self.env.obs_scaling_constant
        self.act_scaling_constant = self.env.action_scaling_constant
        self.act_limit_low = self.env.action_space.low
        self.act_limit_hi = self.env.action_space.high
        self.reward_scale = self.env.reward_scale
        self.max_ep_len = max_ep_len

        self.main_model = mlp_td3_model(pi_hidden_units=actor_hidden_units,
                                        act_dim=self.act_dim,
                                        q_hidden_units=critic_hidden_units,
                                        activation=activation,
                                        pi_output_activation=policy_output_activation,
                                        q_output_activation=q_output_activation)

        self.target_model = mlp_td3_model(pi_hidden_units=actor_hidden_units,
                                          act_dim=self.act_dim,
                                          q_hidden_units=critic_hidden_units,
                                          activation=activation,
                                          pi_output_activation=policy_output_activation,
                                          q_output_activation=q_output_activation)

        self.epsilon = 1e-6
        self.alpha = alpha
        self.beta = beta
        self.tau = polyak
        self.gamma = discount
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.decay_rate = noise_decay_rate
        self.decay_steps = noise_decay_steps
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_iterations = 0
        self.total_timesteps = int(total_timesteps)
        self.start_timesteps = int(start_timesteps)
        self.replay_buffer_size = int(replay_buffer_size)
        self.batch_size = int(train_batch_size)
        self.update_every = update_every
        self.gradient_descents_per_update = gradient_descents_per_update
        self.model_prefix = model_prefix
        self.eval_freq = int(eval_freq)
        self.save_freq = int(save_freq)
        self.num_eval_episodes = num_eval_episodes
        # optimizers
        self.q_optimizer = tf.keras.optimizers.Adam(learning_rate=q_lr)
        self.p_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_lr)

    def init_model(self):
        o = self.env.reset()
        o = o / self.obs_scaling_constant
        a = self.env.action_space.sample()
        a = a / self.act_scaling_constant
        _, _, _, _ = self.main_model([o.reshape(1, -1), a.reshape(1, -1)])
        _, _, _, _ = self.target_model([o.reshape(1, -1), a.reshape(1, -1)])
        # set init parameters identically in both main & target networks
        self.target_model.set_weights(self.main_model.get_weights())
        print("Main & Target Model Initialization Completed.")

    def sample_action(self, o):
        o = o / self.obs_scaling_constant
        a = self.main_model.get_action(o.reshape(1, -1))
        return a * self.act_scaling_constant

    def decayed_noise(self, init_noise, t):
        if (t - self.start_timesteps) % self.decay_steps == 0:
            return max(init_noise * self.decay_rate ** int((t - self.start_timesteps) / self.decay_steps), 0.00001)
        else:
            return init_noise

    def train(self):
        self.init_model()

        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_buffer_size)

        o, ep_ret, ep_len, d = self.env.random_reset(), 0, 0, False
        self.env.Train = True
        o = o / self.obs_scaling_constant

        for t in range(self.total_timesteps):
            # for first start_steps, randomly sample actions for better exploration

            self.expl_noise = self.decayed_noise(self.expl_noise, t)

            if t >= self.start_timesteps:
                a = self.main_model.get_action(o.reshape(1, -1)).numpy().reshape(-1, )
                a = a + np.random.normal(0, self.expl_noise, size=self.act_dim)
                a = a * self.act_scaling_constant
                a = a.reshape(self.act_shape)
                a = np.clip(a, a_min=self.act_limit_low, a_max=self.act_limit_hi)
            else:
                a = self.env.action_space.sample()

            o2, r, d, _ = self.env.step(a)
            o2 = o2 / self.obs_scaling_constant
            ep_ret += r
            ep_len += 1

            done = float(d)
            r_scaled = r / self.reward_scale
            a_scaled = a.reshape(-1, ) / self.act_scaling_constant

            priority = self.td_error(o, a_scaled, r_scaled, o2, done)

            # store state transition -- observations, actions & rewards are scaled for stability!!!
            self.replay_buffer.store(o, a_scaled, r_scaled, o2, done, priority)

            # update state
            o = o2

            # Update
            if t >= self.start_timesteps and t % self.update_every == 0:
                self.update()

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                o, ep_ret, ep_len, d = self.env.random_reset(), 0, 0, False
                o = o / self.obs_scaling_constant

            # Save, Eval model
            if (t + 1) % self.eval_freq == 0:
                # Check Agent performance
                self.eval_agent()

            if (t + 1) % self.save_freq == 0:
                # Save model
                tf.keras.models.save_model(self.main_model, f"{self.model_prefix}_{t + 1}")

    def td_error(self, state, action, reward, next_state, done):

        state = state.reshape(1, -1)
        action = action.reshape(1, -1)
        reward = reward.reshape(-1, )
        next_state = next_state.reshape(1, -1)

        next_action, q1, q2, _ = self.main_model([tf.convert_to_tensor(state), tf.convert_to_tensor(action)])
        _, q1_targ, q2_targ, _ = self.target_model(
            [tf.convert_to_tensor(next_state), tf.convert_to_tensor(next_action)])

        q1 = q1.numpy().reshape(-1, )
        q2 = q2.numpy().reshape(-1, )
        q1_targ = q1_targ.numpy().reshape(-1, )
        q2_targ = q2_targ.numpy().reshape(-1, )

        Q_backup = reward + self.gamma * (1 - done) * np.minimum(q1_targ, q2_targ)

        td_err = 0.5 * (abs(Q_backup - q1) + abs(Q_backup - q2)) + self.epsilon

        return td_err ** self.alpha

    def eval_agent(self):

        episode_rewards = []
        episode_lengths = []
        for i in range(self.num_eval_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            self.test_env.Train = False
            while not (d or (ep_len == self.max_ep_len)):
                o = o / self.obs_scaling_constant
                a = self.main_model.get_action(o.reshape(1, -1)).numpy().reshape(self.act_shape)
                a = a * self.act_scaling_constant
                a = np.clip(a, a_min=self.act_limit_low, a_max=self.act_limit_hi)
                o, r, d, _ = self.test_env.step(a)
                ep_ret += r
                ep_len += 1
            episode_rewards.append(ep_ret)
            episode_lengths.append(ep_len)

        print("Mean Reward: {}, Mean Episode Length: {}".format(sum(episode_rewards) / len(episode_rewards),
                                                                sum(episode_lengths) / len(episode_lengths)))
        # print("Current Values: Timesteps: {}, Exploration Noise: {}, learning_rate: {}".format(self.total_iterations,
        #                                                                                       self.expl_noise,
        #                                                                                       self.q_optimizer.learning_rate.numpy()))

    def update(self):
        self.total_iterations += 1
        for i in range(self.gradient_descents_per_update):

            # prioritized sampling
            batch = self.replay_buffer.sample_batch(self.batch_size)
            N = self.replay_buffer.size
            priority = batch['priority']
            # importance sampling weights
            w = ((1 / N) * (1 / priority)) ** self.beta
            w = w / np.max(w)

            with tf.GradientTape(persistent=True) as tape:
                pi, q1, q2, q1_pi = self.main_model(
                    [tf.convert_to_tensor(batch['obs1']), tf.convert_to_tensor(batch['acts'])])
                pi_targ, _, _, _ = self.target_model(
                    [tf.convert_to_tensor(batch['obs2']), tf.convert_to_tensor(batch['acts'])])

                # Target policy smoothing, by adding clipped noise to target actions
                # self.policy_noise = self.decayed_noise(self.policy_noise)
                noise = tf.random.normal(tf.shape(pi_targ), mean=0, stddev=self.policy_noise)
                noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
                next_action = pi_targ + noise
                next_action = tf.reshape(next_action, [-1, self.act_shape[0], self.act_shape[1]])
                next_action = tf.clip_by_value(next_action, self.act_limit_low, self.act_limit_hi)
                next_action = tf.reshape(next_action, [-1, self.act_dim])

                # Target Q-values, using action from target policy
                _, q1_targ, q2_targ, _ = self.target_model([tf.convert_to_tensor(batch['obs2']), next_action])

                # loss functions -- Bellman backup for Q functions, using Clipped Double-Q targets
                min_q_targ = tf.minimum(q1_targ, q2_targ)
                backup = tf.stop_gradient(batch['rews'].reshape(-1, 1) + self.gamma * (
                            1 - np.array(batch['done']).reshape(-1, 1)) * min_q_targ)

                pi_loss = -tf.reduce_mean(q1_pi)
                q1_loss = tf.reduce_mean(w * (q1 - backup) ** 2)
                q2_loss = tf.reduce_mean(w * (q2 - backup) ** 2)
                q_loss = q1_loss + q2_loss
                td_err = 0.5 * (tf.math.abs(backup - q1) + tf.math.abs(backup - q2)) + self.epsilon
                # print(backup.shape, q1.shape, td_err.shape)

                # print("iter: {}, policy_noise: {}".format(self.total_iterations, policy_noise))
                # print("iter: {}, pi_loss: {}, q1_loss: {}, q2_loss: {}, q_loss: {}".format(self.total_iterations,
                #                                                                           pi_loss.numpy(),
                #                                                                           q1_loss.numpy(),
                #                                                                           q2_loss.numpy(),
                #                                                                           q_loss.numpy()))

            # Gradient descent updates for q1 & q2 main networks
            q_grads = tape.gradient(q_loss,
                                    self.main_model.q1.trainable_variables + self.main_model.q2.trainable_variables)
            self.q_optimizer.apply_gradients(
                zip(q_grads, self.main_model.q1.trainable_variables + self.main_model.q2.trainable_variables))

            if self.total_iterations % self.policy_freq == 0:
                pi_grads = tape.gradient(pi_loss, self.main_model.pi.trainable_variables)
                self.p_optimizer.apply_gradients(zip(pi_grads, self.main_model.pi.trainable_variables))

                # target update: polyak*v_targ + (1-polyak)*v_main
                v_targ = flat_vars(self.target_model.trainable_variables)
                v_main = flat_vars(self.main_model.trainable_variables)

                new_weights = self.tau * v_targ + (1 - self.tau) * v_main
                new_weights_unflat = unflat_vars(new_weights, self.target_model.trainable_variables)

                self.target_model.set_weights(new_weights_unflat)
                # print("iter: {}, target_model updated with polyak averaging".format(self.total_iterations))

            # update priorities
            self.replay_buffer.update(batch['batch_indices'], td_err.numpy().reshape(-1, ) ** self.alpha)

    def restore(self, model_path):
        self.init_model()
        pred_model = tf.keras.models.load_model(model_path)
        self.main_model.set_weights(pred_model.get_weights())
