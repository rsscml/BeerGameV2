import numpy as np
import pandas as pd
import math
import gym
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import scipy.signal
from joblib import Parallel, delayed
import itertools

# PPO Agent -- works best in non-sparse reward environments

EPS = 1e-8

def gaussian_likelihood_tfd(x, mu, std):
    dist = tfd.Normal(mu, std)
    likelihood = dist.log_prob(x)
    return tf.math.reduce_sum(likelihood, axis=1)


def gaussian_likelihood_calc(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.math.reduce_sum(pre_sum, axis=1)


class mlp(tf.keras.layers.Layer):
    def __init__(self, hidden_units, activation='selu', output_activation=None):
        super(mlp, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(i, activation=activation) for i in hidden_units[:-1]]
        self.dense_out = tf.keras.layers.Dense(hidden_units[-1], activation=output_activation)

    def call(self, obs, training):
        o = obs
        for layer in self.dense_layers:
            o = layer(o)
        out = self.dense_out(o)
        return out


# policy network

class continuous_policy(tf.keras.Model):
    def __init__(self, hidden_units, act_dim=1, activation='selu', output_activation=None):
        super(continuous_policy, self).__init__()
        self.act_dim = act_dim
        self.mu = mlp(hidden_units + [act_dim], activation=activation, output_activation=output_activation)
        self.logstd = tf.Variable(initial_value=-0.5 * tf.ones((1, self.act_dim), dtype=tf.float32), trainable=True)

    def call(self, obs):
        mu = self.mu(obs)
        std = tf.exp(self.logstd)
        std = tf.tile(std, [tf.shape(mu)[0], 1])
        pi = mu + tf.random.normal(tf.shape(mu)) * std
        logp_pi = gaussian_likelihood_tfd(pi, mu, std)

        return pi, logp_pi, mu, std


# value network

class value(tf.keras.Model):
    def __init__(self, hidden_units, activation='selu', output_activation=None):
        super(value, self).__init__()
        self.val = mlp(hidden_units + [1], activation=activation, output_activation=output_activation)

    def call(self, obs, training):
        v = self.val(obs)
        return v


# actor_critic model

class mlp_actor_critic_continuous(tf.keras.Model):
    def __init__(self, policy_hidden_units, act_dim, value_hidden_units, activation, policy_output_activation,
                 value_output_activation):
        super(mlp_actor_critic_continuous, self).__init__()
        self.policy_model = continuous_policy(policy_hidden_units, act_dim, activation, policy_output_activation)

        self.value_model = value(value_hidden_units, activation, value_output_activation)

    def call(self, obs):
        pi, logp_pi, mu, std = self.policy_model(obs)
        v = self.value_model(obs)

        return pi, logp_pi, mu, std, v


# Trajectory buffer

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class TrajectoryBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # normalize advantage
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf, self.val_buf, self.rew_buf]


class PPOAgent:
    def __init__(self,
                 env_fn,
                 env_inp,
                 num_workers=4,
                 actor_hidden_units=[128, 128],
                 critic_hidden_units=[128, 128],
                 activation='selu',
                 policy_output_activation=None):

        self.env_fn = env_fn
        self.env_inp = env_inp
        self.env = env_fn
        self.act_dim = int(self.env.action_space.shape[0])
        self.obs_dim = int(self.env.observation_space.shape[0])
        self.obs_scaling_constant = self.env.obs_scaling_constant
        self.act_scaling_constant = self.env.action_scaling_constant
        self.min_a = self.env.action_space.low
        self.max_a = self.env.action_space.high
        self.reward_scale = self.env.reward_scale
        self.num_workers = num_workers

        self.model = mlp_actor_critic_continuous(policy_hidden_units=actor_hidden_units,
                                                 act_dim=self.act_dim,
                                                 value_hidden_units=critic_hidden_units,
                                                 activation=activation,
                                                 policy_output_activation=policy_output_activation,
                                                 value_output_activation=None)

    def init_model(self):
        o = self.env.reset()
        _, _, _, _, _ = self.model(o.reshape(1, -1))

    def sample_action(self, obs, samples=1):
        obs = tf.tile(obs, [samples, 1])
        pi, logp_pi, mu, std, v = self.model(obs)
        return pi, logp_pi, v

    def run_trajectory(self):
        try:
            del env
        except:
            pass

        env = self.env_fn

        buf = TrajectoryBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.nsteps, gamma=self.gamma,
                               lam=self.lam)

        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        try:
            obs_scaling_constant = env.obs_scaling_constant
        except:
            obs_scaling_constant = 1.0

        for t in range(self.nsteps):
            o = o / obs_scaling_constant
            a, logp_t, v_t = self.sample_action(o.reshape(1, -1))
            a = a.numpy()
            v_t = v_t.numpy()
            logp_t = logp_t.numpy()
            a_rescaled = a * self.act_scaling_constant
            # a_rescaled = tf.clip_by_value(a_rescaled, clip_value_min=self.min_a, clip_value_max=self.max_a)

            o2, r, d, _ = env.step(a_rescaled.reshape(-1, ))
            r = r / self.reward_scale
            ep_ret += r
            ep_len += 1

            # save and log: scaled observation, unscaled action, reward, value, logprob of suggested action
            buf.store(o, a, r, v_t, logp_t)

            # Update obs
            o = o2

            terminal = d or (ep_len == self.max_episode_len)
            if terminal or (t == self.nsteps):
                if not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)

                # if trajectory didn't reach terminal state, bootstrap value target
                o = o / obs_scaling_constant
                _, _, v_term = self.sample_action(o.reshape(1, -1))
                last_val = 0 if d else v_term
                buf.finish_path(last_val)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # get from buf
        obs_buf, act_buf, adv_buf, ret_buf, logp_buf, val_buf, rew_buf = buf.get()
        return (obs_buf, act_buf, adv_buf, ret_buf, logp_buf, val_buf, rew_buf)

    def train(self,
              gamma=0.99,
              lam=0.97,
              clip_ratio=0.2,
              target_kl=0.01,
              ent_coef=0,
              vf_coef=1.0,
              max_grad_norm=0.1,
              pi_lr=3e-4,
              vf_lr=1e-3,
              train_pi_iters=80,
              train_v_iters=80,
              joint_train=True,
              minibatch_size=512,
              optimizer='Adam',
              max_episode_len=1000,
              nsteps=4000,
              updates=200,
              epochs=5,
              save_every_n_updates=10,
              model_prefix='D:\\DeepRL_SergeyLevine\\TRPO_TF2'):

        # trajectory buffer vars
        self.gamma = gamma
        self.lam = lam
        self.max_episode_len = max_episode_len
        self.nsteps = nsteps
        print("Rewards scaled by: ", self.reward_scale)

        # initialize model
        self.init_model()

        if optimizer == 'Adam':
            pi_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_lr)
            vf_optimizer = tf.keras.optimizers.Adam(learning_rate=vf_lr)
        else:
            print("other optimizers ...")

        # chec init params:
        # print("init params: ", self.model.policy_model.trainable_variables)

        for update in range(updates):
            print("Update: {}".format(update))

            # Gather state-action pairs from multiple trajectories
            traj = Parallel(n_jobs=self.num_workers)(delayed(self.run_trajectory)() for _ in range(self.num_workers))

            obs_buf_list = [tup[0] for tup in traj]
            act_buf_list = [tup[1] for tup in traj]
            adv_buf_list = [tup[2] for tup in traj]
            ret_buf_list = [tup[3] for tup in traj]
            logp_buf_list = [tup[4] for tup in traj]
            val_buf_list = [tup[5] for tup in traj]
            reward_buf_list = [tup[6] for tup in traj]

            obs = np.vstack(obs_buf_list)
            act = np.vstack(act_buf_list)
            adv = np.concatenate(adv_buf_list, axis=0)
            ret = np.concatenate(ret_buf_list, axis=0)
            logp = np.concatenate(logp_buf_list, axis=0)
            val = np.concatenate(val_buf_list, axis=0)
            rew = np.concatenate(reward_buf_list, axis=0)

            # total batches for training per epoch
            num_batches = max(int(math.ceil(obs.shape[0] / minibatch_size)), 1)
            replace = True if obs.shape[0] < minibatch_size else False

            # mean episode reward
            mean_ep_reward = (np.sum(rew) / self.nsteps) * self.max_episode_len

            print("Mean Episode Reward: ", mean_ep_reward)

            for epoch in range(epochs):

                for i in range(num_batches):

                    batch_indices = np.random.choice(obs.shape[0], minibatch_size, replace=replace)
                    obs_batch = obs[batch_indices]
                    pi_batch = act[batch_indices]
                    logp_batch = logp[batch_indices]
                    adv_batch = adv[batch_indices]
                    ret_batch = ret[batch_indices]
                    v_batch = val[batch_indices]

                    with tf.GradientTape(persistent=True) as tape:
                        _, _, mu, std, v = self.model(obs_batch)

                        # log prob of old act under current policy
                        logp_pi = gaussian_likelihood_tfd(pi_batch, mu, std)

                        # importance ratio -- pi(a,new)/pi(a,old)
                        ratio = tf.exp(logp_pi - logp_batch)

                        # Policy loss
                        pg_losses1 = adv_batch * ratio
                        pg_losses2 = adv_batch * tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

                        # approx kl div
                        # approx_kl = tf.math.reduce_mean(logp_batch - logp_pi)
                        approx_kl = 0.5 * tf.math.reduce_mean(tf.math.square(logp_batch - logp_pi))

                        # entropy
                        approx_ent = tf.math.reduce_mean(-logp_pi)

                        # policy loss
                        pg_loss = tf.math.reduce_mean(-tf.math.minimum(pg_losses1, pg_losses2))

                        # value fn loss
                        vf_clipped = v_batch + tf.clip_by_value(v - v_batch, -clip_ratio, clip_ratio)
                        vf_losses1 = tf.math.square(v - ret_batch)
                        vf_losses2 = tf.math.square(vf_clipped - ret_batch)
                        vf_loss = .5 * tf.math.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                        # Total loss
                        loss = pg_loss - approx_ent * ent_coef + vf_loss * vf_coef

                    if joint_train:
                        grads = tape.gradient(loss, self.model.trainable_variables)

                        # check for too much policy divergence
                        if np.abs(approx_kl) > target_kl:
                            break
                            '''
                            if max_grad_norm is not None:
                                grads, _ = tf.clip_by_global_norm(grads, clip_norm = max_grad_norm)

                                params_backup = self.model.get_weights()
                                pi_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                                # re-check kl
                                _, _, n_mu, n_std, n_v = self.model(obs_batch)

                                # log prob of old act under current policy
                                n_logp_pi = gaussian_likelihood_tfd(pi_batch, n_mu, n_std)
                                n_approx_kl = tf.math.reduce_mean(logp_batch - n_logp_pi)
                                n_approx_ent = tf.math.reduce_mean(-n_logp_pi)

                                if np.abs(n_approx_kl) > 1.5*target_kl:
                                    print("High KL Div. after grad clipping. Reversing grad apply ...")
                                    self.model.set_weights(params_backup)
                                    break
                            '''
                        else:
                            pi_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                        print(
                            "  Epoch: {}, iteration: {}, kl_div: {}, entropy: {}, loss: {}".format(epoch, i, approx_kl,
                                                                                                   approx_ent, loss))

                    else:
                        policy_grads = tape.gradient(pg_loss, self.model.policy_model.trainable_variables)
                        vf_grads = tape.gradient(vf_loss, self.model.value_model.trainable_variables)

                        # clip grad
                        policy_grads, _ = tf.clip_by_global_norm(policy_grads, clip_norm=max_grad_norm)
                        vf_grads, _ = tf.clip_by_global_norm(vf_grads, clip_norm=max_grad_norm)

                        # check for too much policy divergence
                        if np.abs(approx_kl) > target_kl:
                            break
                            '''
                            if max_grad_norm is not None:
                                policy_grads, _ = tf.clip_by_global_norm(policy_grads, clip_norm = max_grad_norm)

                                params_backup = self.model.policy_model.get_weights()
                                pi_optimizer.apply_gradients(zip(policy_grads, self.model.policy_model.trainable_variables))

                                # re-check kl
                                _, _, n_mu, n_std, n_v = self.model(obs_batch)  
                                n_logp_pi = gaussian_likelihood_tfd(pi_batch, n_mu, n_std)
                                n_approx_kl = tf.math.reduce_mean(logp_batch - n_logp_pi)
                                n_approx_ent = tf.math.reduce_mean(-n_logp_pi)

                                if np.abs(n_approx_kl) > 1.5*target_kl:
                                    print("High KL Div. after grad clipping. Reversing grad apply ...")
                                    self.model.policy_model.set_weights(params_backup)
                                    break
                            '''
                        else:
                            pi_optimizer.apply_gradients(zip(policy_grads, self.model.policy_model.trainable_variables))
                        vf_optimizer.apply_gradients(zip(vf_grads, self.model.value_model.trainable_variables))
                        print("  Epoch: {}, iteration: {}, kl_div: {}, entropy: {}, pg_loss: {}, vf_loss: {}".format(
                            epoch, i, approx_kl, approx_ent, pg_loss, vf_loss))

            if update % save_every_n_updates == 0:
                model_path = model_prefix + '_' + str(update)
                tf.keras.models.save_model(self.model, model_path)

    def restore(self, model_path):
        self.init_model()
        self.model = tf.keras.models.load_model(model_path)

    def output(self, model_path):
        # restore best weights
        self.restore(model_path)

        # reset env
        state = self.env.reset()
        total_reward = 0
        for t in range(self.env.planning_periods):
            state = state / self.obs_scaling_constant
            action, _, _ = self.sample_action(state.reshape(1, -1))
            action = action.numpy() * self.act_scaling_constant
            next_state, reward, done, _ = self.env.step(action.numpy().reshape(-1, ))
            state = next_state
            total_reward += reward

        # distribution plan
        out = self.env.distribution_plan()
        statedump = self.env.statedump()
        costdump = self.env.costdump()

        return (out, total_reward, statedump, costdump)
