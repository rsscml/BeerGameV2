from flask import Flask, render_template, redirect, url_for, flash, abort, jsonify, send_file, session
from flask_session import Session
#from werkzeug.utils import secure_filename
from forms import UserActionForm, GameInitForm, HomeForm
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import tensorflow as tf
#from flask_executor import Executor
from flask_bootstrap import Bootstrap
import copy
import orgym
import rl_agent_ppo_separate
import rl_agent_ppo_separate_actscaled
import rl_agent_ppo_unified
import orgym_timeaware
import rl_agent_td3_timeaware
import pickle
import hashlib

app = Flask(__name__)
app.config['SECRET_KEY'] = "beergamev2"
app.config['PROJECT_ROOT'] = "projects/"
app.config['LOG_DIRECTORY'] = "log/"
app.config['MODEL_DIRECTORY'] = "models/"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["CACHE_TYPE"] = "null"
Bootstrap(app)
Session(app)

session_logfile = app.config['LOG_DIRECTORY'] + "session.log"
rl_ppo_model_path = app.config['MODEL_DIRECTORY'] + "orgym_ppo_agent_separate_softplus_nonepisodic_800"
rl_td3_model_path = app.config['MODEL_DIRECTORY'] + "orgym_td3_per_timeaware_415000"
rl_agent = None
algorithm = 'TD3'

def load_model():
    global rl_agent
    global algorithm
    if algorithm == 'PPO':
        rl_agent = rl_agent_ppo_separate_actscaled.PPOAgent(env_fn = orgym.InvManagementLostSalesEnv(mean_demand=20),
                                                        env_inp = None,
                                                        num_workers = 1,
                                                        actor_hidden_units = [256,256],
                                                        critic_hidden_units = [256,256],
                                                        activation = 'elu',
                                                        policy_output_activation = 'softplus')

        rl_agent.restore(rl_ppo_model_path)
    else:
        rl_agent = rl_agent_td3_timeaware.TD3Agent(env_fn = orgym_timeaware.InvManagementLostSalesEnv(mean_demand=20),
                                                 max_ep_len = 30,
                                                 actor_hidden_units=[256,256,256],
                                                 critic_hidden_units=[256,256,256],
                                                 activation='selu',
                                                 policy_output_activation='sigmoid',
                                                 q_output_activation=None,
                                                 pi_lr=0.0001,
                                                 q_lr=0.0001,
                                                 gradient_descents_per_update=100,
                                                 update_every = 250,
                                                 start_timesteps=2000,
                                                 total_timesteps=500000,
                                                 replay_buffer_size=1000000,
                                                 alpha = 0.7,
                                                 beta = 0.5,
                                                 polyak=0.995,
                                                 discount=0.99,
                                                 expl_noise=0.1,
                                                 policy_noise=0.1,
                                                 noise_clip=0.5,
                                                 noise_decay_rate=1.0,
                                                 noise_decay_steps=5000,
                                                 policy_freq=2,
                                                 eval_freq=1000,
                                                 save_freq=5000,
                                                 num_eval_episodes=10,
                                                 train_batch_size=512,
                                                 model_prefix='/dbfs/rahul/all/model/orgym_td3_per_timeaware')

        rl_agent.restore(rl_td3_model_path)


def dfo_func(policy, env, *args):
    '''
    Runs an episode based on current base-stock model
    settings. This allows us to use our environment for the
    DFO optimizer.
    '''
    env.reset()  # Ensure env is fresh
    rewards = []
    done = False
    for i in range(env.num_periods):
        # non-timeaware
        #action = env.base_stock_action(policy)
        # timeaware
        action = env.base_stock_action(policy)
        action = action.reshape(1,-1)
        action = np.tile(action, (env.num_periods,1))
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break

    """
    while not done:
        action = env.base_stock_action(policy)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break
    """

    rewards = np.array(rewards)
    prob = env.demand_dist.pmf(env.D, **env.dist_param)

    # Return negative of expected profit
    return -1 / env.num_periods * np.sum(prob * rewards)


def optimize_inventory_policy(env, fun, init_policy=None, method='Powell'):
    if init_policy is None:
        init_policy = np.ones(env.num_stages - 1)

    # Optimize policy
    min_inv = 0
    max_inv = env.demand_dist.mean(**env.dist_param)*env.num_periods
    out = minimize(fun=fun, x0=init_policy, args=env, method=method, bounds=[(min_inv,max_inv),(min_inv,max_inv),(min_inv,max_inv)])
    policy = out.x.copy()

    # Policy must be positive integer
    policy = np.round(np.maximum(policy, 0), 0).astype(int)

    return policy, out

@app.route("/")
def home():
    # displays game info, objective & parameters
    # redirects to init_game
    session.clear()
    form = HomeForm()
    if form.validate_on_submit():
        return redirect(url_for("init_game"))
    return render_template("home.html")

@app.route("/init_game", methods=['GET','POST'])
def init_game():
    load_model() # RL Trained Agent
    form = GameInitForm()
    if form.validate_on_submit():
        game_id = form.GameName.data
        # store game_id as session object
        session["game_id"] = game_id
        PlanningPeriods = form.PlanningPeriods.data
        session["max_planning_periods"] = PlanningPeriods
        session["curr_planning_period"] = 0
        mean_demand_param = form.MeanCustomerDemand.data
        user_env = orgym_timeaware.InvManagementLostSalesEnv(mean_demand=mean_demand_param)
        agent_env = orgym_timeaware.InvManagementLostSalesEnv(mean_demand=mean_demand_param)
        dfo_env = orgym_timeaware.InvManagementLostSalesEnv(mean_demand=mean_demand_param)
        dfo_policy, _ = optimize_inventory_policy(dfo_env, dfo_func)
        # pickle these -- to be restored in concerned requests
        user_env_pickle = app.config['PROJECT_ROOT'] + str(session["game_id"]) +  "_user_env.pkl"
        agent_env_pickle = app.config['PROJECT_ROOT'] + str(session["game_id"]) + "_agent_env.pkl"
        dfo_env_pickle = app.config['PROJECT_ROOT'] + str(session["game_id"]) + "_dfo_env.pkl"
        dfo_policy_pickle = app.config['PROJECT_ROOT'] + str(session["game_id"]) + "_dfo_policy.pkl"

        with open(user_env_pickle, "wb") as f:
            pickle.dump(user_env, f, -1)
        with open(agent_env_pickle, "wb") as f:
            pickle.dump(agent_env, f, -1)
        with open(dfo_env_pickle, "wb") as f:
            pickle.dump(dfo_env, f, -1)
        with open(dfo_policy_pickle, "wb") as f:
            pickle.dump(dfo_policy, f)

        session["user_env"] = user_env_pickle
        session["agent_env"] = agent_env_pickle
        session["dfo_env"] = dfo_env_pickle
        session["dfo_policy"] = dfo_policy_pickle
        return redirect(url_for("start_game"))
    return render_template("initialize_game.html", form=form)


@app.route("/start_game", methods=['GET','POST'])
def start_game():
    # restore env objects
    with open(session.get("user_env"), 'rb') as f:
        user_env = pickle.load(f)
    with open(session.get("agent_env"), 'rb') as f:
        agent_env = pickle.load(f)
    with open(session.get("dfo_env"), 'rb') as f:
        dfo_env = pickle.load(f)
    with open(session.get("dfo_policy"), 'rb') as f:
        dfo_policy = pickle.load(f)

    state_dict = {}
    if session["curr_planning_period"] == 0:
        #random_seed = np.random.randint(100)
        random_seed = int(hashlib.md5(session["game_id"].encode('utf-8')).hexdigest(), base=16)%10000
        print("random seed: ", random_seed)
        user_env.seed(random_seed)
        agent_env.seed(random_seed)
        dfo_env.seed(random_seed)
        user_init_state = user_env.reset()
        agent_init_state = agent_env.reset()
        dfo_init_state = dfo_env.reset()

    state_dict["Week no."] = user_env.period + 1
    state_dict["Retailer Inventory"] = user_env.I[user_env.period, 0]
    state_dict["Wholesaler Inventory"] = user_env.I[user_env.period, 1]
    state_dict["Distributor Inventory"] = user_env.I[user_env.period, 2]
    state_dict["Last Period's Demand"] = user_env.D[max(user_env.period-1,0)]
    state_dict["Current & Future Demand Estimate"] = user_env.demand_forecast[user_env.period:].tolist()
    state_dict["Inbound Inventory (Retailer)"] = user_env.T[user_env.period, 0]
    state_dict["Inbound Inventory (Wholesaler)"] = user_env.T[user_env.period, 1]
    state_dict["Inbound Inventory (Distributor)"] = user_env.T[user_env.period, 2]
    state_dict["Last Period's Profit"] = user_env.P[max(user_env.period-1,0)]
    print(" user init state: ", user_env.state)
    print(" agent init state: ", agent_env.state)
    print(" dfo init state: ", dfo_env.state)

    form = UserActionForm()
    if form.validate_on_submit():
        # get action for period
        RetailerOrderQty = form.RetailerOrderQty.data
        WholesalerOrderQty = form.WholesalerOrderQty.data
        DistributorOrderQty = form.DistributorOrderQty.data

        # PPO
        #user_action = np.array([RetailerOrderQty, WholesalerOrderQty, DistributorOrderQty]).reshape(-1,)
        # execute User action in env in loop
        #user_next_state, user_reward, user_done, _ = user_env.step(user_action)
        # TD3
        user_action = np.array([RetailerOrderQty, WholesalerOrderQty, DistributorOrderQty]).reshape(1,-1)
        user_action = np.tile(user_action, (user_env.num_periods, 1))
        user_next_state, user_reward, user_done, _ = user_env.step(user_action)

        print("\n next_state user:", user_env.state)
        print("\n demand user:", user_env.D[max(user_env.period-1,0)])
        print("\n action user:", user_action)
        print("\n reward user:", user_reward)

        # execute Agent action in env
        global rl_agent
        # PPO
        #agent_action, _, _ = rl_agent.sample_action(agent_env.state.reshape(1, -1)/agent_env.obs_scaling_constant, samples=100)
        #agent_action = tf.reduce_mean(agent_action, axis=0, keepdims=True)
        #agent_action = np.ceil(agent_action.numpy() * agent_env.action_scaling_constant)
        #agent_action = np.clip(agent_action.reshape(-1, ), agent_env.action_space.low, agent_env.action_space.high)
        #agent_next_state, agent_reward, agent_done, _ = agent_env.step(agent_action)
        # TD3
        agent_action = rl_agent.main_model.get_action(agent_env.state.reshape(1, -1)/agent_env.obs_scaling_constant).numpy().reshape(agent_env.action_space.shape)
        agent_action = np.around(agent_action * agent_env.action_scaling_constant, 0)
        agent_next_state, agent_reward, agent_done, _ = agent_env.step(agent_action)

        print("\n next_state agent:", agent_next_state)
        print("\n demand agent:", agent_env.D[max(agent_env.period-1,0)])
        print("\n action agent:", agent_action)
        print("\n reward agent:", agent_reward)

        # execute base_stock_policy
        action = dfo_env.base_stock_action(dfo_policy)
        action = action.reshape(1, -1)
        action = np.tile(action, (dfo_env.num_periods, 1))
        _, _, _, _ = dfo_env.step(action)

        # increment period counter
        session["curr_planning_period"] += 1

        # save objects
        with open(session["user_env"], "wb") as f:
            pickle.dump(user_env, f, -1)
        with open(session["agent_env"], "wb") as f:
            pickle.dump(agent_env, f, -1)
        with open(session["dfo_env"], "wb") as f:
            pickle.dump(dfo_env, f, -1)

        if session["curr_planning_period"] == session["max_planning_periods"]:
            return redirect(url_for("game_log"))
        else:
            return redirect(url_for("start_game"))

    return render_template("game_play.html", form=form, state=state_dict)


@app.route("/game_log", methods=['GET','POST'])
def game_log():
    # restore env objects
    with open(session.get("user_env"), 'rb') as f:
        user_env = pickle.load(f)
    with open(session.get("agent_env"), 'rb') as f:
        agent_env = pickle.load(f)
    with open(session.get("dfo_env"), 'rb') as f:
        dfo_env = pickle.load(f)

    total_user_reward = round(np.sum(user_env.P),0)
    total_agent_reward = round(np.sum(agent_env.P),0)
    total_dfo_reward = round(np.sum(dfo_env.P),0)

    user_total_sales = user_env.S
    agent_total_sales = agent_env.S
    dfo_total_sales = dfo_env.S

    user_total_demand = user_env.D
    agent_total_demand = agent_env.D
    dfo_total_demand = dfo_env.D

    user_action_log =  user_env.R
    agent_action_log = agent_env.R
    dfo_action_log = dfo_env.R

    user_log = np.concatenate((user_total_demand.reshape(-1,1), user_total_sales[:,0:1], user_action_log), axis=1)
    agent_log = np.concatenate((agent_total_demand.reshape(-1, 1), agent_total_sales[:, 0:1], agent_action_log), axis=1)
    dfo_log = np.concatenate((dfo_total_demand.reshape(-1, 1), dfo_total_sales[:, 0:1], dfo_action_log), axis=1)

    df_user = pd.DataFrame(data=user_log, columns=['Demand','Sales','Retailer Order','Wholesaler Order','Distributor Order'])
    df_user.index.name = "Period"
    df_agent = pd.DataFrame(data=agent_log, columns=['Demand','Sales','Retailer Order', 'Wholesaler Order', 'Distributor Order'])
    df_agent.index.name = "Period"
    df_dfo = pd.DataFrame(data=dfo_log, columns=['Demand','Sales','Retailer Order', 'Wholesaler Order', 'Distributor Order'])
    df_dfo.index.name = "Period"

    return render_template("game_log.html",
                           total_user_reward=total_user_reward,
                           total_agent_reward=total_agent_reward,
                           total_dfo_reward=total_dfo_reward,
                           user_actions=[df_user.to_html(classes='data', header="true")],
                           agent_actions=[df_agent.to_html(classes='data', header="true")],
                           dfo_actions=[df_dfo.to_html(classes='data', header="true")])


if __name__ == "__main__":
    app.run(use_reloader=False, debug=True, host='0.0.0.0', port=8080)
