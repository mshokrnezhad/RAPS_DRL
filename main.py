from Environment import Environment
from Functions import parse_state, plot_learning_curve
from Agent import Agent
import numpy as np
from sklearn.preprocessing import MinMaxScaler

NUM_NODES = 9
NUM_REQUESTS = 5
NUM_SERVICES = 3

env_obj = Environment(NUM_NODES, NUM_REQUESTS, NUM_SERVICES)
best_score = -np.inf
load_checkpoint = False
n_games = 50


'''
state = env_obj.get_state()
print(state)
scaler = MinMaxScaler(feature_range=(-1, 1))
state2 = scaler.fit_transform(state.reshape(1, -1))
print(state2.shape)
'''





















agent = Agent(GAMMA=0.99, EPSILON=0.1, LR=0.0001, NUM_ACTIONS=(NUM_NODES-len(env_obj.net_obj.get_first_tier_nodes())),
              INPUT_SHAPE=env_obj.get_state().size, MEMORY_SIZE=50000, BATCH_SIZE=32, EPSILON_MIN=0.1,
              EPSILON_DEC=1e-3, REPLACE_COUNTER=10000, NAME=str(NUM_NODES)+str(NUM_REQUESTS)+str(NUM_SERVICES),
              CHECKPOINT_DIR='models/')

if load_checkpoint:
    agent.load_models()

file_name = "_lr" + str(agent.LR)+"_"+str(n_games)+"games"
figure_file = 'plots/' + file_name + '.png'

n_steps = 0
scores, eps_history, steps_array = [], [], []

for i in range(n_games):
    done = False
    SEED = np.random.randint(1, 1000)
    env_obj.reset(SEED)
    state = env_obj.get_state()  # parse_state(state, NUM_NODES, NUM_REQUESTS, env_obj)

    score = 0

    while not done:
        selected_request = env_obj.req_obj.REQUESTS.min()
        raw_action = agent.choose_action(state)
        # should be replaced by agent's action selection function
        # assigned_node = np.random.choice(np.setdiff1d(env_obj.net_obj.NODES, env_obj.net_obj.get_first_tier_nodes()))
        action = {"req_id": selected_request, "node_id": raw_action + len(env_obj.net_obj.get_first_tier_nodes())}

        resulted_state, reward, done, info = env_obj.step(action)
        # print(f"req: {action['req_id']}, assigned node: {action['node_id']}, reward: {reward}, status: {not done}")
        score += reward

        if not load_checkpoint:
            agent.store_transition(state, raw_action, reward, resulted_state, int(done))
            agent.learn()

        state = resulted_state
        n_steps += 1

    scores.append(score)
    steps_array.append(n_steps)

    avg_score = np.mean(scores[-100:])
    print('episode:', i, 'score:', score, 'average score: %.1f, best_score: %.1f, eps: %.4f'
          % (avg_score, best_score, agent.EPSILON), 'steps', n_steps)

    if avg_score > best_score:
        if not load_checkpoint:
            agent.save_models()
        best_score = avg_score

    eps_history.append(agent.EPSILON)

x = [i+1 for i in range(len(scores))]
plot_learning_curve(steps_array, scores, eps_history)
