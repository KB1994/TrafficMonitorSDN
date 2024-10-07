import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
import math
from tensorflow.keras import layers, models
from collections import Counter
import datetime
from collections import deque
import matplotlib as mpl



################################# Constants ######################################################

current_time = datetime.datetime.now()

time_str = current_time.strftime("%Y%m%d_%H%M%S")

s_size = 4
a_size = 4
total_episodes = 600
max_env_steps = 6
gamma = 0.95
epsilon = 0.1  # Probability of selecting a random action (exploration)
max_epsilon = 1.0  # Initial epsilon value
min_epsilon = 0.01  # Minimum epsilon value
decay_rate = 0.005  # Rate at which epsilon decays

time_history = []
packet_loss_history_c_dualdqn = []
packet_loss_history_d_dualdqn = []
th_history_c_dualdqn = []
th_history_d_dualdqn = []
d_history_c_dualdqn = []
d_history_d_dualdqn = []
bw_history_c_dualdqn = []
bw_history_d_dualdqn = []


packet_loss_history_c_a2c = []
packet_loss_history_d_a2c = []
th_history_c_a2c = []
th_history_d_a2c = []
d_history_c_a2c = []
d_history_d_a2c = []
bw_history_c_a2c = []
bw_history_d_a2c = []

packet_loss_history_c_rainbowdqn = []
packet_loss_history_d_rainbowdqn = []
th_history_c_rainbowdqn = []
th_history_d_rainbowdqn = []
d_history_c_rainbowdqn = []
d_history_d_rainbowdqn = []
bw_history_c_rainbowdqn = []
bw_history_d_rainbowdqn = []

packet_loss_history_c_ddpg = []
packet_loss_history_d_ddpg = []
th_history_c_ddpg = []
th_history_d_ddpg = []
d_history_c_ddpg = []
d_history_d_ddpg = []
bw_history_c_ddpg = []
bw_history_d_ddpg = []

packet_loss_history_d_ql = []
packet_loss_history_c_ql = []
th_history_d_ql = []
th_history_c_ql = []
d_history_d_ql = []
d_history_c_ql = []
bw_history_d_ql = []
bw_history_c_ql = []
episode_history = []
rw_history_dualDQN = []
rw_history_RainbowDQN = []
rw_history_DDPG = []
rw_history_A2C = []

################################ Define Environment #################################

class Environment:
    def __init__(self):
        self.num_states = 4
        self.num_actions = 2
        self.state = 0
        self.BANDWIDTH_MAX = 100  # Bandwidth in Mbits per second
        self.BANDWIDTH_MIN = 50  # Bandwidth in Mbits per second
        self.queue_length = 10   # Length of the queue in packets
        self.service_rate = 100  # Router's service rate in packets per second
        self.processing_time_per_packet = 0.001  # Processing time per packet in seconds
        self.actions = ['increase_bandwidth', 'decrease_bandwidth']

    def current_state(self, PACKET_SIZE_c, PACKET_SIZE_d, TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d, Bw_c, Bw_d, packets_received_d, packets_received_c):
        loss_c = (TOTAL_PACKETS_SENT_c - packets_received_c) / TOTAL_PACKETS_SENT_c
        loss_d = (TOTAL_PACKETS_SENT_d - packets_received_d) / TOTAL_PACKETS_SENT_d
        delay_c = (PACKET_SIZE_c / (Bw_c * 10**6)) + (self.queue_length / self.service_rate) + self.processing_time_per_packet
        delay_d = (PACKET_SIZE_d / (Bw_d * 10**6)) + (self.queue_length / self.service_rate) + self.processing_time_per_packet
        thrghpt_c = ((packets_received_c * PACKET_SIZE_c * 8) / delay_c)/1000000
        thrghpt_d = ((packets_received_d * PACKET_SIZE_d * 8)/ delay_d)/1000000
        return [delay_c, loss_c, delay_d, loss_d], thrghpt_c, thrghpt_d

    def take_action(self, action_n, current_bandwidth):
        selected_action = self.actions[action_n % 2]
        if selected_action == 'increase_bandwidth':
            Bw = self.increase_bandwidth(current_bandwidth, self.BANDWIDTH_MAX)
        elif selected_action == 'decrease_bandwidth':
            Bw = self.decrease_bandwidth(current_bandwidth, self.BANDWIDTH_MIN)
        return Bw, selected_action

    @staticmethod
    def increase_bandwidth(current_bandwidth, BANDWIDTH_MAX):
        if float(current_bandwidth) < BANDWIDTH_MAX:
            current_bandwidth += 10  # Increase bandwidth by 10 Mbps
        return float(current_bandwidth)

    @staticmethod
    def decrease_bandwidth(current_bandwidth, BANDWIDTH_MIN):
        if float(current_bandwidth) > BANDWIDTH_MIN:
            current_bandwidth -= 10  # Decrease bandwidth by 10 Mbps
        return float(current_bandwidth)

############ Dualing DQN Agent ####################################################
    
agent1_dualdqn = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(1 + 4)  # Output layer for Dueling DQN
])

class DuelingDQNOutput(keras.layers.Layer):
    def call(self, inputs):
        value = inputs[:, :1]  # Value stream
        advantages = inputs[:, 1:]  # Advantage stream
        q_values = value + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))
        return q_values

agent1_dualdqn.add(DuelingDQNOutput())

agent1_dualdqn.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss='mse')

agent2_dualdqn = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(1 + 4)  # Output layer for Dueling DQN
])

agent2_dualdqn.add(DuelingDQNOutput())

agent2_dualdqn.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss='mse')









#################################### DDGP #######################################

actor_1_ddpg = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(1, activation='tanh')  # Assuming a single continuous action, adjust as needed
])

actor_1_ddpg.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss='mse')

# Critic Network for DDPG
# This network evaluates a state-action pair and outputs the Q-value
critic_1_ddpg = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4 + 1,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # State and action concatenated
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(1)  # Output is a single Q-value
])

critic_1_ddpg.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss='mse')

# You can do the same for the second agent (agent2_ddpg)
actor_2_ddpg = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(1, activation='tanh')  # Assuming a single continuous action
])

actor_2_ddpg.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss='mse')

critic_2_ddpg = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4 + 1,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # State and action concatenated
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(1)  # Output is a single Q-value
])

critic_2_ddpg.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss='mse')






############ RainbowDQN Model #####################################################

class NoisyDense(layers.Layer):
    def __init__(self, units):
        super(NoisyDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        # Initialize learnable parameters for weights and biases
        self.w_mu = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.w_sigma = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b_mu = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)
        self.b_sigma = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inputs):
        # Generate noise for weights and biases
        noise_w = tf.random.normal(shape=self.w_mu.shape)
        noise_b = tf.random.normal(shape=self.b_mu.shape)

        # Compute the noisy weights and biases
        w = self.w_mu + self.w_sigma * noise_w
        b = self.b_mu + self.b_sigma * noise_b

        return tf.matmul(inputs, w) + b



agent1_rainbow = models.Sequential([
    NoisyDense(64),  # No input_shape needed here
    tf.keras.layers.ReLU(),
    NoisyDense(64),
    tf.keras.layers.ReLU(),
    NoisyDense(128),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(1 + 4)  # Output layer for Dueling DQN
])

# Dueling DQN output processing
class DuelingDQNOutput(tf.keras.layers.Layer):
    def call(self, inputs):
        value = inputs[:, :1]  # Value stream
        advantages = inputs[:, 1:]  # Advantage stream
        q_values = value + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))
        return q_values

agent1_rainbow.add(DuelingDQNOutput())

# Compile the model
agent1_rainbow.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='mse')


agent2_rainbow = models.Sequential([
    NoisyDense(64),  # No input_shape needed here
    tf.keras.layers.ReLU(),
    NoisyDense(64),
    tf.keras.layers.ReLU(),
    NoisyDense(128),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(1 + 4)  # Output layer for Dueling DQN
])



agent2_rainbow.add(DuelingDQNOutput())

# Compile the Rainbow DQN with RMSprop optimizer
agent2_rainbow.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='mse')



############ LSTM Model #####################################################

shared_layers_1 = models.Sequential([
    layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
])

# Actor Network (outputs actions probabilities for a discrete action space)
actor_1 = models.Sequential([
    shared_layers_1,
    layers.Dense(4, activation='softmax')  # Assuming 4 discrete actions
])

# Critic Network (outputs the value of the state)
critic_1 = models.Sequential([
    shared_layers_1,
    layers.Dense(1)  # Single output for state-value estimation
])

# Compile Actor and Critic networks separately
actor_1.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='categorical_crossentropy')
critic_1.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='mse')


shared_layers_2 = models.Sequential([
    layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
])

# Actor Network (outputs actions probabilities for a discrete action space)
actor_2 = models.Sequential([
    shared_layers_2,
    layers.Dense(4, activation='softmax')  # Assuming 4 discrete actions
])

# Critic Network (outputs the value of the state)
critic_2 = models.Sequential([
    shared_layers_2,
    layers.Dense(1)  # Single output for state-value estimation
])

# Compile Actor and Critic networks separately
actor_2.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='categorical_crossentropy')
critic_2.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='mse')





################################## Train RainbowDQN #################################

print('Training RainbowDQN')


scaler = StandardScaler()
env = Environment()

# Assuming the Rainbow DQN model (agent1_rainbow) is already defined as in previous answers
replay_buffer = deque(maxlen=2000)  # Prioritized Experience Replay buffer
n_step_buffer = deque(maxlen=3)  # For N-step returns
batch_size = 64
reward = 0


def get_prioritized_experience_index():
    # Simplified: set all probabilities based on the size of the replay buffer
    probabilities = np.ones(len(replay_buffer))  # Uniform probabilities for now

    # Normalize the probabilities to ensure they sum to 1
    probabilities /= np.sum(probabilities)

    # Sample an index based on these normalized probabilities
    return np.random.choice(len(replay_buffer), p=probabilities)


for e in range(total_episodes):
    state = np.reshape(np.zeros(4), [1, s_size])
    done = False
    max_en_step = 0
    action1 = action2 = 0
    num_decrease_bandwidth_c = num_decrease_bandwidth_d = 0
    num_increase_bandwidth_c = num_increase_bandwidth_d = 0

    while not done:
        TOTAL_PACKETS_SENT_d = random.uniform(1000, 5000)
        TOTAL_PACKETS_SENT_c = random.uniform(10, 30)
        packet_loss_percentage = random.uniform(0, 10)
        PACKET_SIZE_d = 1518
        PACKET_SIZE_c = 300
        packets_received_d = TOTAL_PACKETS_SENT_d * (1 - packet_loss_percentage / 100)
        packets_received_c = TOTAL_PACKETS_SENT_c * (1 - packet_loss_percentage / 100)
        Bw_c = Bw_d = random.uniform(50, 100)

        state = np.array(env.current_state(PACKET_SIZE_c, PACKET_SIZE_d, TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d, Bw_c, Bw_d, packets_received_d, packets_received_c)[0]).reshape(1, s_size)
        scaled_state = scaler.fit_transform(state)

        # Noisy networks eliminate epsilon-greedy exploration
        action1 = np.argmax(agent1_rainbow.predict(scaled_state)[0])
        action2 = np.argmax(agent2_rainbow.predict(scaled_state)[0])

        Bw_c, selected_action_c = env.take_action(action1, Bw_c)
        Bw_d, selected_action_d = env.take_action(action2, Bw_d)

        # Track bandwidth changes for both channels
        if selected_action_c == 'decrease_bandwidth':
            num_decrease_bandwidth_c += 1
        else:
            num_increase_bandwidth_c += 1

        if selected_action_d == 'decrease_bandwidth':
            num_decrease_bandwidth_d += 1
        else:
            num_increase_bandwidth_d += 1

        next_state, thrghpt_1, thrghpt_2 = env.current_state(PACKET_SIZE_c, PACKET_SIZE_d, TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d, Bw_c, Bw_d, packets_received_d, packets_received_c)

        next_state = np.reshape(next_state, [1, s_size])
        scaled_next_state = scaler.transform(next_state)
        done = (next_state[0][1] < 0.03 and next_state[0][3] < 0.03 and next_state[0][0] < 0.3 and next_state[0][2] < 0.3)
        max_en_step += 1

    reward += (0.5 * (0.75 * num_decrease_bandwidth_c + 0.25 * num_increase_bandwidth_c)) + (0.5 * (0.75 * num_decrease_bandwidth_d + 0.25 * num_increase_bandwidth_d))
    
    exp_rewardRN = 1 - math.exp(-decay_rate * reward)

    # Add experience to the buffer for Prioritized Experience Replay
    replay_buffer.append((state, action1, action2, exp_rewardRN, next_state, done))

    # Implement N-step returns by storing intermediate transitions
    n_step_buffer.append((state, action1, action2, exp_rewardRN, next_state, done))
    if len(n_step_buffer) >= 3:
        n_step_experience = n_step_buffer.popleft()
        n_step_reward = sum([exp[3] * (gamma ** i) for i, exp in enumerate(n_step_buffer)])  # Sum of n-step rewards

    # Sample experience from prioritized buffer
    if len(replay_buffer) >= batch_size:
        idx = get_prioritized_experience_index()
        state_b, action1_b, action2_b, reward_b, next_state_b, done_b = replay_buffer[idx]

        # Update the Q-values with the multi-step returns and the distributional approach
        target1 = reward_b + gamma * np.max(agent1_rainbow.predict(next_state_b)[0])
        target2 = reward_b + gamma * np.max(agent2_rainbow.predict(next_state_b)[0])

        target_f1 = agent1_rainbow.predict(scaled_state)
        target_f2 = agent2_rainbow.predict(scaled_state)

        target_f1[0][action1_b] = target1
        target_f2[0][action2_b] = target2

        agent1_rainbow.fit(scaled_state, target_f1, epochs=1, batch_size=64, verbose=0)
        agent2_rainbow.fit(scaled_state, target_f2, epochs=1, batch_size=64, verbose=0)

    state = next_state
    max_en_step += 1

    # Add the episode results to the history logs
    episode_history.append(e)
    rw_history_RainbowDQN.append(exp_rewardRN)
    packet_loss_history_d_rainbowdqn.append(next_state[0][3])
    packet_loss_history_c_rainbowdqn.append(next_state[0][1])
    d_history_d_rainbowdqn.append(next_state[0][2])
    d_history_c_rainbowdqn.append(next_state[0][0])
    th_history_d_rainbowdqn.append(thrghpt_1)
    th_history_c_rainbowdqn.append(thrghpt_2)
    bw_history_d_rainbowdqn.append(Bw_d)
    bw_history_c_rainbowdqn.append(Bw_c)

print(rw_history_RainbowDQN)



#####################################3 DDPG

env = Environment()
scaler = StandardScaler()

tau = 0.005   # Target network update factor (soft update)
exploration_noise_std = 0.2  # Standard deviation for exploration noise
action_low, action_high = -1, 1  # Action bounds for continuous actions
step_reward = 0


def continuous_to_discrete(action, threshold=0):
    """
    Maps continuous action values from [-1, 1] to discrete actions.
    :param action: The continuous action value.
    :param threshold: Threshold to decide the discrete action.
    :return: Discrete action (e.g., 0 or 1).
    """
    return 0 if action < threshold else 1

# Function to update target networks (soft update)
def update_target_networks(main_model, target_model, tau):
    """
    Soft update target network parameters.
    target_model = tau * main_model + (1 - tau) * target_model
    :param main_model: The main actor or critic model
    :param target_model: The target actor or critic model
    :param tau: Soft update parameter (typically close to 0)
    """
    # Get the weights of both the main model and target model
    main_weights = main_model.get_weights()
    target_weights = target_model.get_weights()

    # Perform the soft update
    updated_weights = []
    for main_weight, target_weight in zip(main_weights, target_weights):
        updated_weight = tau * main_weight + (1 - tau) * target_weight
        updated_weights.append(updated_weight)

    # Set the updated weights to the target model
    target_model.set_weights(updated_weights)

# Exploration noise for continuous actions
def add_noise(action, std_dev):
    noise = np.random.normal(0, std_dev, size=action.shape)
    return np.clip(action + noise, action_low, action_high)

# Copy weights from main networks to target networks (initialize target networks)
target_actor_1_ddpg = tf.keras.models.clone_model(actor_1_ddpg)
target_actor_1_ddpg.set_weights(actor_1_ddpg.get_weights())
target_critic_1_ddpg = tf.keras.models.clone_model(critic_1_ddpg)
target_critic_1_ddpg.set_weights(critic_1_ddpg.get_weights())

target_actor_2_ddpg = tf.keras.models.clone_model(actor_2_ddpg)
target_actor_2_ddpg.set_weights(actor_2_ddpg.get_weights())
target_critic_2_ddpg = tf.keras.models.clone_model(critic_2_ddpg)
target_critic_2_ddpg.set_weights(critic_2_ddpg.get_weights())

# Training Loop
for e in range(total_episodes):
    state = np.reshape(np.zeros(4), [1, 4])  # Example: 4-dimensional state
    done = False
    rewards, states, actions_1, actions_2 = [], [], [], []
    num_decrease_bandwidth_c = num_decrease_bandwidth_d = 0
    num_increase_bandwidth_c = num_increase_bandwidth_d = 0
    max_en_step = 0
    
    while not done:
        # Simulating environment values (you would replace these with actual values)
        TOTAL_PACKETS_SENT_d = random.uniform(1000, 5000)
        TOTAL_PACKETS_SENT_c = random.uniform(10, 30)
        packet_loss_percentage = random.uniform(0, 10)
        PACKET_SIZE_d = 1518
        PACKET_SIZE_c = 300
        packets_received_d = TOTAL_PACKETS_SENT_d * (1 - packet_loss_percentage / 100)
        packets_received_c = TOTAL_PACKETS_SENT_c * (1 - packet_loss_percentage / 100)
        Bw_c = Bw_d = random.uniform(50, 100)

        # Get current state and scale it
        state = np.array(env.current_state(PACKET_SIZE_c, PACKET_SIZE_d, TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d, Bw_c, Bw_d, packets_received_d, packets_received_c)[0]).reshape(1, 4)
        scaled_state = scaler.fit_transform(state)

        # Select actions using the actor networks
        action_1 = actor_1_ddpg.predict(scaled_state)
        action_2 = actor_2_ddpg.predict(scaled_state)

        # Add noise to encourage exploration during training (exploration strategy for DDPG)
        action_1 = add_noise(action_1, exploration_noise_std)
        action_2 = add_noise(action_2, exploration_noise_std)

        discrete_action_1 = continuous_to_discrete(action_1)
        discrete_action_2 = continuous_to_discrete(action_2)

        # Pass the discrete action to the environment
        Bw_c, selected_action_c = env.take_action(discrete_action_1, Bw_c)
        Bw_d, selected_action_d = env.take_action(discrete_action_2, Bw_d)

        

        # Get the next state and the throughput (replace with actual environment function)
        next_state, thrghpt_1, thrghpt_2 = env.current_state(PACKET_SIZE_c, PACKET_SIZE_d, TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d, Bw_c, Bw_d, packets_received_d, packets_received_c)

        # Update bandwidth counters based on actions taken
        if selected_action_c == 'decrease_bandwidth':
            num_decrease_bandwidth_c += 1
        else:
            num_increase_bandwidth_c += 1

        if selected_action_d == 'decrease_bandwidth':
            num_decrease_bandwidth_d += 1
        else:
            num_increase_bandwidth_d += 1

        # Scale the next state
        scaler.fit(state)
        next_state = np.reshape(next_state, [1, 4])
        scaled_next_state = scaler.transform(next_state)

       

        # Check if the episode is done
        done = (next_state[0][1] < 0.03 and next_state[0][3] < 0.03 and next_state[0][0] < 0.3 and next_state[0][2] < 0.3)
        max_en_step += 1

     # Calculate rewards and store states and actions
    step_reward += (0.5 * (0.75 * num_decrease_bandwidth_c + 0.25 * num_increase_bandwidth_c)) + (0.5 * (0.75 * num_decrease_bandwidth_d + 0.25 * num_increase_bandwidth_d))
    rewards.append(step_reward)
    states.append(scaled_state)
    actions_1.append(action_1)
    actions_2.append(action_2)

    # Sample action from target networks for the next state
    target_action_1 = target_actor_1_ddpg.predict(scaled_next_state)
    target_action_2 = target_actor_2_ddpg.predict(scaled_next_state)

    # Critic target using target networks (for DDPG's Bellman equation)
    target_q_value_1 = target_critic_1_ddpg.predict(np.concatenate([scaled_next_state, target_action_1], axis=1))
    target_q_value_2 = target_critic_2_ddpg.predict(np.concatenate([scaled_next_state, target_action_2], axis=1))

    # Calculate target Q-values for the critic update
    target_value_1 = step_reward + gamma * target_q_value_1
    target_value_2 = step_reward + gamma * target_q_value_2

    # Update the critic networks (Q-value estimation)
    critic_1_ddpg.fit(np.concatenate([scaled_state, action_1], axis=1), target_value_1, epochs=1, verbose=0)
    critic_2_ddpg.fit(np.concatenate([scaled_state, action_2], axis=1), target_value_2, epochs=1, verbose=0)

    
    # Ensure you're within a TensorFlow gradient context
    with tf.GradientTape() as tape_1, tf.GradientTape() as tape_2:
        # Predict the action for the current state
        actor_action_1 = actor_1_ddpg(scaled_state)
        actor_action_2 = actor_2_ddpg(scaled_state)

        # Predict Q-values from the critic using the action predicted by the actor
        q_value_1 = critic_1_ddpg(tf.concat([scaled_state, actor_action_1], axis=1))
        q_value_2 = critic_2_ddpg(tf.concat([scaled_state, actor_action_2], axis=1))

        # The actor's goal is to maximize the Q-value, so we minimize the negative Q-value
        actor_loss_1 = -tf.reduce_mean(q_value_1)
        actor_loss_2 = -tf.reduce_mean(q_value_2)

    # Compute the gradients for both actors
    grads_1 = tape_1.gradient(actor_loss_1, actor_1_ddpg.trainable_variables)
    grads_2 = tape_2.gradient(actor_loss_2, actor_2_ddpg.trainable_variables)

    # Apply the gradients to the actors' optimizers
    actor_1_ddpg.optimizer.apply_gradients(zip(grads_1, actor_1_ddpg.trainable_variables))
    actor_2_ddpg.optimizer.apply_gradients(zip(grads_2, actor_2_ddpg.trainable_variables))


    # Soft update of target networks
    update_target_networks(actor_1_ddpg, target_actor_1_ddpg, tau)
    update_target_networks(critic_1_ddpg, target_critic_1_ddpg, tau)
    update_target_networks(actor_2_ddpg, target_actor_2_ddpg, tau)
    update_target_networks(critic_2_ddpg, target_critic_2_ddpg, tau)

    exp_reward_ddpg = 1 - math.exp(-decay_rate * np.max(rewards))



    # Episode ended, update the histories and reward tracking
    packet_loss_history_d_ddpg.append(next_state[0][3])
    packet_loss_history_c_ddpg.append(next_state[0][1])
    d_history_d_ddpg.append(next_state[0][2])
    d_history_c_ddpg.append(next_state[0][0])
    th_history_d_ddpg.append(thrghpt_1)
    th_history_c_ddpg.append(thrghpt_2)
    bw_history_d_ddpg.append(Bw_d)
    bw_history_c_ddpg.append(Bw_c)
    rw_history_DDPG.append(exp_reward_ddpg)

print(rw_history_DDPG)

###########################3 Train Double DQN ######################## 

scaler = StandardScaler()
env = Environment()

step_reward = 0

for e in range(total_episodes):
    state = np.reshape(np.zeros(4), [1, s_size])
    done = False
    max_en_step = 0
    
    num_decrease_bandwidth_c = num_decrease_bandwidth_d = 0
    num_increase_bandwidth_c = num_increase_bandwidth_d = 0
    rewards, states, actions_1, actions_2 = [], [], [], []
    
    while not done:
        TOTAL_PACKETS_SENT_d = random.uniform(1000, 5000)
        TOTAL_PACKETS_SENT_c = random.uniform(10, 30)
        packet_loss_percentage = random.uniform(0, 10)
        PACKET_SIZE_d = 1518
        PACKET_SIZE_c = 300
        packets_received_d = TOTAL_PACKETS_SENT_d * (1 - packet_loss_percentage / 100)
        packets_received_c = TOTAL_PACKETS_SENT_c * (1 - packet_loss_percentage / 100)
        Bw_c = Bw_d = random.uniform(50, 100)

        # Get current state and scale it
        state = np.array(env.current_state(PACKET_SIZE_c, PACKET_SIZE_d, TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d, Bw_c, Bw_d, packets_received_d, packets_received_c)[0]).reshape(1, s_size)
        scaled_state = scaler.fit_transform(state)

        # Select actions using the actor networks (probabilistic policy)
        action_probs_1 = actor_1.predict(scaled_state)
        action_probs_2 = actor_2.predict(scaled_state)

        action_1 = np.random.choice(a_size, p=action_probs_1[0])  # Sample action for channel C
        action_2 = np.random.choice(a_size, p=action_probs_2[0])  # Sample action for channel D

        Bw_c, selected_action_c = env.take_action(action_1, Bw_c)
        Bw_d, selected_action_d = env.take_action(action_2, Bw_d)

        next_state, thrghpt_1, thrghpt_2 = env.current_state(PACKET_SIZE_c, PACKET_SIZE_d, TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d, Bw_c, Bw_d, packets_received_d, packets_received_c)

        # Update bandwidth counters based on actions taken
        if selected_action_c == 'decrease_bandwidth':
            num_decrease_bandwidth_c += 1
        else:
            num_increase_bandwidth_c += 1

        if selected_action_d == 'decrease_bandwidth':
            num_decrease_bandwidth_d += 1
        else:
            num_increase_bandwidth_d += 1

        scaler.fit(state)
        next_state = np.reshape(next_state, [1, s_size])
        scaled_next_state = scaler.transform(next_state)

        # Check if the episode is done
        done = (next_state[0][1] < 0.03 and next_state[0][3] < 0.03 and next_state[0][0] < 0.3 and next_state[0][2] < 0.3)

        max_en_step += 1

        # Calculate rewards and store states and actions
    step_reward += (0.5 * (0.75 * num_decrease_bandwidth_c + 0.25 * num_increase_bandwidth_c)) + (0.5 * (0.75 * num_decrease_bandwidth_d + 0.25 * num_increase_bandwidth_d))
    rewards.append(step_reward)
    states.append(scaled_state)
    actions_1.append(action_1)
    actions_2.append(action_2)
    
    # Once the episode is done, compute returns and advantages
    returns, advantages_1, advantages_2 = [], [], []
    next_value_1 = critic_1.predict(scaled_next_state)
    next_value_2 = critic_2.predict(scaled_next_state)
    G = 0

    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns.insert(0, G)  # Reverse to maintain correct order

        # Calculate the advantage for actor updates
        value_1 = critic_1.predict(states[t])
        value_2 = critic_2.predict(states[t])

        advantage_1 = G - value_1  # Temporal difference for actor 1
        advantage_2 = G - value_2  # Temporal difference for actor 2

        advantages_1.insert(0, advantage_1)  # Reverse order
        advantages_2.insert(0, advantage_2)  # Reverse order

    # Update the actor and critic networks for both channels
    actions_one_hot_1 = tf.keras.utils.to_categorical(actions_1, a_size)
    actions_one_hot_2 = tf.keras.utils.to_categorical(actions_2, a_size)

    # Actor networks are updated using the advantage as weights for policy gradients
    actor_1.fit(np.vstack(states), actions_one_hot_1, sample_weight=np.array(advantages_1).flatten(), epochs=1, verbose=0)
    actor_2.fit(np.vstack(states), actions_one_hot_2, sample_weight=np.array(advantages_2).flatten(), epochs=1, verbose=0)

    # Critic networks are updated using the returns (target values)
    critic_1.fit(np.vstack(states), np.array(returns), epochs=1, verbose=0)
    critic_2.fit(np.vstack(states), np.array(returns), epochs=1, verbose=0)

    state = next_state
    exp_reward_a2c = 1 - math.exp(-decay_rate * np.max(rewards))
    rw_history_A2C.append(exp_reward_a2c)

    # Tracking the results
    packet_loss_history_d_a2c.append(next_state[0][3])
    packet_loss_history_c_a2c.append(next_state[0][1])
    d_history_d_a2c.append(next_state[0][2])
    d_history_c_a2c.append(next_state[0][0])
    th_history_d_a2c.append(thrghpt_1)
    th_history_c_a2c.append(thrghpt_2)
    bw_history_d_a2c.append(Bw_d)
    bw_history_c_a2c.append(Bw_c)

print(rw_history_A2C)

###########################3 Train Dual DQN ######################## 

print('Training Dual DQN')

scaler = StandardScaler()
env = Environment()




for e in range(total_episodes):
    state = np.reshape(np.zeros(4), [1, s_size])
    action_1 = action_2 = 0
    Bw_c = Bw_d = 0
    done = False
    max_en_step = 0
    
    num_decrease_bandwidth_c = num_decrease_bandwidth_d = num_increase_bandwidth_c = num_increase_bandwidth_d = 0


    while not done:
        TOTAL_PACKETS_SENT_d = random.uniform(1000, 5000)
        TOTAL_PACKETS_SENT_c = random.uniform(10, 30)
        packet_loss_percentage = random.uniform(0, 10)
        PACKET_SIZE_d = 1518
        PACKET_SIZE_c = 300
        packets_received_d = TOTAL_PACKETS_SENT_d * (1 - packet_loss_percentage / 100)
        packets_received_c = TOTAL_PACKETS_SENT_c * (1 - packet_loss_percentage / 100)
        Bw_c = Bw_d = random.uniform(50, 100)
        
        state =  np.array(env.current_state(PACKET_SIZE_c, PACKET_SIZE_d, TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d, Bw_c, Bw_d, packets_received_d, packets_received_c)[0]).reshape(1, s_size)
        scaled_state = scaler.fit_transform(state)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * e)
        if np.random.rand() < epsilon:
            action1 = np.random.randint(a_size)
            action2 = np.random.randint(a_size)
        else:
            action1 = np.argmax(agent1_dualdqn.predict(state)[0])
            action2 = np.argmax(agent2_dualdqn.predict(state)[0])

        Bw_c, selected_action_c = env.take_action(action_1, Bw_c)
        Bw_d, selected_action_d = env.take_action(action_2, Bw_d)

        next_state, thrghpt_1, thrghpt_2 = env.current_state(PACKET_SIZE_c, PACKET_SIZE_d, TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d, Bw_c, Bw_d, packets_received_d, packets_received_c)

        if selected_action_c == 'decrease_bandwidth':
            num_decrease_bandwidth_c += 1
        else:
            num_increase_bandwidth_c += 1

        if selected_action_d == 'decrease_bandwidth':
            num_decrease_bandwidth_d += 1
        else:
            num_increase_bandwidth_d += 1

        scaler.fit(state)
        next_state = np.reshape(next_state, [1, s_size])
        scaled_next_state = scaler.transform(next_state)
        

        done = (next_state[0][1] < 0.03 and next_state[0][3] < 0.03 and next_state[0][0] < 0.3 and next_state[0][2] < 0.3)

        max_en_step += 1
    
    #reward += (0.5 * num_decrease_bandwidth_c + 0.5 * num_decrease_bandwidth_d)
    reward += (0.5 * (0.75 * num_decrease_bandwidth_c + 0.25 * num_increase_bandwidth_c)) + (0.5 * (0.75 * num_decrease_bandwidth_d + 0.25 * num_increase_bandwidth_d))
    exp_reward = 1 - math.exp(-decay_rate * reward)
    

    target1 = exp_reward + gamma * np.amax(agent1_dualdqn.predict(scaled_next_state)[0])
    target2 = exp_reward + gamma * np.amax(agent2_dualdqn.predict(scaled_next_state)[0])
    
       

    target_f1 = agent1_dualdqn.predict(scaled_state)
    target_f2 = agent2_dualdqn.predict(scaled_state)

    target_f1[0][action1] = target1
    target_f2[0][action1] = target2

    agent1_dualdqn.fit(scaled_state, target_f1,  epochs=1, batch_size = 64, verbose=0)
    agent2_dualdqn.fit(scaled_state, target_f2,  epochs=1, batch_size = 64, verbose=0)

    state = next_state
    rw_history_dualDQN.append(exp_reward)


    packet_loss_history_d_dualdqn.append(next_state[0][3])
    packet_loss_history_c_dualdqn.append(next_state[0][1])
    d_history_d_dualdqn.append(next_state[0][2])
    d_history_c_dualdqn.append(next_state[0][0])
    th_history_d_dualdqn.append(thrghpt_1)
    th_history_c_dualdqn.append(thrghpt_2)
    bw_history_d_dualdqn.append(Bw_d)
    bw_history_c_dualdqn.append(Bw_c)


#print(rw_history_dualDQN)
##################################3 Plotting Results #################################


# After training, you can plot the comparison

techniques = ['DuelDQN', 'RainbowDQN', 'A2C', 'DDPG']

fig, ax = plt.subplots(2, 3, figsize=(20, 10))  # 2 rows, 4 columns

# Rotate the x-axis labels to prevent overlapping
rotation_angle = 45

# Throughput plot for Data Plane
ax[0, 0].bar(techniques, [np.mean(th_history_d_dualdqn), np.mean(th_history_d_rainbowdqn), np.mean(th_history_d_a2c), np.mean(th_history_d_ddpg)], color='#2980b9')
ax[0, 0].set_title('Throughput Comparison Data Plane')
ax[0, 0].set_ylabel('Throughput (Mbps)')
ax[0, 0].tick_params(axis='x', rotation=rotation_angle)

# Packet Loss plot for Data Plane
ax[0, 1].bar(techniques, [np.mean(packet_loss_history_d_dualdqn), np.mean(packet_loss_history_d_rainbowdqn), np.mean(packet_loss_history_d_a2c), np.mean(packet_loss_history_d_ddpg)], color='#76d7c4')
ax[0, 1].set_title('Packet Loss Comparison Data Plane')
ax[0, 1].set_ylabel('Packet Loss')
ax[0, 1].tick_params(axis='x', rotation=rotation_angle)

# Delay plot for Data Plane
ax[0, 2].bar(techniques, [np.mean(d_history_d_dualdqn), np.mean(d_history_d_rainbowdqn), np.mean(d_history_d_a2c), np.mean(d_history_d_ddpg)], color='#45b39d')
ax[0, 2].set_title('Delay Comparison Data Plane')
ax[0, 2].set_ylabel('Delay (ms)')
ax[0, 2].tick_params(axis='x', rotation=rotation_angle)



# Throughput plot for Control Plane
ax[1, 0].bar(techniques, [np.mean(th_history_c_dualdqn), np.mean(th_history_c_rainbowdqn), np.mean(th_history_c_a2c), np.mean(th_history_c_ddpg)], color='#85c1e9')
ax[1, 0].set_title('Throughput Comparison Control Plane')
ax[1, 0].set_ylabel('Throughput (Mbps)')
ax[1, 0].tick_params(axis='x', rotation=rotation_angle)

# Packet Loss plot for Control Plane
ax[1, 1].bar(techniques, [np.mean(packet_loss_history_c_dualdqn), np.mean(packet_loss_history_c_rainbowdqn), np.mean(packet_loss_history_c_a2c), np.mean(packet_loss_history_c_ddpg)], color='#45b39d')
ax[1, 1].set_title('Packet Loss Comparison Control Plane')
ax[1, 1].set_ylabel('Packet Loss')
ax[1, 1].tick_params(axis='x', rotation=rotation_angle)

# Delay plot for Control Plane
ax[1, 2].bar(techniques, [np.mean(d_history_c_dualdqn), np.mean(d_history_c_rainbowdqn), np.mean(d_history_c_a2c), np.mean(d_history_c_ddpg)], color='#85c1e9')
ax[1, 2].set_title('Delay Comparison Control Plane')
ax[1, 2].set_ylabel('Delay (ms)')
ax[1, 2].tick_params(axis='x', rotation=rotation_angle)





plt.tight_layout()
plt.savefig(f'comparison_plot_{time_str}.png', bbox_inches='tight')


print("Plot Learning Performance")
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})



fig, ax = plt.subplots(figsize=(10,4))
plt.grid(True, linestyle='--')
plt.title('Learning Performance across Models')
plt.plot( episode_history, rw_history_dualDQN, label='DuelDQN', marker="", linestyle="-")#, color='k')
plt.plot( episode_history, rw_history_A2C, label='A2C', marker="", linestyle="-")
plt.plot( episode_history, rw_history_DDPG, label='DDPG', marker="", linestyle="-")#, color='k')
plt.plot( episode_history, rw_history_RainbowDQN, label='RainbowDQN', marker="", linestyle="-")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(prop={'size': 12})

plt.savefig(f'Reward Performance_{time_str}.pdf', bbox_inches='tight')





