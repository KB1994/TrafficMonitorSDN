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

################################# Constants ######################################################

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
bw_history_c_dualdqn = []
bw_history_d_dualdqn = []


packet_loss_history_c_doubledqn = []
packet_loss_history_d_doubledqn = []
th_history_c_doubledqn = []
th_history_d_doubledqn = []
bw_history_c_doubledqn = []
bw_history_d_doubledqn = []

packet_loss_history_c_cnn = []
packet_loss_history_d_cnn = []
th_history_c_cnn = []
th_history_d_cnn = []
bw_history_c_cnn = []
bw_history_d_cnn = []

packet_loss_history_c_lstm = []
packet_loss_history_d_lstm = []
th_history_c_lstm = []
th_history_d_lstm = []
bw_history_c_lstm = []
bw_history_d_lstm = []

packet_loss_history_d_ql = []
packet_loss_history_c_ql = []
th_history_d_ql = []
th_history_c_ql = []
bw_history_d_ql = []
bw_history_c_ql = []
episode_history = []
rw_history = []

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
        thrghpt_c = TOTAL_PACKETS_SENT_c / delay_c
        thrghpt_d = TOTAL_PACKETS_SENT_d / delay_d
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

agent1_dualdqn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

agent2_dualdqn = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(1 + 4)  # Output layer for Dueling DQN
])

agent2_dualdqn.add(DuelingDQNOutput())

agent2_dualdqn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')





#################################### Double DQN #######################################


agent1_doubledqn = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(5)  # Output layer for Q-values (assuming 4 actions + 1 for the value estimation)
])

# Custom Double DQN layer (if needed)
class DoubleDQNOutput(keras.layers.Layer):
    def call(self, inputs):
        value = inputs[:, :1]  # Value stream
        q_values = inputs[:, 1:]  # Q-value stream
        return value + (q_values - tf.reduce_mean(q_values, axis=1, keepdims=True))

# Add the custom layer to the model
agent1_doubledqn.add(DoubleDQNOutput())

# Compile the model with a Mean Squared Error loss function and Adam optimizer
agent1_doubledqn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

agent2_doubledqn = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(5)  # Output layer for Q-values (assuming 4 actions + 1 for the value estimation)
])



# Add the custom layer to the model
agent2_doubledqn.add(DoubleDQNOutput())

# Compile the model with a Mean Squared Error loss function and Adam optimizer
agent2_doubledqn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

#################################### DQN #######################################

agent1_ql = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
])

agent1_ql.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

agent2_ql = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
])

agent2_ql.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')






############ CNN Model #####################################################

# Using Conv1D with padding='same' to preserve the input size
agent1_cnn = models.Sequential([
    layers.Conv1D(32, 2, activation='relu', padding='same', input_shape=(4, 1), kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(10, activation='softmax')  # Assuming 10 actions
])

# Compile CNN model
agent1_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

agent2_cnn = models.Sequential([
    layers.Conv1D(32, 2, activation='relu', padding='same', input_shape=(4, 1), kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(10, activation='softmax')  # Assuming 10 actions
])

# Compile CNN model
agent2_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



############ LSTM Model #####################################################

agent1_lstm = models.Sequential([
    layers.LSTM(64, input_shape=(10, 4), return_sequences=True, kernel_regularizer=regularizers.l2(0.01)),
    layers.LSTM(128, kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(1)  # Output for regression or binary classification
])

# Compile LSTM model
agent1_lstm.compile(optimizer='adam', loss='mse')

agent2_lstm = models.Sequential([
    layers.LSTM(64, input_shape=(10, 4), return_sequences=True, kernel_regularizer=regularizers.l2(0.01)),
    layers.LSTM(128, kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(1)  # Output for regression or binary classification
])

# Compile LSTM model
agent2_lstm.compile(optimizer='adam', loss='mse')

################################## Train CNN #################################

print('Training CNN')

scaler = StandardScaler()
env = Environment()
reward = 0

for e in range(total_episodes):


    state = np.zeros((1, s_size))

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
        
        state = np.array(env.current_state(PACKET_SIZE_c, PACKET_SIZE_d, TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d, Bw_c, Bw_d, packets_received_d, packets_received_c)[0]).reshape(1, s_size)
        state = scaler.fit_transform(state).reshape(1, 4, 1)  # Reshape for Conv1D input
        
        action1 = np.argmax(agent1_cnn.predict(state)[0])
        action2 = np.argmax(agent2_cnn.predict(state)[0])
        
        Bw_c, selected_action_c = env.take_action(action1, Bw_c)
        Bw_d, selected_action_d = env.take_action(action2, Bw_d)
        
        next_state, thrghpt_1, thrghpt_2 = env.current_state(PACKET_SIZE_c, PACKET_SIZE_d, TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d, Bw_c, Bw_d, packets_received_d, packets_received_c)
        
        if selected_action_c == 'decrease_bandwidth':
            num_decrease_bandwidth_c += 1
        else:
            num_increase_bandwidth_c += 1

        if selected_action_d == 'decrease_bandwidth':
            num_decrease_bandwidth_d += 1
        else:
            num_increase_bandwidth_d += 1


        


        next_state = scaler.transform(np.array(next_state).reshape(1, s_size)).reshape(1, 4, 1)  # Reshape for Conv1D
        
        done = (next_state[0][1] < 0.03 and next_state[0][3] < 0.03 and next_state[0][0] < 0.3 and next_state[0][2] < 0.3)


        max_en_step += 1

    reward += (0.5 * num_decrease_bandwidth_c + 0.5 * num_decrease_bandwidth_d)
    exp_reward = 1 - math.exp(-decay_rate * reward)
    

    target1 = exp_reward + gamma * np.amax(agent1_cnn.predict(next_state)[0])
    target2 = exp_reward + gamma * np.amax(agent2_cnn.predict(next_state)[0])
    
       

    target_f1 = agent1_cnn.predict(state)
    target_f2 = agent2_cnn.predict(state)

    target_f1[0][action1] = target1
    target_f2[0][action1] = target2

    agent1_cnn.fit(state, target_f1,  epochs=1, batch_size = 64, verbose=0)
    agent2_cnn.fit(state, target_f2,  epochs=1, batch_size = 64, verbose=0)

    state = next_state
    # Save the results
    packet_loss_history_d_cnn.append(next_state[0][3])
    packet_loss_history_c_cnn.append(next_state[0][1])
    th_history_d_cnn.append(thrghpt_1)
    th_history_c_cnn.append(thrghpt_2)
    bw_history_d_cnn.append(Bw_d)
    bw_history_c_cnn.append(Bw_c)

################################## Train LSTM #################################

print('Training LSTM')
'''
scaler = StandardScaler()
env = Environment()

reward = 0
state_sequence = []
max_sequence_length = 10

for e in range(total_episodes):
    state = np.zeros((1, s_size))

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

        state = np.array(env.current_state(PACKET_SIZE_c, PACKET_SIZE_d, TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d, Bw_c, Bw_d, packets_received_d, packets_received_c)[0])
        state = scaler.fit_transform(state.reshape(-1, 1)).reshape(1, s_size)  # Rescale the state
        
        # Append the current state to the state sequence
        state_sequence.append(state)

        # Ensure the sequence does not exceed the max sequence length
        if len(state_sequence) > max_sequence_length:
            state_sequence.pop(0)

        # Convert the state_sequence to an appropriate shape for LSTM input (1, sequence_length, s_size)
        lstm_input = np.array(state_sequence).reshape(1, len(state_sequence), s_size)

        # Use the LSTM models to predict actions
        action1 = np.argmax(agent1_lstm.predict(lstm_input)[0])
        action2 = np.argmax(agent2_lstm.predict(lstm_input)[0])

        # Take action in the environment
        Bw_c, selected_action_c = env.take_action(action1, Bw_c)
        Bw_d, selected_action_d = env.take_action(action2, Bw_d)

        # Get the next state from the environment
        next_state, thrghpt_1, thrghpt_2 = env.current_state(PACKET_SIZE_c, PACKET_SIZE_d, TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d, Bw_c, Bw_d, packets_received_d, packets_received_c)
        next_state = np.array(next_state)
        next_state = scaler.transform(next_state.reshape(-1, 1)).reshape(1, s_size)  # Rescale the next state

        # Append the next state to the state sequence
        state_sequence.append(next_state)
        if len(state_sequence) > max_sequence_length:
            state_sequence.pop(0)

        # Reshape the state sequence for the next step
        lstm_input = np.array(state_sequence).reshape(1, len(state_sequence), s_size)

        # Check if the episode is done (based on custom criteria)
        done = (next_state[0][1] < 0.03 and next_state[0][3] < 0.03 and next_state[0][0] < 0.3 and next_state[0][2] < 0.3)
        max_en_step += 1

    # Reward calculation and update
    reward += (0.5 * num_decrease_bandwidth_c + 0.5 * num_decrease_bandwidth_d)
    exp_reward = 1 - math.exp(-decay_rate * reward)

    # Target Q-values for LSTM
    target1 = exp_reward + gamma * np.amax(agent1_lstm.predict(lstm_input)[0])
    target2 = exp_reward + gamma * np.amax(agent2_lstm.predict(lstm_input)[0])

    # Get current Q-values and update the action's Q-value
    target_f1 = agent1_lstm.predict(lstm_input)
    target_f2 = agent2_lstm.predict(lstm_input)

    target_f1[0][action1] = target1
    target_f2[0][action2] = target2

    # Fit the models with updated Q-values
    agent1_lstm.fit(lstm_input, target_f1, epochs=1, batch_size=64, verbose=0)
    agent2_lstm.fit(lstm_input, target_f2, epochs=1, batch_size=64, verbose=0)
    state = next_state


    # Save the results
    packet_loss_history_d_lstm.append(next_state[3])
    packet_loss_history_c_lstm.append(next_state[1])
    th_history_d_lstm.append(thrghpt_1)
    th_history_c_lstm.append(thrghpt_2)
    bw_history_d_lstm.append(Bw_d)
    bw_history_c_lstm.append(Bw_c)'''

###########################3 Train DQNModel ######################## 

print('Training DQN')

scaler = StandardScaler()
env = Environment()

reward = 0

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
            action1 = np.argmax(agent1_ql.predict(state)[0])
            action2 = np.argmax(agent2_ql.predict(state)[0])

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

    

    reward += (0.5 * num_decrease_bandwidth_c + 0.5 * num_decrease_bandwidth_d)
    exp_reward = 1 - math.exp(-decay_rate * reward)
    

    target1 = exp_reward + gamma * np.amax(agent1_ql.predict(scaled_next_state)[0])
    target2 = exp_reward + gamma * np.amax(agent2_ql.predict(scaled_next_state)[0])
    
       

    target_f1 = agent1_ql.predict(scaled_state)
    target_f2 = agent2_ql.predict(scaled_state)

    target_f1[0][action1] = target1
    target_f2[0][action1] = target2

    agent1_ql.fit(scaled_state, target_f1,  epochs=1, batch_size = 64, verbose=0)
    agent2_ql.fit(scaled_state, target_f2,  epochs=1, batch_size = 64, verbose=0)
    state = next_state

    state = next_state


    packet_loss_history_d_ql.append(next_state[0][3])
    packet_loss_history_c_ql.append(next_state[0][1])
    th_history_d_ql.append(thrghpt_1)
    th_history_c_ql.append(thrghpt_2)
    bw_history_d_ql.append(Bw_d)
    bw_history_c_ql.append(Bw_c)


###########################3 Train Double DQN ######################## 

print('Training Double DQN')

scaler = StandardScaler()
env = Environment()

reward = 0

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
            action1 = np.argmax(agent1_doubledqn.predict(state)[0])
            action2 = np.argmax(agent2_doubledqn.predict(state)[0])

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

    

    reward += (0.5 * num_decrease_bandwidth_c + 0.5 * num_decrease_bandwidth_d)
    exp_reward = 1 - math.exp(-decay_rate * reward)
    

    target1 = exp_reward + gamma * np.amax(agent1_doubledqn.predict(scaled_next_state)[0])
    target2 = exp_reward + gamma * np.amax(agent2_doubledqn.predict(scaled_next_state)[0])
    
       

    target_f1 = agent1_doubledqn.predict(scaled_state)
    target_f2 = agent2_doubledqn.predict(scaled_state)

    target_f1[0][action1] = target1
    target_f2[0][action1] = target2

    agent1_doubledqn.fit(scaled_state, target_f1,  epochs=1, batch_size = 64, verbose=0)
    agent2_doubledqn.fit(scaled_state, target_f2,  epochs=1, batch_size = 64, verbose=0)

    state = next_state


    packet_loss_history_d_doubledqn.append(next_state[0][3])
    packet_loss_history_c_doubledqn.append(next_state[0][1])
    th_history_d_doubledqn.append(thrghpt_1)
    th_history_c_doubledqn.append(thrghpt_2)
    bw_history_d_doubledqn.append(Bw_d)
    bw_history_c_doubledqn.append(Bw_c)


###########################3 Train Dual DQN ######################## 

print('Training Dual DQN')

scaler = StandardScaler()
env = Environment()

reward = 0

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

    

    reward += (0.5 * num_decrease_bandwidth_c + 0.5 * num_decrease_bandwidth_d)
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
    episode_history.append(e)


    packet_loss_history_d_dualdqn.append(next_state[0][3])
    packet_loss_history_c_dualdqn.append(next_state[0][1])
    th_history_d_dualdqn.append(thrghpt_1)
    th_history_c_dualdqn.append(thrghpt_2)
    bw_history_d_dualdqn.append(Bw_d)
    bw_history_c_dualdqn.append(Bw_c)


##################################3 Plotting Results #################################

# After training, you can plot the comparison

techniques = ['Dueling DQN', 'DQN', 'Double DQN', 'CNN']

# Plotting without normalization
fig, ax = plt.subplots(1, 4, figsize=(15, 5))

# Throughput plot for Data Plane
ax[0].bar(techniques, [np.mean(th_history_d_dualdqn), np.mean(th_history_d_ql), np.mean(th_history_d_doubledqn), np.mean(th_history_d_cnn)], color='blue')
ax[0].set_title('Throughput Comparison Data Plane')
ax[0].set_ylabel('Throughput')

# Packet Loss plot for Data Plane
ax[1].bar(techniques, [np.mean(packet_loss_history_d_dualdqn), np.mean(packet_loss_history_d_ql), np.mean(packet_loss_history_d_doubledqn), np.mean(packet_loss_history_d_cnn)], color='green')
ax[1].set_title('Packet Loss Comparison Data Plane')
ax[1].set_ylabel('Packet Loss')

# Throughput plot for Control Plane
ax[2].bar(techniques, [np.mean(th_history_c_dualdqn), np.mean(th_history_c_ql), np.mean(th_history_c_doubledqn), np.mean(th_history_c_cnn)], color='blue')
ax[2].set_title('Throughput Comparison Control Plane')
ax[2].set_ylabel('Throughput')

# Packet Loss plot for Control Plane
ax[3].bar(techniques, [np.mean(packet_loss_history_c_dualdqn), np.mean(packet_loss_history_c_ql), np.mean(th_history_c_doubledqn), np.mean(packet_loss_history_c_cnn)], color='green')
ax[3].set_title('Packet Loss Comparison Control Plane')
ax[3].set_ylabel('Packet Loss')

plt.tight_layout()
plt.savefig('comparison_plot.png', bbox_inches='tight')



fig1, ax = plt.subplots(figsize=(10, 6))  # No need for (1, 4) if plotting a single plot

# Plot the throughput history for different algorithms on the same axes
ax.plot(th_history_d_dualdqn, label='DualDQN')
ax.plot(th_history_d_ql, label='DQN')
ax.plot(th_history_d_doubledqn, label='DoubleDQN')
ax.plot(th_history_d_cnn, label='CNN')

# Add title and labels
ax.set_title('Throughput over Episodes Data plane')
ax.set_xlabel('Episode')
ax.set_ylabel('Throughput')

# Add a legend to differentiate between the lines
ax.legend()

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('comparison_Throughput_plot Data.png', bbox_inches='tight')

fig1, ax = plt.subplots(figsize=(10, 6))  # No need for (1, 4) if plotting a single plot

# Plot the throughput history for different algorithms on the same axes
ax.plot(th_history_c_dualdqn, label='DualDQN')
ax.plot(th_history_c_ql, label='DQN')
ax.plot(th_history_c_doubledqn, label='DoubleDQN')
ax.plot(th_history_c_cnn, label='CNN')

# Add title and labels
ax.set_title('Throughput over Episodes Control plane')
ax.set_xlabel('Episode')
ax.set_ylabel('Throughput')

# Add a legend to differentiate between the lines
ax.legend()

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('comparison_Throughput_plot Control.png', bbox_inches='tight')

