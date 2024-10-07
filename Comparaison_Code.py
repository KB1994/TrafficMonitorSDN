import random
import tensorflow as tf
import tf_slim as slim
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from collections import Counter
from sklearn.preprocessing import StandardScaler
#from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers
import math
from tensorflow.keras import layers, regularizers, models





################################# Constants ######################################################333


s_size = 4
a_size = 4
total_episodes = 600
max_env_steps = 6
#epsilon_min = 0.02
#epsilon_decay = 0.999
time_history = []
th_history_d = []
th_history_c = []
bw_history_d = []
bw_history_c = []
in_history_d = []
de_history_d = []
in_history_c = []
de_history_c = []
packet_loss_history_d = []
packet_loss_history_c = []
packet_loss_history_d_ql = []
packet_loss_history_c_ql = []
th_history_d_ql = []
th_history_c_ql = []
bw_history_d_ql = []
bw_history_c_ql = []
packet_loss_history_d_dqn = []
packet_loss_history_c_dqn = []
th_history_d_dqn = []
th_history_c_dqn = []
bw_history_d_dqn = []
bw_history_c_dqn = []
packet_loss_history_d_cnn = []
packet_loss_history_c_cnn = []
th_history_d_cnn = []
th_history_c_cnn = []
bw_history_d_cnn = []
bw_history_c_cnn = []
packet_loss_history_d_lstm = []
packet_loss_history_c_lstm = []
th_history_d_lstm = []
th_history_c_lstm = []
bw_history_d_lstm = []
bw_history_c_lstm = []
episode_history = []
rw_history = []
gamma = 0.95
epsilon = 0.1  # Probability of selecting a random action (exploration)
histo_Bw = []
selected_action_c = ''
selected_action_d = ''

histo_selected_action_d = []
histo_selected_action_c = []
max_epsilon = 1.0  # Initial epsilon value
min_epsilon = 0.01  # Minimum epsilon value
decay_rate = 0.005  # Rate at which epsilon decays


#####################################################################################

def normalize(data, min_value, max_value):
    return [(value - min_value) / (max_value - min_value) for value in data]


def discretize_state(state, state_size):

    
    if state[0][0] < 0.3   and state[0][1]  < 0.03 or state[0][2] < 0.3 and state[0][3] < 0.03 :
        return 0
    elif state[0][0]  > 0.3   and state[0][1]  < 0.03 or state[0][2] > 0.3 and state[0][3] < 0.03 :
        return 1
    elif state[0][0] < 0.3   and state[0][1]  > 0.03 or state[0][2] < 0.3 and state[0][3] > 0.03 :
        return 2
    elif state[0][0] > 0.3   and state[0][1]  > 0.03 or state[0][2] > 0.3 and state[0][3] > 0.03 :
        return 3
    

################################ Define Environment #################################



# define the action
# define the policy of update 

class Environment:
    def __init__(self):
        self.num_states = 4
        self.num_actions = 2
        self.state = 0
        self.BANDWIDTH_MAX = 100  # Bandwidth in Mbits per second
        self.BANDWIDTH_MIN = 50  # Bandwidth in Mbits per second
        #self.PACKET_SIZE_c = PACKET_SIZE_c
        #self.PACKET_SIZE_d = PACKET_SIZE_d
        #self.TOTAL_PACKETS_SENT_c = TOTAL_PACKETS_SENT_c
        #self.TOTAL_PACKETS_SENT_d = TOTAL_PACKETS_SENT_d
        #self.Bw_c = Bw_c
        #self.Bw_d = Bw_d
        #self.packets_received_d = packets_received_d
        #self.packets_received_c = packets_received_c
        self.queue_length = 10   # Length of the queue in packets
        self.service_rate = 100  # Router's service rate in packets per second
        self.processing_time_per_packet = 0.001  # Processing time per packet in seconds
        self.actions = ['increase_bandwidth', 'decrease_bandwidth']
        
        

    def reset(self):
        self.current_state = 0
        return self.current_state

    def current_state(self, PACKET_SIZE_c, PACKET_SIZE_d, TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d, Bw_c, Bw_d, packets_received_d, packets_received_c):

        loss_c = (TOTAL_PACKETS_SENT_c - packets_received_c) / TOTAL_PACKETS_SENT_c
        loss_d = (TOTAL_PACKETS_SENT_d - packets_received_d) / TOTAL_PACKETS_SENT_d
        delay_c = (PACKET_SIZE_c / (Bw_c * 10**6))+ (self.queue_length / self.service_rate) + self.processing_time_per_packet
        delay_d = (PACKET_SIZE_d / (Bw_d * 10**6))+ (self.queue_length / self.service_rate) + self.processing_time_per_packet
        thrghpt_c = TOTAL_PACKETS_SENT_c/delay_c
        thrghpt_d = TOTAL_PACKETS_SENT_d/delay_d

        return [delay_c, loss_c, delay_d, loss_d], thrghpt_c, thrghpt_d


    def take_action(self, action_n, current_bandwidth):
        # Select a random action
        
        selected_action = self.actions[action_n % 2]
    
        # Execute selected action
        if selected_action == 'increase_bandwidth':
            Bw = self.increase_bandwidth(current_bandwidth, self.BANDWIDTH_MAX)
        elif selected_action == 'decrease_bandwidth':
            Bw = self.decrease_bandwidth(current_bandwidth, self.BANDWIDTH_MIN)

        return Bw, selected_action




    # Example functions for actions
    @staticmethod
    def increase_bandwidth(current_bandwidth, BANDWIDTH_MAX):
        if float(current_bandwidth) < BANDWIDTH_MAX:
            current_bandwidth += 10  # Increase bandwidth by 10 Mbps
        return float(current_bandwidth)

    @staticmethod
    def decrease_bandwidth(current_bandwidth, BANDWIDTH_MIN):
        
        if float(current_bandwidth)> BANDWIDTH_MIN:
            current_bandwidth -= 10  # Decrease bandwidth by 10 Mbps
        
        return float(current_bandwidth)


############ Dualing DQN Agent ####################################################
    
agent1 = keras.Sequential([
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

agent1.add(DuelingDQNOutput())

agent1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

agent2 = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(1 + 4)  # Output layer for Dueling DQN
])

agent2.add(DuelingDQNOutput())

agent2.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')





#################################### Simple Q-Learning agent #######################################


agent1_dqn = keras.Sequential([
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
agent1_dqn.add(DoubleDQNOutput())

# Compile the model with a Mean Squared Error loss function and Adam optimizer
agent1_dqn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

agent2_dqn = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(5)  # Output layer for Q-values (assuming 4 actions + 1 for the value estimation)
])



# Add the custom layer to the model
agent2_dqn.add(DoubleDQNOutput())

# Compile the model with a Mean Squared Error loss function and Adam optimizer
agent2_dqn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')


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


##################################3 CNN Model 


agent1_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3), kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(10, activation='softmax')  # For 10-class classification
])

# Compile the CNN model
agent1_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

agent2_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3), kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(10, activation='softmax')  # For 10-class classification
])

# Compile the CNN model
agent2_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


############################ Define the LSTM Model


agent1_lstm = models.Sequential([
    layers.LSTM(64, input_shape=(10, 8), return_sequences=True, kernel_regularizer=regularizers.l2(0.01)),
    layers.LSTM(128, kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(1)  # Output for regression or binary classification
])

# Compile the LSTM model
agent1_lstm.compile(optimizer='adam', loss='mse')

# Define the LSTM Model
agent2_lstm = models.Sequential([
    layers.LSTM(64, input_shape=(10, 8), return_sequences=True, kernel_regularizer=regularizers.l2(0.01)),
    layers.LSTM(128, kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(1)  # Output for regression or binary classification
])

# Compile the LSTM model
agent2_lstm.compile(optimizer='adam', loss='mse')


###########################3 Train Q-Learning Model ######################## 

print('Training simple Q-Learning')

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

    print(next_state[0])


    packet_loss_history_d_ql.append(next_state[0][3])
    packet_loss_history_c_ql.append(next_state[0][1])
    th_history_d_ql.append(thrghpt_1)
    th_history_c_ql.append(thrghpt_2)
    bw_history_d_ql.append(Bw_d)
    bw_history_c_ql.append(Bw_c)

###########################3 Train Deep Q-Learning Model ######################## 

print('Training Deep Q-Learning')

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
            action1 = np.argmax(agent1_dqn.predict(state)[0])
            action2 = np.argmax(agent2_dqn.predict(state)[0])

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
    

    target1 = exp_reward + gamma * np.amax(agent1_dqn.predict(scaled_next_state)[0])
    target2 = exp_reward + gamma * np.amax(agent2_dqn.predict(scaled_next_state)[0])
    
       

    target_f1 = agent1_dqn.predict(scaled_state)
    target_f2 = agent2_dqn.predict(scaled_state)

    target_f1[0][action1] = target1
    target_f2[0][action1] = target2

    agent1_dqn.fit(scaled_state, target_f1,  epochs=1, batch_size = 64, verbose=0)
    agent2_dqn.fit(scaled_state, target_f2,  epochs=1, batch_size = 64, verbose=0)

    print(next_state[0])


    packet_loss_history_d_dqn.append(next_state[0][3])
    packet_loss_history_c_dqn.append(next_state[0][1])
    th_history_d_dqn.append(thrghpt_1)
    th_history_c_dqn.append(thrghpt_2)
    bw_history_d_dqn.append(Bw_d)
    bw_history_c_dqn.append(Bw_c)


###########################3 Train CNN Model ######################## 

print('Training CNN')

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
            action1 = np.argmax(agent1_cnn.predict(state)[0])
            action2 = np.argmax(agent2_cnn.predict(state)[0])

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
    

    target1 = exp_reward + gamma * np.amax(agent1_cnn.predict(scaled_next_state)[0])
    target2 = exp_reward + gamma * np.amax(agent2_cnn.predict(scaled_next_state)[0])
    
       

    target_f1 = agent1_cnn.predict(scaled_state)
    target_f2 = agent2_cnn.predict(scaled_state)

    target_f1[0][action1] = target1
    target_f2[0][action1] = target2

    agent1_cnn.fit(scaled_state, target_f1,  epochs=1, batch_size = 64, verbose=0)
    agent2_cnn.fit(scaled_state, target_f2,  epochs=1, batch_size = 64, verbose=0)

    print(next_state[0])


    packet_loss_history_d_cnn.append(next_state[0][3])
    packet_loss_history_c_cnn.append(next_state[0][1])
    th_history_d_cnn.append(thrghpt_1)
    th_history_c_cnn.append(thrghpt_2)
    bw_history_d_cnn.append(Bw_d)
    bw_history_c_cnn.append(Bw_c)


###########################3 Train CNN Model ######################## 

print('Training CNN')

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
            action1 = np.argmax(agent1_lstm.predict(state)[0])
            action2 = np.argmax(agent2_lstm.predict(state)[0])

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
    

    target1 = exp_reward + gamma * np.amax(agent1_lstm.predict(scaled_next_state)[0])
    target2 = exp_reward + gamma * np.amax(agent2_lstm.predict(scaled_next_state)[0])
    
       

    target_f1 = agent1_lstm.predict(scaled_state)
    target_f2 = agent2_lstm.predict(scaled_state)

    target_f1[0][action1] = target1
    target_f2[0][action1] = target2

    agent1_lstm.fit(scaled_state, target_f1,  epochs=1, batch_size = 64, verbose=0)
    agent2_lstm.fit(scaled_state, target_f2,  epochs=1, batch_size = 64, verbose=0)

    print(next_state[0])


    packet_loss_history_d_lstm.append(next_state[0][3])
    packet_loss_history_c_lstm.append(next_state[0][1])
    th_history_d_lstm.append(thrghpt_1)
    th_history_c_lstm.append(thrghpt_2)
    bw_history_d_lstm.append(Bw_d)
    bw_history_c_lstm.append(Bw_c)




###################################### Training Dual Q-Learning ############################33

print('Training Dual Q-Learning')


env = Environment()

reward = 0
for e in range(total_episodes):
    state = next_state = np.reshape(np.zeros(4), [1, s_size])
    action1 = action2 = 0
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
            action1 = np.argmax(agent1.predict(state)[0])
            action2 = np.argmax(agent2.predict(state)[0])

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


        
        done = (next_state[1] < 0.03 and next_state[3] < 0.03 and next_state[0] < 0.3 and next_state[2] < 0.3 )
        
        next_state = np.reshape(next_state, [1, s_size])
        scaled_next_state = scaler.transform(next_state)
        max_en_step += 1


    histo_selected_action_d.append(selected_action_d)
    count_stat_d = Counter(histo_selected_action_d)
    histo_selected_action_c.append(selected_action_c)
    count_stat_c = Counter(histo_selected_action_c)
    
    #reward += (0.5 * (0.75 * num_decrease_bandwidth_c + 0.25 * num_increase_bandwidth_c)) + (0.5 * (0.75 * num_decrease_bandwidth_d + 0.25 * num_increase_bandwidth_d))
    
    reward += (0.5 * num_decrease_bandwidth_c  + 0.5 * num_decrease_bandwidth_d )
    exp_reward =  1 - math.exp(- decay_rate * reward )
    
    rw_history.append(exp_reward)
        

    

   
    
    target1 = exp_reward + gamma * np.amax(agent1.predict(scaled_next_state)[0])
    target2 = exp_reward + gamma * np.amax(agent2.predict(scaled_next_state)[0])
    
       

    target_f1 = agent1.predict(scaled_state)
    target_f2 = agent2.predict(scaled_state)

    target_f1[0][action1] = target1
    target_f2[0][action1] = target2

    agent1.fit(scaled_state, target_f1,  epochs=1, batch_size = 64, verbose=0)
    agent2.fit(scaled_state, target_f2,  epochs=1, batch_size = 64, verbose=0)
    print(next_state[0])

    state = next_state
    #if epsilon > epsilon_min: epsilon *= epsilon_decay
            

    
    histo_Bw.append(max_en_step)


    
    
    packet_loss_history_d.append(next_state[0][3])
    packet_loss_history_c.append(next_state[0][1])
    th_history_d.append(thrghpt_1)
    th_history_c.append(thrghpt_2)
    bw_history_d.append(Bw_d)
    bw_history_c.append(Bw_c)

    episode_history.append(e)
    #print( e, Counter(histo_selected_action))




  







############################ Parameter avearge #########################################


avg_throughput_d = []
avg_packet_loss_d  = []
avg_bandwidth_d  = []

avg_throughput_c = []
avg_packet_loss_c = []
avg_bandwidth_c  = []

avg_throughput_d =  [np.mean(th_history_d) , np.mean(th_history_d_ql),  np.mean(th_history_d_dqn), np.mean(th_history_d_cnn) , np.mean(th_history_d_lstm)]
avg_packet_loss_d  = [np.mean(packet_loss_history_d) , np.mean(packet_loss_history_d_ql) , np.mean(packet_loss_history_d_dqn),  np.mean(packet_loss_history_d_cnn), np.mean(packet_loss_history_d_lstm)]
avg_bandwidth_d  = [np.mean(bw_history_d) , np.mean(bw_history_d_ql),  np.mean(bw_history_d_dqn),  np.mean(bw_history_d_cnn) , np.mean(bw_history_d_lstm)]

avg_packet_loss_c = [np.mean(packet_loss_history_c) , np.mean(packet_loss_history_c_ql), np.mean(packet_loss_history_c_dqn), np.mean(packet_loss_history_c_cnn) , np.mean(packet_loss_history_c_lstm)]
avg_throughput_c  = [np.mean(th_history_c) , np.mean(th_history_c_ql), np.mean(th_history_c_dqn), np.mean(th_history_c_cnn) , np.mean(th_history_c_lstm)]
avg_bandwidth_c  = [np.mean(bw_history_c) , np.mean(bw_history_c_ql), np.mean(bw_history_c_dqn), np.mean(bw_history_c_cnn) , np.mean(bw_history_c_lstm)]

throughput_normalized_d = normalize(avg_throughput_d, min(avg_throughput_d), max(avg_throughput_d))
packet_loss_normalized_d = normalize(avg_packet_loss_d, min(avg_packet_loss_d), max(avg_packet_loss_d))
throughput_normalized_c = normalize(avg_throughput_c, min(avg_throughput_c), max(avg_throughput_c))
packet_loss_normalized_c = normalize(avg_packet_loss_c, min(avg_packet_loss_c), max(avg_packet_loss_c))
#bandwidth_normalized = normalize(bandwidth_utilization, min(bandwidth_utilization), max(bandwidth_utilization))

techniques = ['Dueling DQN', 'DQN', 'Double DQN', 'CNN', 'LSTM']
throughput_y_max = max(max(avg_throughput_d), max(avg_throughput_c))   # Add 10% buffer
packet_loss_y_max = max(max(avg_packet_loss_d), max(avg_packet_loss_c))   # Add 10% buffer

# Plotting without normalization
fig, ax = plt.subplots(1, 4, figsize=(15, 5))

# Throughput plot for Data Plane
ax[0].bar(techniques, avg_throughput_d, color='blue')
ax[0].set_title('Throughput Comparison Data Plane')
ax[0].set_ylabel('Throughput')
ax[0].set_ylim(0, throughput_y_max)  # Set y-axis based on max throughput value
ax[0].grid(True, linestyle='--')

# Packet Loss plot for Data Plane
ax[1].bar(techniques, avg_packet_loss_d, color='green')
ax[1].set_title('Packet Loss Comparison Data Plane')
ax[1].set_ylabel('Packet Loss')
ax[1].set_ylim(0, packet_loss_y_max)  # Set y-axis based on max packet loss value
ax[1].grid(True, linestyle='--')

# Throughput plot for Control Plane
ax[2].bar(techniques, avg_throughput_c, color='blue')
ax[2].set_title('Throughput Comparison Control Plane')
ax[2].set_ylabel('Throughput')
ax[2].set_ylim(0, throughput_y_max)  # Set y-axis same as for Data Plane throughput for consistency
ax[2].grid(True, linestyle='--')

# Packet Loss plot for Control Plane
ax[3].bar(techniques, avg_packet_loss_c, color='green')
ax[3].set_title('Packet Loss Comparison Control Plane')
ax[3].set_ylabel('Packet Loss')
ax[3].set_ylim(0, packet_loss_y_max)  # Set y-axis same as for Data Plane packet loss for consistency
ax[3].grid(True, linestyle='--')


# Show the plots
plt.tight_layout()
plt.savefig('comparison_plot.png', bbox_inches='tight')


