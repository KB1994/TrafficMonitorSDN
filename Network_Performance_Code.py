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
import datetime


current_time = datetime.datetime.now()

time_str = current_time.strftime("%Y%m%d_%H%M%S")


s_size = 4
a_size = 4
total_episodes = 600
max_env_steps = 6
#epsilon_min = 0.02
#epsilon_decay = 0.999
time_history = []
th_history_d_1 = []
th_history_c_1 = []
bw_history_d_1 = []
bw_history_c_1 = []
d_history_d_1 = []
d_history_c_1 = []
packet_loss_history_d_1 = []
packet_loss_history_c_1 = []

th_history_d_2 = []
th_history_c_2 = []
bw_history_d_2 = []
bw_history_c_2 = []
d_history_d_2 = []
d_history_c_2 = []
packet_loss_history_d_2 = []
packet_loss_history_c_2 = []

th_history_d_3 = []
th_history_c_3 = []
bw_history_d_3 = []
bw_history_c_3 = []
d_history_d_3 = []
d_history_c_3 = []
packet_loss_history_d_3 = []
packet_loss_history_c_3 = []

th_history_d_4 = []
th_history_c_4 = []
bw_history_d_4 = []
bw_history_c_4 = []
d_history_d_4 = []
d_history_c_4 = []
packet_loss_history_d_4 = []
packet_loss_history_c_4 = []



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


# define the action
# define the policy of update 

class Environment:
    def __init__(self):
        self.num_states = 4
        self.num_actions = 2
        self.state = 0
        self.BANDWIDTH_MAX = 100  # Bandwidth in Mbits per second
        self.BANDWIDTH_MIN = 50  # Bandwidth in Mbits per second
        self.PACKET_SIZE_c = PACKET_SIZE_c
        self.PACKET_SIZE_d = PACKET_SIZE_d
        self.TOTAL_PACKETS_SENT_c = TOTAL_PACKETS_SENT_c
        self.TOTAL_PACKETS_SENT_d = TOTAL_PACKETS_SENT_d
        self.Bw_c = Bw_c
        self.Bw_d = Bw_d
        self.packets_received_d = packets_received_d
        self.packets_received_c = packets_received_c
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

class DuelingDQNOutput(keras.layers.Layer):
    def call(self, inputs):
        value = inputs[:, :1]  # Value stream
        advantages = inputs[:, 1:]  # Advantage stream
        q_values = value + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))
        return q_values

agent1 = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(1 + 4)  # Output layer for Dueling DQN
])

agent1.add(DuelingDQNOutput())

agent1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')

agent2 = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(1 + 4)  # Output layer for Dueling DQN
])

agent2.add(DuelingDQNOutput())

agent2.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')


agent3 = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(1 + 4)  # Output layer for Dueling DQN
])



agent3.add(DuelingDQNOutput())

agent3.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy')

agent4 = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(1 + 4)  # Output layer for Dueling DQN
])

agent4.add(DuelingDQNOutput())

agent4.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy')

agent5 = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(1 + 4)  # Output layer for Dueling DQN
])



agent5.add(DuelingDQNOutput())

agent5.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.001), loss='binary_crossentropy')

agent6 = keras.Sequential([
    keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(1 + 4)  # Output layer for Dueling DQN
])

agent6.add(DuelingDQNOutput())

agent6.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.001), loss='binary_crossentropy')



#################################### Training 1 ########################################

scaler = StandardScaler()

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
        env = Environment()
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
    print(exp_reward)
    rw_history.append(exp_reward)
        

    

   
    
    target1 = exp_reward + gamma * np.amax(agent1.predict(scaled_next_state)[0])
    target2 = exp_reward + gamma * np.amax(agent2.predict(scaled_next_state)[0])
    
       

    target_f1 = agent1.predict(scaled_state)
    target_f2 = agent2.predict(scaled_state)

    target_f1[0][action1] = target1
    target_f2[0][action1] = target2

    agent1.fit(scaled_state, target_f1,  epochs=1, batch_size = 64, verbose=0)
    agent2.fit(scaled_state, target_f2,  epochs=1, batch_size = 64, verbose=0)

    state = next_state
    #if epsilon > epsilon_min: epsilon *= epsilon_decay
            

    
    histo_Bw.append(max_en_step)
    
    
    th_history_d_1.append(thrghpt_1)
    th_history_c_1.append(thrghpt_2)
    packet_loss_history_d_1.append(next_state[0][3])
    packet_loss_history_c_1.append(next_state[0][1])
    d_history_d_1.append(next_state[0][2])
    d_history_c_1.append(next_state[0][0])
    bw_history_d_1.append(Bw_d)
    bw_history_c_1.append(Bw_c)
    episode_history.append(e)
    








################################### Training 2 ########################################

scaler = StandardScaler()

reward = 0
for e in range(total_episodes):
    state = next_state = np.reshape(np.zeros(4), [1, s_size])
    action3 = action4 = 0
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
        env = Environment()
        state =  np.array(env.current_state(PACKET_SIZE_c, PACKET_SIZE_d, TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d, Bw_c, Bw_d, packets_received_d, packets_received_c)[0]).reshape(1, s_size)
        scaled_state = scaler.fit_transform(state)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * e)
        if np.random.rand() < epsilon:
            action3 = np.random.randint(a_size)
            action4 = np.random.randint(a_size)
        else:
            action3 = np.argmax(agent1.predict(state)[0])
            action4 = np.argmax(agent2.predict(state)[0])

        Bw_c, selected_action_c = env.take_action(action3, Bw_c)
        Bw_d, selected_action_d = env.take_action(action4, Bw_d)
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
    #print(exp_reward)
    rw_history.append(exp_reward)
        

    

   
    
    target1 = exp_reward + gamma * np.amax(agent3.predict(scaled_next_state)[0])
    target2 = exp_reward + gamma * np.amax(agent4.predict(scaled_next_state)[0])
    
       

    target_f1 = agent3.predict(scaled_state)
    target_f2 = agent4.predict(scaled_state)

    target_f1[0][action3] = target1
    target_f2[0][action4] = target2

    agent3.fit(scaled_state, target_f1,  epochs=1, batch_size = 64, verbose=0)
    agent4.fit(scaled_state, target_f2,  epochs=1, batch_size = 64, verbose=0)

    state = next_state
    #if epsilon > epsilon_min: epsilon *= epsilon_decay
            

    
    histo_Bw.append(max_en_step)
    
    
    th_history_d_2.append(thrghpt_1)
    th_history_c_2.append(thrghpt_2)
    bw_history_d_2.append(Bw_d)
    bw_history_c_2.append(Bw_c)
    packet_loss_history_d_2.append(next_state[0][3])
    packet_loss_history_c_2.append(next_state[0][1])
    d_history_d_2.append(next_state[0][2])
    d_history_c_2.append(next_state[0][0])
    
    




################################### Training 3 ########################################

scaler = StandardScaler()

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
        env = Environment()
        state =  np.array(env.current_state(PACKET_SIZE_c, PACKET_SIZE_d, TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d, Bw_c, Bw_d, packets_received_d, packets_received_c)[0]).reshape(1, s_size)
        scaled_state = scaler.fit_transform(state)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * e)
        if np.random.rand() < epsilon:
            action5 = np.random.randint(a_size)
            action6 = np.random.randint(a_size)
        else:
            action5 = np.argmax(agent5.predict(state)[0])
            action6 = np.argmax(agent6.predict(state)[0])

        Bw_c, selected_action_c = env.take_action(action5, Bw_c)
        Bw_d, selected_action_d = env.take_action(action6, Bw_d)
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
    #print(exp_reward)
    rw_history.append(exp_reward)
        

    

   
    
    target1 = exp_reward + gamma * np.amax(agent5.predict(scaled_next_state)[0])
    target2 = exp_reward + gamma * np.amax(agent6.predict(scaled_next_state)[0])
    
       

    target_f1 = agent5.predict(scaled_state)
    target_f2 = agent6.predict(scaled_state)

    target_f1[0][action5] = target1
    target_f2[0][action6] = target2

    agent5.fit(scaled_state, target_f1,  epochs=1, batch_size = 64, verbose=0)
    agent6.fit(scaled_state, target_f2,  epochs=1, batch_size = 64, verbose=0)

    state = next_state
    #if epsilon > epsilon_min: epsilon *= epsilon_decay
            

    
    histo_Bw.append(max_en_step)
    
    
    th_history_d_3.append(thrghpt_1)
    th_history_c_3.append(thrghpt_2)
    bw_history_d_3.append(Bw_d)
    bw_history_c_3.append(Bw_c)
    packet_loss_history_d_3.append(next_state[0][3])
    packet_loss_history_c_3.append(next_state[0][1])
    d_history_d_3.append(next_state[0][2])
    d_history_c_3.append(next_state[0][0])
    

# After training, you can plot the comparison

techniques = ['Adam', 'RMSprop', 'Adagrad']

fig, ax = plt.subplots(2, 3, figsize=(20, 10))  # 2 rows, 4 columns

# Rotate the x-axis labels to prevent overlapping
rotation_angle = 45

# Throughput plot for Data Plane
ax[0, 0].bar(techniques, [np.mean(th_history_d_1), np.mean(th_history_d_2), np.mean(th_history_d_3)], color='#2980b9')
ax[0, 0].set_title('Throughput Comparison Data Plane')
ax[0, 0].set_ylabel('Throughput (Mbps)')
ax[0, 0].tick_params(axis='x', rotation=rotation_angle)

# Packet Loss plot for Data Plane
ax[0, 1].bar(techniques, [np.mean(packet_loss_history_d_1), np.mean(packet_loss_history_d_2), np.mean(packet_loss_history_d_3)], color='#76d7c4')
ax[0, 1].set_title('Packet Loss Comparison Data Plane')
ax[0, 1].set_ylabel('Packet Loss')
ax[0, 1].tick_params(axis='x', rotation=rotation_angle)

# Delay plot for Data Plane
ax[0, 2].bar(techniques, [np.mean(d_history_d_1), np.mean(d_history_d_2), np.mean(d_history_d_3)], color='#45b39d')
ax[0, 2].set_title('Delay Comparison Data Plane')
ax[0, 2].set_ylabel('Delay (ms)')
ax[0, 2].tick_params(axis='x', rotation=rotation_angle)



# Throughput plot for Control Plane
ax[1, 0].bar(techniques, [np.mean(th_history_c_1), np.mean(th_history_c_2), np.mean(th_history_c_3)], color='#85c1e9')
ax[1, 0].set_title('Throughput Comparison Control Plane')
ax[1, 0].set_ylabel('Throughput (Mbps)')
ax[1, 0].tick_params(axis='x', rotation=rotation_angle)

# Packet Loss plot for Control Plane
ax[1, 1].bar(techniques, [np.mean(packet_loss_history_c_1), np.mean(packet_loss_history_c_2), np.mean(packet_loss_history_c_3)], color='#45b39d')
ax[1, 1].set_title('Packet Loss Comparison Control Plane')
ax[1, 1].set_ylabel('Packet Loss')
ax[1, 1].tick_params(axis='x', rotation=rotation_angle)

# Delay plot for Control Plane
ax[1, 2].bar(techniques, [np.mean(d_history_c_1), np.mean(d_history_c_2), np.mean(d_history_c_3)], color='#85c1e9')
ax[1, 2].set_title('Delay Comparison Control Plane')
ax[1, 2].set_ylabel('Delay (ms)')
ax[1, 2].tick_params(axis='x', rotation=rotation_angle)





plt.tight_layout()
plt.savefig(f'comparison_plot_{time_str}_binary_crossentropy.png', bbox_inches='tight')

row_labels = ["Throughput (Mbps)", "Packet Loss", "Delay (ms)"]

# Données du tableau
data_d = [[np.mean(th_history_d_1), np.mean(th_history_d_2), np.mean(th_history_d_3)],  [np.mean(packet_loss_history_d_1), np.mean(packet_loss_history_d_2), np.mean(packet_loss_history_d_3)], [np.mean(d_history_d_1), np.mean(d_history_d_2), np.mean(d_history_d_3)]]


# Création de la figure
fig, ax = plt.subplots()

# Création du tableau
ax.table(cellText=data_d, colLabels=techniques, rowLabels=row_labels, loc='center')

# Masquer les axes
ax.axis('off')

plt.tight_layout()
plt.savefig(f'comparison_table_data_{time_str}_binary_crossentropy.png', bbox_inches='tight')



# Données du tableau
data_c = [[np.mean(th_history_c_1), np.mean(th_history_c_2), np.mean(th_history_c_3)],  [np.mean(packet_loss_history_c_1), np.mean(packet_loss_history_c_2), np.mean(packet_loss_history_c_3)], [np.mean(d_history_c_1), np.mean(d_history_c_2), np.mean(d_history_c_3)]]


# Création de la figure
fig, ax = plt.subplots()

# Création du tableau
ax.table(cellText=data_c, colLabels=techniques, rowLabels=row_labels, loc='center')

# Masquer les axes
ax.axis('off')

plt.tight_layout()
plt.savefig(f'comparison_table_control_{time_str}_binary_crossentropy.png', bbox_inches='tight')