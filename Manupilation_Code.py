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
        #PACKET_SIZE_d = random.uniform(1024, 1518)
        #PACKET_SIZE_c = random.uniform(100, 300)
        PACKET_SIZE_d = 1518
        PACKET_SIZE_c = 300
        packets_received_d = random.uniform(1000, 5000)
        packets_received_c = random.uniform(10, 30)
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
    
    
    th_history_d.append(thrghpt_1)
    th_history_c.append(thrghpt_2)
    bw_history_d.append(Bw_d)
    bw_history_c.append(Bw_c)
    in_history_d.append(count_stat_d['increase_bandwidth'])
    de_history_d.append(count_stat_d['decrease_bandwidth'])
    in_history_c.append(count_stat_c['increase_bandwidth'])
    de_history_c.append(count_stat_c['decrease_bandwidth'])
    episode_history.append(e)
    #print( e, Counter(histo_selected_action))



max_throughput = max(max(th_history_d), max(th_history_c))
scaled_th_history_d = [th / max_throughput for th in th_history_d]
scaled_th_history_c = [th / max_throughput for th in th_history_c]
one_value_list = [1] * total_episodes



print("Plot Learning Performance")
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})



fig, ax = plt.subplots(figsize=(10,4))
plt.grid(True, linestyle='--')
plt.title('Learning Performance')
plt.plot( range(total_episodes), in_history_d, label='Agent_in_1', marker="", linestyle="-")#, color='k')
plt.plot( range(total_episodes), in_history_c, label='Agent_in_2', marker="", linestyle="-")
plt.plot( range(total_episodes), de_history_d, label='Agent_de_1', marker="", linestyle="-")#, color='k')
plt.plot( range(total_episodes), de_history_c, label='Agent_de_2', marker="", linestyle="-")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(prop={'size': 12})

plt.savefig('learning_multiagent_Agent_Performance_DDQN.pdf', bbox_inches='tight')



# Plot accuracy
plt.figure(figsize=(10, 4))
plt.grid(True, linestyle='--')
plt.title('Accuracy')
plt.plot(range(total_episodes), scaled_th_history_d, label='Accuracy_Agent_1', marker="", linestyle="-")
plt.plot(range(total_episodes), scaled_th_history_c, label='Accuracy_Agent_2', marker="", linestyle="-")
plt.plot(range(total_episodes), one_value_list, label='Target_Accuracy', marker="", linestyle="-")
plt.xlabel('Episode')
plt.ylabel('Accuracy')
plt.legend(prop={'size': 12})
plt.savefig('accuracy_plot_DDQN.pdf', bbox_inches='tight')
#plt.show()



#rewarf value
plt.figure(figsize=(10, 4))
plt.grid(True, linestyle='--')
plt.title('Reward')
plt.plot(range(total_episodes), rw_history, label='Accuracy_Agent_1', marker="", linestyle="-")
plt.xlabel('Episode')
plt.ylabel('reward')
plt.legend(prop={'size': 12})
plt.savefig('reward.pdf', bbox_inches='tight')



# display a plot for bandwidth variation with throughput for both agent 
# how many time the agent increase the bandwith to get the purpose of our model
# compare number of increase and descrease our model make --> if in most case decrease we can say that the model optimimal solution.
# use decreasing badwidth as reward function
# the model depend on the first action take, he keep priorizing the first action, which my not be efficent when the network try to just increase the bandwidth
# we have try to increase the exploration to make the model explore the other solutions
# we can make a reward function based on the number of decrease bandwidth getting in each episode
# add reward function based on the getting result
# add more parametre to the state, and tore them
# start the redaction, defining the model, design of the state space, action space, and reward function
# use the fact that storing the packet size could be benefical if we want to use MTU for network optimization
# fixed the packet size for maximum and minmun and see the optimal solution or action


# training instable
# increase the batch size no improvement
# change the optimizer
# change the loss function
# i have added accuracy but it make the system instable

# normalize the input data
