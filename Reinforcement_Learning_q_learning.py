from environment import MountainCar
import sys
import numpy as np

def main(args):
    mode = str(sys.argv[1])
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    num_episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    discount_factor = float(sys.argv[7])
    learning_rate = float(sys.argv[8])

    #define function for performing greedy search for picking action
    def greedy(state, weight, action_space):
        Q_list = []
        for each in range(0, (action_space)):
            Q = 0
            for k, v in state.items():
                Q += v*weight[k, each]
            Q += b
            Q_list.append(Q)
        a = np.argmax(Q_list)
        max_Q = max(Q_list)
        return Q, a, max_Q

    #define function to calculate q after selecting action
    def q_calc(state, weight, a, b):
        q = 0
        for k, v in state.items():
            q += v*weight[k, a]
        q += b
        return q

    #define function to update the weights
    def update(state, action_space, weight, learning_rate, q, reward, discount_factor, max_Q):
        for each in range(0, (action_space)):
            for k,v in state.items():
                if each == a:
                    weight[k, each] = weight[k, each] - (learning_rate * ((q - (reward + (discount_factor*max_Q)))))*v
        return weight

    env = MountainCar(mode)                                     #call the environment
    weight = np.zeros((env.state_space, env.action_space))      #initialize weights
    b = 0                                                       #initialize bias
    returns_out = open(sys.argv[3], 'w')
    for e in range(0, num_episodes):                            #iterating over the number of episodes
        env.reset()                                             #reset
        reward = 0                                              #initialize reward
        for it in range(0, max_iterations):                     #iterating over number of max iterations
            state = env.state                                   #initialize state
            state = env.transform(state)                        #transform to dictionary
            action_space = env.action_space                     #call action space
            probabilty = np.random.uniform(0.0, 1.0)
            if probabilty < epsilon:
                a = np.random.randint(0, 3)                     #random search for a
            else:
                _, a, _ = greedy(state, weight, action_space)   #greedy search for a
            s_next, reward_next, done = env.step(a)             #compute the next state, reward for chosen action. If done = TRUE, stop.
            reward = reward + reward_next                       #update reward
            q = q_calc(state, weight, a, b)                     #calculate q for the chosen action(a)
            _, a_next, max_Q = greedy(s_next, weight, action_space) #calculate max_Q for the next state
            weight = update(state, action_space, weight, learning_rate, q, reward_next, discount_factor, max_Q) #update weights
            b = b - (learning_rate * (q - (reward_next + (discount_factor*max_Q)))) #update bias
            if done:
                break                                           #break when done = TRUE
        returns_out.write(str(reward)+"\n")                     #print rewards for each episode

    output_list = []
    output_list.append(b)
    for w in weight:
        for each in w:
            output_list.append(each)
    with open(sys.argv[2], 'w') as f:
        for item in output_list:
            f.write("%s\n" % item)                              #print final bias and weights
    pass

if __name__ == "__main__":
    main(sys.argv)
