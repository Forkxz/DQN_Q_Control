import torch
from environment import Env,MultiBit
from Net_dql import DQN,ReplayMemory,choose_action,Transition
import numpy as np
import argparse
import torch.optim as optim

def train(hyp, env):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(env.n_actions, env.n_features).to(device)
    target_net = DQN(env.n_actions, env.n_features).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters(),lr=hyp.learning_rate)
    fid_10 = 0
    learn_step_counter = 0
    step = 0
    memory = ReplayMemory(hyp.memory_size)
    epsilon = 0 if hyp.e_greedy_increment is not None else hyp.epsilon_max
    learn = False
    for episode in range(hyp.epsilon_max):
        state = env.reset()

        for i in range(hyp.N):
            action = choose_action(policy_net,state.to(device),epsilon,env.n_actions)
            next_state, reward, done, fid = env.step(action)
            memory.push(state, action, next_state, reward)
            
            if (step > 500) and (step%5 == 0):
                learn = True
            else:
                learn = False

            if learn:
                # check to replace target parameters
                if learn_step_counter % hyp.replace_target_iter == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                transitions = memory.sample(hyp.batch_size)
                batch = Transition(*zip(*transitions))

                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None]).to(device)

                state_batch = torch.cat(batch.state).to(device)
                action_batch = torch.cat(batch.action).to(device)
                reward = torch.cat(batch.reward).to(device)

                state_action_values = policy_net(state_batch).gather(1, action_batch)
                next_state_values = torch.zeros(hyp.batch_size, device=device)
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
                expected_state_action_values = reward.view(hyp.batch_size,1) + hyp.reward_decay * next_state_values.view(hyp.batch_size,1)

                # Compute Huber loss
                criterion = torch.nn.SmoothL1Loss()
                loss = criterion(state_action_values, expected_state_action_values)
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

                epsilon = epsilon + hyp.e_greedy_increment if epsilon < hyp.epsilon_max else hyp.epsilon_max
                learn_step_counter += 1

            state = next_state
            step +=1
            # print('\r', step, end='', flush=True)        
            if done:
                break

        if episode > hyp.epsilon_max-10:
            fid_10 = max(fid_10,fid)


    return fid_10,policy_net
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon_max', type=int, default=500, help='epsilon max number, larger number will lead to better result ')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--reward_decay', type=float, default=0.9)
    parser.add_argument('--e_greedy', type=float, default=0.99)
    parser.add_argument('--replace_target_iter', type=int, default=200)
    parser.add_argument('--memory_size', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--N', type=int, default=20)
    parser.add_argument('--e_greedy_increment', type=float, default=0.001)
    parser.add_argument('--multibit',default=False, help='use multibit control if true')
    hyp = parser.parse_args()
         
    env = MultiBit() if hyp.multibit else Env(action_space=list(range(2)),dt=np.pi/20) 
    fidelity,_ = train(hyp, env)

    print("Final_fidelity=", fidelity)
        


