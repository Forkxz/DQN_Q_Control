from Train import train
import matplotlib.pyplot as plt
from environment import Env,MultiBit
from Net_dql import ReplayMemory,Transition
import numpy as np
import torch
import argparse

def plot_N(N,F,real,img):
    plt.figure(1, figsize = (6, 6))
    plt.title('FIG-2')
    plt.xlabel('N')
    plt.ylabel('F')
    plt.ylim(0,1.2)
    
    plt.plot(N,F,marker='o')

    
    t = np.linspace(0, np.pi * 2, 100)
    s = np.linspace(0, np.pi, 100)

    t, s = np.meshgrid(t, s)
    x = np.cos(t) * np.sin(s)
    y = np.sin(t) * np.sin(s)
    z = np.cos(s)

    px = np.cos((real*2-0.5)*np.pi) * np.sin(-img*np.pi)
    py = np.sin((real*2-0.5)*np.pi) * np.sin(-img*np.pi)
    pz = np.cos(-img*np.pi)

    plt.figure(2, figsize = (6, 6))
    plt.title('FIG-3b')
    subplot3d = plt.subplot(111, projection='3d')
    subplot3d.scatter(px,py,pz,s= 40,c='r')
    subplot3d.plot3D(px,py,pz,c='red',linewidth=4)    #绘制空间曲线
    subplot3d.plot_wireframe(x, y, z, rstride=5, cstride=5 , linewidth=1,color= 'gray')

    plt.figure(3, figsize = (6, 6))
    plt.title('FIG-3a')
    plt.xlabel('N')
    plt.ylabel('J')
    plt.ylim(-1.2,1.2)
    
    plt.plot(real,marker='o')

    plt.show()

def choose_action(DQN_net, observation):
    with torch.no_grad():
        return DQN_net(observation).max(1)[1].view(1, 1)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     
    env = MultiBit() if hyp.multibit else Env(action_space=list(range(2)),dt=np.pi/20) 

    ########### plot path ################
    hyp.N = 20
    memory = ReplayMemory(hyp.memory_size)
    F,net = train(hyp, env)
    state = env.reset()
    output_memory = [state.cpu().numpy()]
    for i in range(20):
        action = choose_action(net,state.to(device))
        next_state, reward, done, fid = env.step(action)
        state = next_state
        output_memory.append(state.cpu().numpy())
        if done:
            break
    output_memory = np.array(output_memory)
    # print(output_memory)

    N = np.linspace(10,50,5).astype(np.int)
    fidelity = []
    for n in N:
        hyp.N = n
        print('\r', 'training N = ', n , end='', flush=True)
        F,_ = train(hyp, env)
        fidelity.append(F)

    plot_N(N,fidelity,output_memory[:,0,-2],output_memory[:,0,-1])