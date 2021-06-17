import numpy as np
from scipy.linalg import expm
import torch
import itertools

class Env( object):
    def __init__(self,
        action_space=[0,1,2],
        dt=0.1):
        super(Env, self).__init__()
        self.action_space = action_space
        self.n_actions = len(self.action_space)
        self.n_features = 4
        self.state = np.array([1,0,0,0])
        self.nstep=0 
        self.dt=dt

    def reset(self):

        self.state = np.array([1,0,0,0])
        self.nstep = 0 

        return torch.tensor([self.state]).float()

    def step(self, action):


        psi = np.array([self.state[0:int(len(self.state) / 2)] + self.state[int(len(self.state) / 2):int(len(self.state))] * 1j])
        psi = psi.T
        psi=np.mat(psi)

        J = 4  # control field strength
        sx = np.mat([[0, 1], [1, 0]], dtype=complex)
        sz = np.mat([[1, 0], [0, -1]], dtype=complex)

        # U = np.matrix(np.identity(2, dtype=complex)) 
        
        H =  J *float(action)/(self.n_actions-1)* sz + 1 * sx
        U = expm(-1j * H * self.dt) 


        psi = U * psi  # final state
        target = np.mat([[0], [1]], dtype=complex) 
        err = 1 - (np.abs(psi.H * target) ** 2).item(0).real  
        # err = sum(abs(psi-target)).item(0)
        rwd = 10 * (err<0.5)+100 * (err<0.1)+5000*(err < 10e-3)   

        done =( (err < 10e-3) or self.nstep>=np.pi/self.dt ) 
        self.nstep +=1  

        psi=np.array(psi)
        psi_T = psi.T
        self.state = np.array(psi_T.real.tolist()[0] + psi_T.imag.tolist()[0])

        return torch.tensor([self.state]).float(), torch.tensor([[rwd]]).float(), done, 1 - err




SPIN_NUM = 8
MAG = 2*40
COUPLING = 2*1
DT = 0.05*(SPIN_NUM-1)*np.pi*0.5

def mg_config(x,dim):
    if dim>1:
        Y=[]
        for ii in range(2):
            for xx in x:
                y=xx+[ii]
                Y.append(y)
        Y=mg_config(Y,dim-1)
    else:
        Y=x
    return Y

mag=MAG*np.array(mg_config([[0],[1]],SPIN_NUM))

class MultiBit(object):
    def __init__(self):
        super(MultiBit, self)
        self.action_space = mag
        self.n_actions = len(self.action_space)
        self.n_features = SPIN_NUM*2
        

    
    def reset(self):
        psi = [0 for i in range(SPIN_NUM)]
        psi[0] = 1

        self.state = np.array([str(i) for i in psi])
        self.state = np.array(list(itertools.chain(*[(i.real, i.imag) for i in psi])))

        return torch.tensor([self.state]).float()
    
    def step(self, actionnum):
        actions = self.action_space[actionnum[0]]
        ham = np.diag([COUPLING for i in range(SPIN_NUM-1)], 1)*(1-0j) + np.diag([COUPLING for i in range(SPIN_NUM-1)], -1)*(1+0j) + np.diag(actions)

        statess = [complex(self.state[2*i], self.state[2*i+1]) for i in range(SPIN_NUM)]
        statelist = np.transpose(np.mat(statess))
        next_state = expm(-1j*ham*DT)*statelist
        fidelity = (abs(next_state[-1])**2)[0,0]
        
        xi=0.999
        if fidelity<xi:
            reward = fidelity *10

        doned = False
        if fidelity >= xi:
            reward = 2500 
            doned = True
            
        #reward = reward*(0.95**stp)

        next_states = [next_state[i,0] for i in range(SPIN_NUM)]
        next_states = np.array(list(itertools.chain(*[(i.real, i.imag) for i in next_states])))

        self.state = next_states
        return torch.tensor([next_states]).float(), torch.tensor([reward]).float(), doned, fidelity