import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

class Kuramoto(object):
    def __init__(self,
                 num,
                 omega,
                 amplitude,
                 offset,
                 phase_diff,
                 weights,
                 iterations = 10000000, 
                 dt = 0.01,
                 ax = 2,
                 ar = 2):
        """
            parameters:
                -   omega: 
                    -   array of natural frequencies for each oscillator
                    -   shape: 
                        -   (n,)
                -   amplitude:
                    -   array of amplitudes for each oscillator
                    -   shape:
                        -   (n,)
                -   offset:
                    -   array of offset for each oscillator
                    -   shape:
                        -   (n,)
                -   phase_diff:
                    -   array of desired phase offset for each oscillator
                    -   shape: 
                        -   (n, n)
                -   weights:
                    -   array of desired phase offset for each oscillator
                -   iterations:
                    -   number of training iterations
                -   num:
                    -   number of oscillators
                -   dt:
                    -   value of dt for integration
                -   ar:
                    -   floating point number representing amplitude gain
                -   ax:
                    -   floating point number representing offset gain
        """
        self.n = iterations
        self.num = num
        self.omega_input = omega
        self.dt = dt 
        self.phase_diff = phase_diff
        self.weights = weights
        self.amplitude_input = amplitude
        self.offset_input = offset
        self.aoffset = ax
        self.aamplitude = ar
        self.Aamplitude = np.zeros((self.num, self.n))
        self.Aoffset = np.zeros((self.num, self.n))
        self.Aphase = np.zeros((self.num, self.n))
        self.phase = np.zeros((self.num,))
        self.amplitude = np.zeros((self.num,))
        self.offset = np.zeros((self.num,))
        self.dphase = np.zeros((self.num,))
        self.damplitude = np.zeros((self.num,))
        self.doffset = np.zeros((self.num,))
        self.d2phase = np.zeros((self.num,))
        self.d2amplitude = np.zeros((self.num,))
        self.d2offset = np.zeros((self.num,))
        self.t = np.arange(0,self.n)*self.dt
        self.ja = np.zeros((self.num,))
        self.out_degree = np.zeros((self.num, self.n))

    def compute_dphase(self):
        self.dphase = self.omega_input
        for j in range(self.num):
            self.dphase += self.dt*self.weights[:, j]*self.amplitude*np.sin(self.phase[j]-self.phase-self.phase_diff[:, j])   
    
    def compute_phase(self):
        self.phase += self.dphase*self.dt
    
    def compute_amplitude(self):
        self.amplitude += self.damplitude*self.dt

    def compute_damplitude(self):   
        self.damplitude += self.dt*self.d2amplitude
    
    def compute_d2amplitude(self):    
        self.d2amplitude = self.aamplitude*((self.aamplitude/4)*(self.amplitude_input-self.amplitude)-self.damplitude)

    def compute_offset(self):
        self.offset += self.doffset*self.dt

    def compute_doffset(self):
        self.doffset += self.dt*self.d2offset
    
    def compute_d2offset(self):    
        self.d2offset = self.aoffset*((self.aoffset/4)*(self.offset_input-self.offset)-self.doffset)

    def simulate(self): 
        for i in tqdm(range(self.n)):
            self.Aphase[:, i] = self.phase
            self.Aamplitude[:, i] = self.amplitude
            self.Aoffset[:, i] = self.offset
            self.out_degree[:, i] = self.ja
            self.compute_dphase()
            self.compute_phase()
            self.compute_d2amplitude()
            self.compute_damplitude()
            self.compute_amplitude()
            self.compute_d2offset()
            self.compute_doffset()
            self.compute_offset()
            self.compute_joint_angles()

    def set_conn_weights(self, weights):
        self.weights = weights

    def set_phase_diff(self, phase_diff):
        self.phase_diff = phase_diff
    
    def plot_phase(self):
        #print(self.Aphase.shape)
        fig, axes = plt.subplots(self.num, 1, figsize=(10, self.num*10))  
        for i in range(self.num):
            axes[i].plot(self.t, self.Aphase[i])
            axes[i].set_title('neuron ' + str(i))
        fig.savefig('figures/phase.png')
    
    def plot_amplitude(self):
        #print(self.Aphase.shape)
        fig, axes = plt.subplots(self.num, 1, figsize=(10, self.num*10))
        for i in range(self.num):
            axes[i].plot(self.t, self.Aamplitude[i])
            axes[i].set_title('neuron ' + str(i))
        fig.savefig('figures/amplitude.png')
    
    def plot_offset(self):
        #print(self.Aphase.shape)
        fig, axes = plt.subplots(self.num, 1, figsize=(10, self.num*10))
        for i in range(self.num):
            axes[i].plot(self.t, self.Aoffset[i])
            axes[i].set_title('neuron ' + str(i))
        fig.savefig('figures/offset.png')
    
    def plot_ja(self):
        #print(self.Aphase.shape)
        fig, axes = plt.subplots(self.num, 1, figsize=(10, self.num*10))
        for i in range(self.num):
            axes[i].plot(self.t[-1000:], self.out_degree[i, -1000:])
            axes[i].set_title('neuron ' + str(i))
        fig.savefig('figures/ja.png')
    
    def compute_joint_angles(self): 
        self.ja = 360.0 * ( self.offset + self.amplitude[i] * np.sin(self.phase) )/(2*np.pi)
num = 4
omega = np.full((4,), np.pi)
amplitude = np.full((4,), np.pi/2)
offset = np.full((4,), 0)
weights = np.ones((num, num))
for i in range(num):
    weights[i][i] = 0.0
phase_diff = np.zeros((num, num))
for i in range(int(num/2)):
    for j in range(int(num/2)):
        if i!=j:
            phase_diff[i][j] = np.pi/2
            phase_diff[j][i] = -np.pi/2
cpg = Kuramoto(num = num, 
               omega = omega,
               offset = offset,
               amplitude = amplitude,
               phase_diff = phase_diff,
               weights = weights,
               iterations = 1000000)
cpg.simulate()
cpg.plot_phase()
cpg.plot_amplitude()
cpg.plot_offset()
cpg.plot_ja()
