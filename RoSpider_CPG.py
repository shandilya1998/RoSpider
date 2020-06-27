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
                 exp_num,
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
        self.exp_num = exp_num

    def compute_dphase(self):
        self.dphase = self.omega_input
        for i in range(self.num):
            for j in range(self.num):
                self.dphase[i] += self.dt*self.weights[i, j]*self.amplitude[j]*np.sin(self.phase[j]-self.phase[i]-self.phase_diff[i, j])   
    
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
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))  
        for i in range(self.num):
            axes[0].plot(self.t[-1000:], self.Aphase[i, -1000:])
            axes[1].plot(self.t[:1000], self.Aphase[i, :1000])
        axes[0].legend(loc=2)
        axes[1].legend(loc=2)
        fig.savefig('figures/phase_'+str(self.exp_num)+'.png')
    
    def plot_amplitude(self):
        #print(self.Aphase.shape)
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        for i in range(self.num):
            axes[0].plot(self.t[-1000:], self.Aamplitude[i, -1000:])
            axes[1].plot(self.t[:1000], self.Aamplitude[i, :1000])
        colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
        colors = [colormap(i) for i in np.linspace(0, 1,len(axes[0].lines))]
        for i,j in enumerate(axes[0].lines):
            j.set_color(colors[i])
        axes[0].legend(loc=2)
        colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
        colors = [colormap(i) for i in np.linspace(0, 1,len(axes[1].lines))]
        for i,j in enumerate(axes[1].lines):
            j.set_color(colors[i])
        axes[1].legend(loc=2)
        fig.savefig('figures/amplitude_'+str(self.exp_num)+'.png')
    
    def plot_offset(self):
        #print(self.Aphase.shape)
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        for i in range(self.num):
            axes[0].plot(self.t[-1000:], self.Aoffset[i, -1000:])
            axes[1].plot(self.t[:1000], self.Aoffset[i, :1000])
        colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
        colors = [colormap(i) for i in np.linspace(0, 1,len(axes[0].lines))]
        for i,j in enumerate(axes[0].lines):
            j.set_color(colors[i])
        axes[0].legend(loc=2)
        colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
        colors = [colormap(i) for i in np.linspace(0, 1,len(axes[1].lines))]
        for i,j in enumerate(axes[1].lines):
            j.set_color(colors[i])
        axes[1].legend(loc=2)
        fig.savefig('figures/offset_'+str(self.exp_num)+'.png')
    
    def plot_ja(self):
        #print(self.Aphase.shape)
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        for i in range(self.num):
            axes[0].plot(self.t[-1000:], self.out_degree[i, -1000:])
            axes[1].plot(self.t[:1000], self.out_degree[i, :1000])
        colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
        colors = [colormap(i) for i in np.linspace(0, 1,len(axes[0].lines))]
        for i,j in enumerate(axes[0].lines):
            j.set_color(colors[i])
        axes[0].legend(loc=2)
        colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
        colors = [colormap(i) for i in np.linspace(0, 1,len(axes[1].lines))]
        for i,j in enumerate(axes[1].lines):
            j.set_color(colors[i])
        axes[1].legend(loc=2)
        fig.savefig('figures/ja_'+str(self.exp_num)+'.png')
    
    def compute_joint_angles(self): 
        self.ja = 360.0 * ( self.offset + self.amplitude[i] * np.sin(self.phase) )/(2*np.pi)
num = 4
omega = np.array([np.pi, np.pi, np.pi, np.pi]) # np.array([np.pi, np.pi/2, np.pi, np.pi/2]) #np.full((4,), np.pi)
amplitude = np.array([np.pi, np.pi, np.pi, np.pi]) #np.full((4,), np.pi/2)
offset = np.full((4,), 0)
weights = np.ones((num, num))
for i in range(num):
    weights[i][i] = 0.0
phase_diff = np.zeros((num, num))
"""
# the proceeding loop is used to set the phase differences for experiments 0 and 1
for i in range(int(num)):
    for j in range(int(num)):
        if i!=j:
            phase_diff[i][j] = np.pi/2
            phase_diff[j][i] = -np.pi/2
"""
# Setting phase differences for experiment 2
phase_diff[0][1] = np.pi/2
phase_diff[0][2] = 0
phase_diff[0][3] = 0
phase_diff[1][0] = -phase_diff[0][1]
phase_diff[1][2] = phase_diff[0][2] - phase_diff[0][1]
phase_diff[1][3] = phase_diff[0][3] - phase_diff[0][1]
phase_diff[2][0] = -phase_diff[0][2]
phase_diff[2][1] = -phase_diff[1][2]
phase_diff[2][3] = phase_diff[0][3] + phase_diff[0][2]
phase_diff[3][0] = -phase_diff[0][3]
phase_diff[3][1] = -phase_diff[1][3]
phase_diff[3][2] = -phase_diff[2][3]
print(phase_diff)
cpg = Kuramoto(num = num, 
               omega = omega,
               offset = offset,
               amplitude = amplitude,
               phase_diff = phase_diff,
               weights = weights,
               iterations = 1000000,
               exp_num = 3)
cpg.simulate()
cpg.plot_phase()
cpg.plot_amplitude()
cpg.plot_offset()
cpg.plot_ja()
