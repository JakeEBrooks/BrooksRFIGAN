import numpy as np

#wideband can be anywhere between 1-30s long. Probability of wideband would be exponential such that
def get_wideband_duration_ps(max_duration: int = 30, rate = 0.5):
    ps = rate*np.exp(- rate*np.arange(0,max_duration,dtype=float)) # lower rate means more likely to be longer
    ps /= np.sum(ps)
    return ps
#narrow band can be any duration within obs bounds, with linear probability such that
def get_narrowband_duration_ps(duration, rate = 1):
    ps = rate*np.arange(0,duration,dtype=float) # higher rate means more likely to be longer
    ps /= np.sum(ps)
    return ps

class RFIEvent():
    def __init__(self, starttime: int, duration: int, wideband: bool, strength: float, channel: int = None):
        self.starttime = starttime
        self.duration = duration
        self.wideband = wideband
        self.strength = strength
        if not self.wideband and channel != None:
            self.channel = channel

    def activenow(self, t):
        if t >= self.starttime and t <= (self.starttime+self.duration):
            return True
        else:
            return False

class RFISimulation():
    def __init__(self, observation_duration: int, simulation_duration_multi: int = 2, wb_p = 0.05, rfi_p=0.01, freq_ps = None, channels=1024, rfi_strength=1, rfi_strength_sig=0.3):
        self.obs_duration = observation_duration
        self.sim_duration_multi = simulation_duration_multi
        self.sim_duration = self.sim_duration_multi*self.obs_duration
        self.channel_num = channels
        self.rfi_p = rfi_p
        self.wb_p = wb_p
        self.rfi_strength = rfi_strength
        self.rfi_strength_sig = rfi_strength_sig
        if np.all(freq_ps) == None:
            self.freq_ps = np.ones(self.channel_num)/self.channel_num
        else:
            self.freq_ps = freq_ps
        assert len(self.freq_ps) == channels
        self.wb_duration_ps = get_wideband_duration_ps()
        self.nb_duration_ps = get_narrowband_duration_ps(self.sim_duration)

        self.rng = np.random.default_rng()
    
    def sim_events(self):
        rfi_events = np.array([])
        for t in np.arange(self.sim_duration):
            p = self.rng.uniform()
            if p <= self.rfi_p:
                p = self.rng.uniform()
                str = self.rng.normal(loc=self.rfi_strength, scale=self.rfi_strength_sig)
                if p <= self.wb_p: # if rfi is wb
                    rfi_events = np.append(rfi_events, RFIEvent(starttime=t, duration=self.rng.choice(30, p=self.wb_duration_ps), wideband=True, strength=str))
                else: # if rfi is not wb
                    rfi_events = np.append(rfi_events, RFIEvent(starttime=t, duration=self.rng.choice(self.sim_duration, p=self.nb_duration_ps), wideband=False, strength=str, channel=self.rng.choice(self.channel_num, p=self.freq_ps)))
        return rfi_events
    
    def make_sim(self, rfi_events):
        rfi = np.zeros((self.obs_duration, self.channel_num))
        rfi_truth = np.zeros((self.obs_duration, self.channel_num)).astype(bool)
        for t in np.arange((self.sim_duration)//4,(3*self.sim_duration)//4):
            tnorm = t - (self.sim_duration)//4
            for ev in rfi_events:
                if ev.activenow(t):
                    if ev.wideband:
                        rfi[tnorm,:] += ev.strength
                        rfi_truth[tnorm,:] += True
                    else:
                        rfi[tnorm,ev.channel] += ev.strength
                        rfi_truth[tnorm, ev.channel] += True
        return rfi, rfi_truth

def make_rfi_images(obs_duration, imnum: int, sim_duration_multi: int = 2, rfi_p = 0.01, wb_p = 0.05, freq_ps = None, channels = 1024, rfi_strength = 1, rfi_strength_sig = 0.3):
    sim = RFISimulation(obs_duration, sim_duration_multi, wb_p=wb_p, rfi_p=rfi_p, freq_ps=freq_ps, channels=channels, rfi_strength=rfi_strength)
    rfi, truthmask = sim.make_sim(sim.sim_events())
    if imnum > 1:
        rfi = rfi[:,:,np.newaxis]
        truthmask = truthmask[:,:,np.newaxis]
        for i in np.arange(1,imnum):
            sim = RFISimulation(obs_duration, sim_duration_multi, wb_p=wb_p, rfi_p=rfi_p, freq_ps=freq_ps, channels=channels, rfi_strength=rfi_strength, rfi_strength_sig=rfi_strength_sig)
            ims, truths = sim.make_sim(sim.sim_events())
            rfi = np.append(rfi, ims[:,:,np.newaxis], axis=2)
            truthmask = np.append(truthmask, truths[:,:,np.newaxis], axis=2)
        return rfi, truthmask
    else:
        return rfi, truthmask