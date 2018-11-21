__all__ = ['Monitor', 'get_monitor_files', 'load_results']

import gym
from gym.core import Wrapper
import time
from glob import glob
import csv
import os.path as osp
import json
import numpy as np

# Ignore warnings
np.seterr(divide='ignore', invalid='ignore')

class Monitor(Wrapper):
    EXT = "monitor.csv"
    f = None

    def __init__(self, env, filename, reset_keywords=(), info_keywords=()):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if osp.isdir(filename):
                    filename = osp.join(filename, Monitor.EXT)
                else:
                    filename = filename + "." + Monitor.EXT
            self.f = open(filename, "wt")
            self.f.write('#%s\n'%json.dumps({"t_start": self.tstart, 'env_id': env.spec and env.spec.id}))
            self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't', 'latency')+reset_keywords+info_keywords)
            self.logger.writeheader()
            self.f.flush()
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.rewards = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_hits = []
        self.max_reward = 0
        self.tfirst_hit = 0
        self.current_reset_info = {} # extra info about the current episode, that was passed in during reset()
        self.tstart_hit = time.time()

    def reset(self, **kwargs):
        self.rewards = []
        self.episode_hits = []
        self.tstart_hit = time.time()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def initial(self):
        return self.env.initial()

    def step(self, action):
        rew, done, ob = self.env.step(action)
        self.rewards.append(rew)
        self.max_reward = max(rew, self.max_reward)
        if rew == self.max_reward:
            self.episode_hits.append(time.time() - self.tstart_hit)
        if done:
            self.tfirst_hit = self.episode_hits[0] if self.episode_hits else 0
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6),
                      "latency": round(self.tfirst_hit/np.mean(self.episode_hits), 6)}
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            epinfo.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(epinfo)
                self.f.flush()
            self.reset()
        return rew, done, ob

    def close(self):
        if self.f is not None:
            self.f.close()
        self.env.close()

class LoadMonitorResultsError(Exception):
    pass

def get_monitor_files(dir):
    return glob(osp.join(dir, "*" + Monitor.EXT))

def load_results(dir):
    import pandas
    monitor_files = (
        glob(osp.join(dir, "*monitor.json")) + 
        glob(osp.join(dir, "*monitor.csv"))) # get both csv and (old) json files
    if not monitor_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (Monitor.EXT, dir))
    dfs = []
    headers = []
    for fname in monitor_files:
        with open(fname, 'rt') as fh:
            if fname.endswith('csv'):
                firstline = fh.readline()
                assert firstline[0] == '#'
                header = json.loads(firstline[1:])
                df = pandas.read_csv(fh, index_col=None)
                headers.append(header)
            elif fname.endswith('json'): # Deprecated json format
                episodes = []
                lines = fh.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                df = pandas.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            df['t'] += header['t_start']
        dfs.append(df)
    df = pandas.concat(dfs)
    df.sort_values('t', inplace=True)
    df.reset_index(inplace=True)
    df['t'] -= min(header['t_start'] for header in headers)
    df.headers = headers # HACK to preserve backwards compatibility
    return df

def test_monitor():
    env = gym.make("CartPole-v1")
    env.seed(0)
    mon_file = "/tmp/baselines-test-%s.monitor.csv" % uuid.uuid4()
    menv = Monitor(env, mon_file)
    menv.reset()
    for _ in range(1000):
        _, _, done, _ = menv.step(0)
        if done:
            menv.reset()

    f = open(mon_file, 'rt')

    firstline = f.readline()
    assert firstline.startswith('#')
    metadata = json.loads(firstline[1:])
    assert metadata['env_id'] == "CartPole-v1"
    assert set(metadata.keys()) == {'env_id', 'gym_version', 't_start'},  "Incorrect keys in monitor metadata"

    last_logline = pandas.read_csv(f, index_col=None)
    assert set(last_logline.keys()) == {'l', 't', 'r'}, "Incorrect keys in monitor logline"
    f.close()
    os.remove(mon_file)