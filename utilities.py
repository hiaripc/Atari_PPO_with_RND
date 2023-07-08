import numpy                        as np
import tensorflow                   as tf
from gymnasium.wrappers.normalize   import RunningMeanStd
from enum                           import Enum

'''
Utilities:
    - EnvWrapper: a simple wrapper that adjust the dims of the state and performs the OR operations between term and trunc
    - Normalizer: a custom wrapper of the RunningMeanStd gymnasium wrapper.
    - OptimizationBatch: creates the "batch" object
    - InfoWriter: used to write the tensorboard's logs
'''
class EnvWrapper(object):
    def __init__(self, super_env,name) -> None:
        self.super_env          = super_env
        self.observation_shape  = self.super_env.observation_space.shape
        self.action_n           = self.super_env.action_space.n
        self.name               = name

    def step(self, action):
        next_state, ext_rwd, term, trunc, _ = self.super_env.step(action)
        
        # Transpose the state from channel first to channel last to be used with keras (n_envs,4,84,84) -> (n_envs,84,84,4)
        next_state = np.expand_dims(next_state, -1)
        
        # Done is the or between term and truc, the general end of an episode
        done = np.maximum(term, trunc)

        return next_state, ext_rwd, done

    def reset_ended(self, idxs):
        state, _ = self.super_env.reset(idxs)
        state = np.expand_dims(state, -1)
        return state
    
    def reset(self):
        state, _ = self.super_env.reset()
        state = np.expand_dims(state, -1)
    
        return state
    


class Normalizer(object): 
    def __init__(self, category, shape=()) -> None:
        self.norm = RunningMeanStd(shape=shape)
        self.category = category
        
    def update(self, obs):
        if self.category=="state":
            obs = self.__reshape(obs)
        self.norm.update(obs)

    def update_reward(self, continous_int_rwds):
        # Continous_int_rwds (steps,nenvs)
        mean    = np.mean(continous_int_rwds)
        var     = np.var(continous_int_rwds)
        count   = np.size(continous_int_rwds)
        self.norm.update_from_moments(mean, var, count)
        

    def __rwd_norm(self, int_rwd):
        return int_rwd/np.sqrt(self.norm.var + 1e-8)
    
    def __reshape(self, state):
        # From (nenvs,4,84,84,1) picking last frame only to (nenvs,84,84,1), 
        state = state[:,3,:,:,:] 
        # Add a new axis to be (nevns,1,84,84,1)
        state = np.expand_dims(state, axis=1)
        return state

    def __state_norm(self, state):
        state = self.__reshape(state)
        state = np.clip((state - self.norm.mean)/np.sqrt(self.norm.var + 1e-8), -5.0, 5.0)

        return state
    
    def __call__(self, obs):
        if self.category =="rwd":
            return self.__rwd_norm(obs)
        elif self.category =="state":
            return self.__state_norm(obs)
             


class OptimizationBatch(object):
    def __init__(self, batch_size, num_envs, state_shape) -> None:
        self.values_shape = (batch_size,num_envs) 
        self.states_shape = self.values_shape + state_shape
        dims = np.arange(len(self.states_shape))
        self.minib_states_shape = (dims[1], dims[0], *dims[2:])
        self.reset()

    
    def append(self, state, action, actions_logit, ext_rwd, int_rwd, ext_value, int_value, done):
        
        self.states[self.t]         = state
        self.actions[self.t]        = np.squeeze(action)
        self.actions_logit[self.t]  = actions_logit
        self.ext_rwds[self.t]       = ext_rwd
        self.int_rwds[self.t]       = int_rwd
        self.ext_values[self.t]     = np.squeeze(ext_value)
        self.int_values[self.t]     = np.squeeze(int_value)
        self.dones[self.t]          = done
        self.t += 1


    #@profile
    def resize_for_minibatch(self):

        # Reshape from (steps, n_envs, state_shape) -> (steps*n_envs, state_shape)
        self.states         = self.states.transpose(self.minib_states_shape).reshape((-1, *self.states_shape[2:]))
    
        # Reshape from (steps, n_envs)  -> (steps * nevns)    
        self.actions        = self.actions.transpose([1,0]).reshape(-1)
        self.actions_logit  = self.actions_logit.transpose([1,0]).reshape(-1)
        self.advs           = self.advs.transpose([1,0]).reshape(-1)
        self.returns_ext    = self.returns_ext.transpose([1,0]).reshape(-1)
        self.returns_int    = self.returns_int.transpose([1,0]).reshape(-1)


    def get_minibatch(self, idxs):    
        minib_states        = self.states[idxs]  
        minib_actions       = self.actions[idxs]
        minib_actions_logit = self.actions_logit[idxs]
        minib_advs          = self.advs[idxs]
        minib_returns_ext   = self.returns_ext[idxs]
        minib_returns_int   = self.returns_int[idxs]

        return minib_states, minib_actions, minib_actions_logit, minib_advs, minib_returns_ext, minib_returns_int 


    def add_advs_returns(self, advs, returns_ext, returns_int):  
        self.advs        = advs   
        self.returns_ext = returns_ext
        self.returns_int = returns_int


    def reset(self):
       
        # Float 32 in order to be compatible with standard Tensors dtype    
        self.states         = np.zeros(self.states_shape, dtype=np.float32)
        self.actions        = np.zeros(self.values_shape, dtype=np.int32)
        self.actions_logit  = np.zeros(self.values_shape, dtype=np.float32)
        self.ext_rwds       = np.zeros(self.values_shape, dtype=np.float32)
        self.int_rwds       = np.zeros(self.values_shape, dtype=np.float32)
        self.ext_values     = np.zeros(self.values_shape, dtype=np.float32)
        self.int_values     = np.zeros(self.values_shape, dtype=np.float32)
        self.dones          = np.zeros(self.values_shape, dtype=np.float32)
        self.t = 0


class InfoStats(Enum):
        ext_rwd       = 0
        last_int_rwd  = 1
        last_ext_rwd  = 2
        rll_int_rwd   = 3
        returns_ext   = 4
        returns_int   = 5
        advs          = 6
        ## Train stats: (from index 7)
        entropy       = 7
        loss_actor    = 8
        loss_critic   = 9
        loss_policy   = 10
        r_theta       = 11
        loss_rnd      = 12


class InfoWriter(object):
    def __init__(self, log_dir, num_envs):
        self.log_dir = log_dir
        self.infos_mean = np.zeros((len(InfoStats)))
        self.count_ext_rwds = np.zeros((num_envs))
        self.count_int_rwds = np.zeros((num_envs))
        # Needed in order to remember last 100 rewards, calculating the mean with them.
        self.rwds_t         = 0
        self.last_100       = np.zeros(100)
        self.mean_rll_ext_rwd_ended    = 0
        self.mean_rll_int_rwd_ended    = 0

        ## Train stats: (from index 7)
        self.train_start = 7
    

    def log_seed(self, seed, infos):
        log_dir_seed = self.log_dir + f"_{seed}_{infos}"
        file_writer = tf.summary.create_file_writer(log_dir_seed + "/metrics")
        file_writer.set_as_default()


    def incremental_mean(self, step, stat, new_value):
        stat_idx = InfoStats[stat].value
        self.infos_mean[stat_idx] += 1/(step) * (new_value - self.infos_mean[stat_idx])


    def reset_train_infos(self):
        self.infos_mean[self.train_start:] = 0


    def reset(self, stat):
        stat_idx = InfoStats[stat].value
        self.infos_mean[stat_idx] = 0


    def incremental_mean_train(self, step, new_value):
        for num in range(self.train_start, len(InfoStats)):
            stat = InfoStats(num)
            self.incremental_mean(step, stat.name, new_value[num-self.train_start])         


    def write_infos(self, step):      
        for num in range(len(InfoStats)):
            stat = InfoStats(num)
            tf.summary.scalar(stat.name, self.infos_mean[stat.value], step)


    def get(self, stat):
        stat_idx = InfoStats[stat].value
        return self.infos_mean[stat_idx]


    def update_advs_returns(self, advs, returns_ext, returns_int):
        self.infos_mean[InfoStats["advs"].value]        = np.mean(advs)
        self.infos_mean[InfoStats["returns_ext"].value] = np.mean(returns_ext)
        self.infos_mean[InfoStats["returns_int"].value] = np.mean(returns_int)


    def update_last_reward(self):
        self.infos_mean[InfoStats["last_ext_rwd"].value] = self.mean_rll_ext_rwd_ended
        self.mean_rll_ext_rwd_ended = 0

        self.infos_mean[InfoStats["last_int_rwd"].value] = self.mean_rll_int_rwd_ended
        self.mean_rll_int_rwd_ended = 0


    def update_ended(self, idx, ended_this_play):
        episode_ext_rwd = self.count_ext_rwds[idx]
        episode_int_rwd = self.count_int_rwds[idx]

        self.last_100[self.rwds_t] = episode_ext_rwd
        if self.rwds_t == 99:
            self.rwds_t = 0
        else: self.rwds_t += 1

        self.infos_mean[InfoStats["ext_rwd"].value] = np.mean(self.last_100)

        self.mean_rll_ext_rwd_ended += 1/(ended_this_play) * (episode_ext_rwd - self.mean_rll_ext_rwd_ended)
        self.mean_rll_int_rwd_ended += 1/(ended_this_play) * (episode_int_rwd - self.mean_rll_int_rwd_ended)

        self.count_ext_rwds[idx] = 0
        self.count_int_rwds[idx] = 0  
    
    
    def update_ext_rewards(self, ext_rwd, int_rwd):
        self.count_ext_rwds += ext_rwd
        self.count_int_rwds += np.asarray(int_rwd)

    def update_int_rewards(self, int_rwd):
        self.infos_mean[InfoStats["rll_int_rwd"].value] = np.mean(int_rwd)