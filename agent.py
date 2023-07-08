import numpy                    as np
import tensorflow               as tf
from tensorflow                 import keras
from keras.layers               import Conv2D, Dense, Input, Flatten
import tensorflow_probability   as tfp



class Agent(object):
    def __init__(self, 
                env,
                n_envs,
                action_space_n,
                int_rwd_gamma,
                ext_rwd_gamma,
                gae_lambda,
                batch_size,
                minib_s,
                rnd_bonus,
                policy_inshape,
                policy_outshape,
                rnd_inshape,
                rnd_outshape,
                clip_epsilon,
                entropy_coeff,
                vf_coeff,
                optimizer,
                verbose) -> None:
        
        self.env            = env
        self.n_envs         = n_envs
        self.action_space_n = action_space_n
        self.int_rwd_gamma  = int_rwd_gamma
        self.ext_rwd_gamma  = ext_rwd_gamma
        self.gae_lambda     = gae_lambda
        self.batch_size     = batch_size
        self.minib_s        = minib_s
        self.rnd_bonus      = rnd_bonus
        self.policy         = self._create_policy(policy_inshape, policy_outshape)
        if self.rnd_bonus:
            self.target     = self._create_rnd("target", rnd_inshape, rnd_outshape)
            self.predictor  = self._create_rnd("predictor", rnd_inshape, rnd_outshape)
        self.clip_epsilon   = clip_epsilon
        self.entropy_coeff  = entropy_coeff
        self.vf_coeff       = vf_coeff
        self.optimizer      = optimizer
        self.verbose        = verbose
        if self.verbose > 0 :
            self.policy.summary()
            self.target.summary()
            self.predictor.summary()

    ### Create Networks
    def _create_policy(self, inshape, outshape):
        critic_outshape = 1
        actor_outshape = outshape

        inpt = Input(shape=inshape)
        
        convnet = self._create_convnet(inpt)

        actor       = Dense(actor_outshape, activation = "softmax", name = "actor") (convnet)
        critic_ext  = Dense(critic_outshape, activation = "linear", name = "critic_ext")(convnet)
        
        # If the rnd bonus is not active, the intrinsic critic will not be trained
        if self.rnd_bonus:
            trainable = True
        else: trainable = False

        critic_int  = Dense(critic_outshape, activation = "linear", name = "critic_int", trainable=trainable)(convnet)
         
        return keras.Model(inputs = inpt, 
                           outputs = [actor, critic_ext, critic_int],
                           name = "policy")
    

    def _create_rnd(self, name, inshape, outshape):
        output_shape    = outshape
        trainable       = True

        # The target network is not trainable
        if name == "target":
            trainable = False

        inpt = Input(shape=inshape)   
        covnet = self._create_convnet(inpt, trainable)       

        x = Dense(output_shape, activation="linear", trainable=trainable)(covnet)

        return keras.Model(inputs = inpt,
                           outputs = x,
                           name = name)
    

    def _create_convnet(self, inpt, trainable=True):

        x = Conv2D(filters = 32, kernel_size = 8, strides = 4, activation = "relu", trainable=trainable) (inpt) 
        x = Conv2D(filters = 64, kernel_size = 4, strides = 2, activation = "relu", trainable=trainable)(x)
        x = Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = "relu", trainable=trainable)(x)
        x = Flatten()(x)
        x = Dense(512, activation = "relu", trainable = trainable)(x)

        return x

    @tf.function
    def _forward_policy(self,state):
        # The policy nn uses a normalized state equal to x -> x/255
        state = tf.divide(state,255)
        actions_probs, ext_value, int_value  =  self.policy(state) 
        return actions_probs, ext_value, int_value


    @tf.function    
    def _forward_rnd(self, state):
        target_value    = self.target(state)
        predictor_value = self.predictor(state)
            
        return target_value, predictor_value
    

    def compute_action(self, actions_probs):
        action_dist = tfp.distributions.Categorical(probs=actions_probs)

        action = np.array([np.random.choice(np.arange(self.env.action_n), 
                                                p = np.reshape(env_actions_logits,-1))
                            for env_actions_logits in actions_probs],
                        dtype=np.int32) ## Actions must be a int32 

        action_logit = action_dist.log_prob(action)

        return action, action_logit 
    

    @tf.function
    def compute_int_rwd(self, norm_state, verbose=True):
        # The target and predictor use a preprocessed normalized state x -> CLIP((x-avg)/var,[-5,5])

        # Both (n_envs,n_action)
        target_value, predictor_value = self._forward_rnd(norm_state)

        # Calculate the norm of the squared error, returning a (nenvs,) tensor
        squared_error = tf.pow(tf.subtract(target_value, predictor_value), 2)
        int_rwd = tf.norm(squared_error, ord=2, axis=-1)

        return int_rwd
    

    def reset_env(self):
        state = self.env.reset()
        return state

    def play_one_step(self, state, random = False):
    
        if random:
            action = np.array([np.random.randint(0, self.action_space_n) 
                               for _ in range(self.n_envs)], dtype=np.int32)
            int_value = ext_value = None
            action_logit = None
        else:
            actions_probs, ext_value, int_value = self._forward_policy(state) 
            action, action_logit = self.compute_action(actions_probs)
        
        next_state, ext_rwd, done = self.env.step(action)


        return next_state, ext_rwd, done, action, action_logit, ext_value, int_value
        
    
    def calculate_returns(self, rwds, next_critic, discount, dones):
        T = rwds.shape[0] - 1
        returns = np.zeros(np.shape(rwds), dtype=np.float32)
        # Boostrap the next return 
        returns[T]  = rwds[T] + discount * next_critic * (1 - dones[T])
        running_ret = returns[T]
        for t in reversed(range(T)):
            returns[t]  = rwds[t] + discount * running_ret * (1 - dones[t])
            running_ret = returns[t]

        return returns

    def calculate_advs_and_returns(self, type, rwds, critic_vs, next_critic, dones):
        if type == "int":
            discount = self.int_rwd_gamma
        else:
            discount = self.ext_rwd_gamma

        T = rwds.shape[0] - 1
        next_critic = np.squeeze(next_critic)

        # Calculate advantages
        advs = np.zeros(np.shape(rwds), dtype=np.float32)
        advs[T] = rwds[T] + discount * next_critic * (1 - dones[T]) - critic_vs[T]
        running_adv = advs[T]
        for t in reversed(range(T)):
            # TD error
            delta = rwds[t] + discount * critic_vs[t + 1] * (1 - dones[t]) - critic_vs[t]
            advs[t] = delta + discount * self.gae_lambda * running_adv * (1 - dones[t])
            running_adv = advs[t]
        
        returns = self.calculate_returns(rwds, next_critic, discount, dones)

        # Normalize the advantage 
        advs -= np.mean(advs)
        advs /= (np.std(advs) + 1e-8)

        return advs , returns
    

    def training_step_ppo(self, n_minib, batch):
        
        sum_entrop      = 0
        sum_loss_actor  = 0
        sum_loss_critic = 0
        sum_loss_policy = 0
        sum_r_theta     = 0

        idxs = np.arange(self.batch_size)
        np.random.shuffle(idxs)
        
        for idx_minib in range(n_minib):
                        
            start = idx_minib * self.minib_s
            end   = start + self.minib_s
            shuffled_idxs   = idxs[start:end]
            
            entropy, loss_actor, loss_critic, loss_policy, r_theta = self.ppo_gradient(*batch.get_minibatch(shuffled_idxs))

            sum_entrop      += entropy
            sum_loss_actor  += loss_actor
            sum_loss_critic += loss_critic
            sum_loss_policy += loss_policy
            sum_r_theta     += tf.reduce_mean(r_theta)

        sum_entrop      /= n_minib
        sum_loss_actor  /= n_minib
        sum_loss_critic /= n_minib
        sum_loss_policy /= n_minib
        sum_r_theta     /= n_minib
        

        return sum_entrop, sum_loss_actor, sum_loss_critic, sum_loss_policy, sum_r_theta
        

    @tf.function
    def ppo_gradient(self, minib_states, minib_actions, minib_action_logits, minib_advs, minib_returns_ext, minib_returns_int):
        minib_states = tf.divide(minib_states, 255)
        with tf.GradientTape() as tape:  
                actions_probs, new_ext_value, new_int_value = self.policy(minib_states)

                action_dist = tfp.distributions.Categorical(probs=actions_probs)
                new_action_logits = action_dist.log_prob(minib_actions)
                entropy = tf.reduce_mean(action_dist.entropy())
                new_action_logits = tf.squeeze(new_action_logits)

                # r_theta = new_policy / old_policy =>> r_theta =  exp(log(new_policy) - log(old_policy))
                r_theta = tf.math.exp(tf.math.subtract(new_action_logits,minib_action_logits))
        
                loss_cpi = tf.math.multiply(r_theta,minib_advs)
                loss_clip = tf.clip_by_value(r_theta, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                
                loss_bound = tf.multiply(loss_clip, minib_advs)
                
                # Loss_clip is the mean given the minimum  btw the two losses
                loss_actor = tf.reduce_mean(tf.minimum(loss_cpi, loss_bound))
                
                # From (batch_size*nenvs,1) to (batch_size*n_envs)
                new_ext_value, new_int_value = tf.squeeze(new_ext_value), tf.squeeze(new_int_value)
                new_ext_value = tf.squeeze(new_ext_value)
                new_int_value = tf.squeeze(new_int_value)

                # Critic ext loss
                loss_critic_ext = tf.keras.losses.mean_squared_error(new_ext_value, minib_returns_ext)
                
                if self.rnd_bonus:                    
                    # Critic int loss
                    loss_critic_int = tf.keras.losses.mean_squared_error(new_int_value, minib_returns_int)                    
                    loss_critic = tf.divide((loss_critic_ext + loss_critic_int),2) 
                else: loss_critic = loss_critic_ext
                    
                loss_policy = loss_actor - self.vf_coeff * loss_critic + self.entropy_coeff * entropy
        
                # In this way the actor loss and entropy become negative, while the critic loss positive
                loss_policy = - loss_policy
            
        
        policy_variables = self.policy.trainable_variables
        policy_grads = tape.gradient(loss_policy, policy_variables)
        self.optimizer.apply_gradients(zip(policy_grads, policy_variables))
        
        return entropy, loss_actor, loss_critic, loss_policy, r_theta


    def training_step_distill(self, n_minib, norm_states):
        mean_loss_rnd = 0
        
        idxs = np.arange(self.batch_size)
        np.random.shuffle(idxs)
    
        for idx_minib in range(n_minib):
            start = idx_minib * self.minib_s
            end   = start + self.minib_s
            shuffled_idxs = idxs[start:end]

            minib_states = tf.gather(norm_states, indices=shuffled_idxs)

            loss_rnd = self.rnd_gradient(minib_states)
            
            mean_loss_rnd += loss_rnd
            
        return mean_loss_rnd/n_minib
        
        
    @tf.function
    def rnd_gradient(self, minib_states):
        with tf.GradientTape() as tape: 
            target_value    = self.target(minib_states)
            predictor_value = self.predictor(minib_states)
            mse = tf.losses.mean_squared_error(target_value,predictor_value)
            loss_rnd = tf.reduce_mean(mse)
            
        variables = self.predictor.trainable_variables
        grads = tape.gradient(loss_rnd, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        return loss_rnd
    
    def save_checkpoints(self, date_str):
        path = f"./checkpoints/{self.env.name}/{date_str}/"
        self.policy.save_weights(path + "policy/")
        if self.rnd_bonus:
            self.target.save_weights(path + "target/")
            self.predictor.save_weights(path + "predictor/")

    def restore_checkpoints(self,date_str):
        path = f"./checkpoints/{self.env.name}/{date_str}/"
        self.policy.load_weights(path + "policy/")
        if self.rnd_bonus:
            self.target.load_weights(path + "target/")
            self.predictor.load_weights(path + "predictor/")
