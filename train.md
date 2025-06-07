## Training and Rollout Preparation

Before training and rollout, we need two helper methods to toggle the model between training and evaluation modes:

```python
def prep_training(self):
    # Enable dropout and batch norm layers in training mode
    self.policy.train()


def prep_rollout(self):
    # Disable dropout and set batch norm layers to evaluation mode
    self.policy.eval()
```

These methods ensure that layers like dropout and batch normalization behave correctly during training and inference.

---

## Action Selection Workflow

Our policy network consists of an encoder and a decoder. Since the MPE environment does not use an explicit state, we pass a zero tensor for the state input to maintain consistent shapes. The `get_actions` method handles preprocessing and calls the encoder and decoder as follows:

```python
def get_actions(self, state, obs, available_actions=None, deterministic=False):
    # Unused `state`; create a zero-state tensor with the correct shape
    ori_shape = obs.shape
    state = torch.zeros((*ori_shape[:-1], 37), dtype=torch.float32)

    # Move tensors to the appropriate device/dtype
    state = check(state).to(**self.tpdv)
    obs = check(obs).to(**self.tpdv)
    if available_actions is not None:
        available_actions = check(available_actions).to(**self.tpdv)

    batch_size = obs.shape[0]

    # Encode observations to get a latent representation and value estimate
    v_loc, obs_rep = self.encoder(state, obs)

    # Choose discrete or continuous action sampling
    if self.action_type == "Discrete":
        output_action, output_action_log = discrete_autoregressive_act(
            self.decoder, obs_rep, obs, batch_size,
            self.n_agent, self.action_dim, self.tpdv,
            available_actions, deterministic
        )
    else:
        output_action, output_action_log = continuous_autoregressive_act(
            self.decoder, obs_rep, obs, batch_size,
            self.n_agent, self.action_dim, self.tpdv,
            deterministic
        )

    return output_action, output_action_log, v_loc
```

1. **Zero-state tensor**: We create a dummy state tensor of shape `(batch_size, ..., 37)` filled with zeros.
2. **Device transfer**: We move `state`, `obs`, and `available_actions` (if provided) to the correct device and dtype.
3. **Encoding**: The encoder produces a value estimate `v_loc` and a representation tensor `obs_rep`.
4. **Decoding**: Based on `action_type`, we call either the discrete or continuous autoregressive action sampler.

---

## Discrete Autoregressive Action Sampling

The discrete sampler generates joint actions for all agents in an autoregressive fashion:

```python
def discrete_autoregressive_act(
    decoder, obs_rep, obs, batch_size,
    n_agent, action_dim, tpdv,
    available_actions=None, deterministic=False
):
    # Initialize a shifted-action tensor for autoregressive inputs
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1), **tpdv)
    # First agent has no previous actions, so we set the zero-th index to 1
    shifted_action[:, 0, 0] = 1

    # Placeholders for outputs
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        # Compute logits for agent i
        logits = decoder(shifted_action, obs_rep, obs)[:, i, :]
        # Mask out unavailable actions
        if available_actions is not None:
            logits[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logits)
        # Sample or select the most probable action
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, 0] = action
        output_action_log[:, i, 0] = action_log

        # Prepare shifted_action for the next agent
        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)

    return output_action, output_action_log
```

* **`shifted_action`**: Holds the one-hot encoding of previous agents' actions. The first agent uses a dummy vector `[1, 0, …, 0]`.
* **Action masking**: Forces the network to only choose from `available_actions` if provided.
* **Autoregressive loop**: Each agent’s action is conditioned on all previous agents’ sampled actions.

---

This structure allows the policy to model dependencies between multiple agents' actions while respecting action constraints at each step.

---

## General Action Selection Function

Below is a more general `get_actions` wrapper that handles both actor and critic RNN states and reshapes inputs/outputs for multi-agent settings:

```python
def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                deterministic=False):
    # Reshape inputs for multi-agent batch processing
    cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
    obs = obs.reshape(-1, self.num_agents, self.obs_dim)
    if available_actions is not None:
        available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

    # Delegate to the transformer policy's get_actions
    values, actions, action_log_probs = self.transformer.get_actions(
        cent_obs, obs, available_actions, deterministic
    )

    # Flatten outputs back to single batch dimension
    actions = actions.view(-1, self.act_num)
    action_log_probs = action_log_probs.view(-1, self.act_num)
    values = values.view(-1, 1)

    # Move RNN states to correct device/dtype (for compatibility)
    rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
    rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)

    return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
```
The general `get_actions` wrapper integrates actor and critic components, handling sequence-based policies (e.g., RNNs) and multi-agent batching. It performs the following steps:

1. **Reshape Inputs**:

   * Centralized observations (`cent_obs`) are reshaped to `(batch_size, num_agents, share_obs_dim)` for the critic.
   * Local observations (`obs`) become `(batch_size, num_agents, obs_dim)` for the actor.
   * If provided, `available_actions` is reshaped to `(batch_size, num_agents, act_dim)` to mask invalid moves per agent.

2. **Delegate to Policy**:
   Calls `self.transformer.get_actions`, which returns:

   * `values`: Value function predictions for each agent.
   * `actions`: Sampled or mode actions in multi-agent format.
   * `action_log_probs`: Log probabilities of the chosen actions.

3. **Flatten Outputs**:
   Transforms the multi-agent outputs back to flat batch tensors:

   * `actions` → `(batch_size * num_agents, act_num)`
   * `action_log_probs` → same shape as `actions`
   * `values` → `(batch_size * num_agents, 1)`

4. **RNN State Compatibility**:
   Even if the current wrapper does not update RNN states, it moves `rnn_states_actor` and `rnn_states_critic` through a dummy `check(...).to(**self.tpdv)` to ensure they remain on the correct device and dtype for future iterations.


---

## Data Collection for Rollout

The `collect` method gathers rollouts from the environment using the policy in evaluation mode:

```python
@torch.no_grad()
def collect(self, step):
    # Ensure policy is in rollout (eval) mode
    self.trainer.prep_rollout()

    # Retrieve values, actions, log probs, and RNN states
    values, actions, action_log_probs, rnn_states, rnn_states_critic = \
        self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step])
        )

    # Split outputs by environment thread
    values = np.array(np.split(_t2n(values), self.n_rollout_threads))
    actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
    action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
    rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
    rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

    # Convert discrete actions to one-hot encoding based on env action space
    if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
        for i in range(self.envs.action_space[0].shape):
            uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
            actions_env = uc_actions_env if i == 0 else np.concatenate((actions_env, uc_actions_env), axis=2)
    elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
        actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
    else:
        raise NotImplementedError("Unsupported action space type")

    return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env
```


The `collect` method leverages the policy in evaluation mode (`prep_rollout`) to sample trajectories from the environment. Key steps:

1. **Set Evaluation Mode**:
   `self.trainer.prep_rollout()` disables training-only layers like dropout.

2. **Action Sampling**:

   * Concatenate buffered arrays (`share_obs`, `obs`, `rnn_states`, `rnn_states_critic`, `masks`) along the time dimension for the current `step`.
   * Call `policy.get_actions` to obtain:

     * `values`: Tensor of shape `(batch_size * num_agents, 1)`
     * `actions`: Raw action indices.
     * `action_log_probs` and updated RNN states.

3. **Thread Splitting**:
   Since environments run in parallel threads (`n_rollout_threads`), split each output tensor back into lists of shape `(n_threads, agents, ...)` using `np.split`. This preserves per-thread rollout sequences.

4. **Action Formatting**:
   Convert discrete action indices to one-hot encodings matching the environment’s action space:

   * **MultiDiscrete**: For each action dimension, create an identity matrix of size `(high[i]+1)`, index by action.
   * **Discrete**: Use a single identity matrix of size `(n,)`.
   * Raises an error for unsupported action spaces.

5. **Return**:
   Returns a tuple:

   ```python
   (values, actions, action_log_probs,
    rnn_states, rnn_states_critic,
    actions_env)
   ```

   where `actions_env` is the one-hot formatted action array ready for environment step.

This method cleanly separates evaluation-mode policy inference from environment interactions, ensuring that data fed back into training buffers is correctly shaped and encoded.

## main run function
```python
def run(self):
    self.warmup()   
    self.gif_dir="./gifs"
    start = time.time()
    episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

    for episode in range(episodes):
        if self.use_linear_lr_decay:
            self.trainer.policy.lr_decay(episode, episodes)

        for step in range(self.episode_length):
            # Sample actions
            values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
            # Obser reward and next obs
            obs, rewards, dones, infos = self.envs.step(actions_env)
            data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
            # insert data into buffer
            self.insert(data)

        # compute return and update network
        self.compute()
        train_infos = self.train()
        # post process
        total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
        # save model
        if (episode % self.save_interval == 0 or episode == episodes - 1):
            self.save(episode)
        # log information
        if episode % self.log_interval == 0:
            end = time.time()
            print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(self.all_args["scenario_name"],
                            self.algorithm_name,
                            self.experiment_name,
                            episode,
                            episodes,
                            total_num_steps,
                            self.num_env_steps,
                            int(total_num_steps / (end - start))))

            if self.env_name == "MPE":
                env_infos = {}
                for agent_id in range(self.num_agents):
                    idv_rews = []
                    for info in infos:
                        if 'individual_reward' in info[agent_id].keys():
                            idv_rews.append(info[agent_id]['individual_reward'])
                    agent_k = 'agent%i/individual_rewards' % agent_id
                    env_infos[agent_k] = idv_rews
```

## Add data to buffer
```python
def insert(self, data):
    obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

    rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
    rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
    masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
    masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

    if self.use_centralized_V:
        share_obs = obs.reshape(self.n_rollout_threads, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
    else:
        share_obs = obs

    self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)
```

### insert function of buffer
```python
def insert(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
            value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
    """
    Insert data into the buffer.
    :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param obs: (np.ndarray) local agent observations.
    :param rnn_states_actor: (np.ndarray) RNN states for actor network.
    :param rnn_states_critic: (np.ndarray) RNN states for critic network.
    :param actions:(np.ndarray) actions taken by agents.
    :param action_log_probs:(np.ndarray) log probs of actions taken by agents
    :param value_preds: (np.ndarray) value function prediction at each step.
    :param rewards: (np.ndarray) reward collected at each step.
    :param masks: (np.ndarray) denotes whether the environment has terminated or not.
    :param bad_masks: (np.ndarray) action space for agents.
    :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
    :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
    """
    self.share_obs[self.step + 1] = share_obs.copy()
    self.obs[self.step + 1] = obs.copy()
    self.rnn_states[self.step + 1] = rnn_states_actor.copy()
    self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
    self.actions[self.step] = actions.copy()
    self.action_log_probs[self.step] = action_log_probs.copy()
    self.value_preds[self.step] = value_preds.copy()
    self.rewards[self.step] = rewards.copy()
    self.masks[self.step + 1] = masks.copy()
    if bad_masks is not None:
        self.bad_masks[self.step + 1] = bad_masks.copy()
    if active_masks is not None:
        self.active_masks[self.step + 1] = active_masks.copy()
    if available_actions is not None:
        self.available_actions[self.step + 1] = available_actions.copy()

    self.step = (self.step + 1) % self.episode_length
```
## how is `train_infos = self.train()` will be worked?
```python
def train(self, buffer):
    """
    Perform a training update using minibatch GD.
    :param buffer: (SharedReplayBuffer) buffer containing training data.
    :param update_actor: (bool) whether to update actor network.

    :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
    """
    advantages_copy = buffer.advantages.copy()
    advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
    mean_advantages = np.nanmean(advantages_copy)
    std_advantages = np.nanstd(advantages_copy)
    advantages = (buffer.advantages - mean_advantages) / (std_advantages + 1e-5)
    

    train_info = {}

    train_info['value_loss'] = 0
    train_info['policy_loss'] = 0
    train_info['dist_entropy'] = 0
    train_info['actor_grad_norm'] = 0
    train_info['critic_grad_norm'] = 0
    train_info['ratio'] = 0

    for _ in range(self.ppo_epoch):
        data_generator = buffer.feed_forward_generator_transformer(advantages, self.num_mini_batch)

        for sample in data_generator:

            value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                = self.ppo_update(sample)

            train_info['value_loss'] += value_loss.item()
            train_info['policy_loss'] += policy_loss.item()
            train_info['dist_entropy'] += dist_entropy.item()
            train_info['actor_grad_norm'] += actor_grad_norm
            train_info['critic_grad_norm'] += critic_grad_norm
            train_info['ratio'] += imp_weights.mean()

    num_updates = self.ppo_epoch * self.num_mini_batch

    for k in train_info.keys():
        train_info[k] /= num_updates

    return train_info
```

### what is `feed_forward_generator_transformer`
```python
def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
    """
    Yield training data for MLP policies.
    :param advantages: (np.ndarray) advantage estimates.
    :param num_mini_batch: (int) number of minibatches to split the batch into.
    :param mini_batch_size: (int) number of samples in each minibatch.
    """
    episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
    batch_size = n_rollout_threads * episode_length

    if mini_batch_size is None:
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(n_rollout_threads, episode_length,
                        n_rollout_threads * episode_length,
                        num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch

    rand = torch.randperm(batch_size).numpy()
    sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
    rows, cols = _shuffle_agent_grid(batch_size, num_agents)

    # keep (num_agent, dim)
    share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
    share_obs = share_obs[rows, cols]
    obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
    obs = obs[rows, cols]
    rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
    rnn_states = rnn_states[rows, cols]
    rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
    rnn_states_critic = rnn_states_critic[rows, cols]
    actions = self.actions.reshape(-1, *self.actions.shape[2:])
    actions = actions[rows, cols]
    if self.available_actions is not None:
        available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
        available_actions = available_actions[rows, cols]
    value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
    value_preds = value_preds[rows, cols]
    returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
    returns = returns[rows, cols]
    masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
    masks = masks[rows, cols]
    active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
    active_masks = active_masks[rows, cols]
    action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
    action_log_probs = action_log_probs[rows, cols]
    advantages = advantages.reshape(-1, *advantages.shape[2:])
    advantages = advantages[rows, cols]

    for indices in sampler:
        # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
        share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[2:])
        obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
        rnn_states_batch = rnn_states[indices].reshape(-1, *rnn_states.shape[2:])
        rnn_states_critic_batch = rnn_states_critic[indices].reshape(-1, *rnn_states_critic.shape[2:])
        actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
        if self.available_actions is not None:
            available_actions_batch = available_actions[indices].reshape(-1, *available_actions.shape[2:])
        else:
            available_actions_batch = None
        value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
        return_batch = returns[indices].reshape(-1, *returns.shape[2:])
        masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
        active_masks_batch = active_masks[indices].reshape(-1, *active_masks.shape[2:])
        old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])
        if advantages is None:
            adv_targ = None
        else:
            adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

        yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                adv_targ, available_actions_batch

```
input shape is `T, N-threads, N-agents, ...` and will be reshape to `T * N-threads, N-agents, ...`.
### how `advantages` will be caculated?
from `self.compute()` of main run train function
```python
def compute(self):
    """Calculate returns for the collected data."""
    self.trainer.prep_rollout()
    if self.buffer.available_actions is None:
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                        np.concatenate(self.buffer.obs[-1]),
                                                        np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
    else:
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                        np.concatenate(self.buffer.obs[-1]),
                                                        np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]),
                                                        np.concatenate(self.buffer.available_actions[-1]))
    next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
    self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
```

#### what is `self.buffer.compute_returns`
```python
def compute_returns(self, next_value, value_normalizer=None):
    """
    Compute returns either as discounted sum of rewards, or using GAE.
    :param next_value: (np.ndarray) value predictions for the step after the last episode step.
    :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
    """
    self.value_preds[-1] = next_value
    gae = 0
    for step in reversed(range(self.rewards.shape[0])):
        if self._use_popart or self._use_valuenorm:
            delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                self.value_preds[step + 1]) * self.masks[step + 1] \
                    - value_normalizer.denormalize(self.value_preds[step])
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

            # here is a patch for mpe, whose last step is timeout instead of terminate
            if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
                gae = 0

            self.advantages[step] = gae
            self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
        else:
            delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
                    self.masks[step + 1] - self.value_preds[step]
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

            # here is a patch for mpe, whose last step is timeout instead of terminate
            if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
                gae = 0

            self.advantages[step] = gae
            self.returns[step] = gae + self.value_preds[step]
```
### back to self.train() what is the `self.ppo_update(sample)`
```python
def ppo_update(self, sample):
    """
    Update actor and critic networks.
    :param sample: (Tuple) contains data batch with which to update networks.
    :update_actor: (bool) whether to update actor network.

    :return value_loss: (torch.Tensor) value function loss.
    :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
    ;return policy_loss: (torch.Tensor) actor(policy) loss value.
    :return dist_entropy: (torch.Tensor) action entropies.
    :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
    :return imp_weights: (torch.Tensor) importance sampling weights.
    """
    share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
    value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
    adv_targ, available_actions_batch = sample

    old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
    adv_targ = check(adv_targ).to(**self.tpdv)
    value_preds_batch = check(value_preds_batch).to(**self.tpdv)
    return_batch = check(return_batch).to(**self.tpdv)
    active_masks_batch = check(active_masks_batch).to(**self.tpdv)

    # Reshape to do in a single forward pass for all steps
    values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                            obs_batch, 
                                                                            rnn_states_batch, 
                                                                            rnn_states_critic_batch, 
                                                                            actions_batch, 
                                                                            masks_batch, 
                                                                            available_actions_batch,
                                                                            active_masks_batch)
    # actor update
    imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

    surr1 = imp_weights * adv_targ
    surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

    if self._use_policy_active_masks:
        policy_loss = (-torch.sum(torch.min(surr1, surr2),
                                    dim=-1,
                                    keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
    else:
        policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

    # critic update
    value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

    loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef

    self.policy.optimizer.zero_grad()
    loss.backward()

    if self._use_max_grad_norm:
        grad_norm = nn.utils.clip_grad_norm_(self.policy.transformer.parameters(), self.max_grad_norm)
    else:
        grad_norm = get_gard_norm(self.policy.transformer.parameters())

    self.policy.optimizer.step()

    return value_loss, grad_norm, policy_loss, dist_entropy, grad_norm, imp_weights
```
`adv_targ` was computed by `compute_returns`
