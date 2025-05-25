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