# Multi-Agent Attention-Based Modelling

This model contains two main components: an **encoder** and a **decoder**. However, you can choose to use either the encoder or decoder independently, depending on your specific needs.

## Explaining `model/algorithm/mat_decoder.py`

### `discrete_autoregreesive_act` Function

This function takes the following inputs:

- `decoder`: The decoder model
- `obs_rep`: Encoded observations
- `obs`: Raw observations
- `n_agent`: Number of agents
- `action_dim`: Dimension of each agent's action space

#### Action Initialization

Agents take actions sequentially. To allow the first agent to take its action without any prior actions, we initialize a shifted action tensor with an additional dimension:

```python
shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1))
shifted_action[:, 0, 0] = 1
```
Here, we create an action space of shape (action_dim + 1), filled with zeros. The first element is set to 1 for the first agent, simulating a "previous action" that doesn't actually exist.

#### Action Generation Loop
```python
for i in range(n_agent):
    logit, v_loc = decoder(shifted_action, obs_rep, obs)
    logit = logit[:, i, :]
    
    if available_actions is not None:
        logit[available_actions[:, i, :] == 0] = -1e10

    distri = Categorical(logits=logit)
    action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
    action_log = distri.log_prob(action)

    output_action[:, i, :] = action.unsqueeze(-1)
    output_action_log[:, i, :] = action_log.unsqueeze(-1)

    if i + 1 < n_agent:
        shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
```
In this loop:

- **The decoder is given the current `shifted_action`, `obs_rep`, and `obs` to predict the next action.**

- **`logit = logit[:, i, :]` retrieves the logits for the $i-th$ agent.**

- **If `available_actions` are provided, the logits of unavailable actions are masked with a large negative value.**

- **An action is selected either deterministically or by sampling from the probability distribution.**

- **`shifted_action` is updated with the current agent’s action (one-hot encoded) for the next agent.**

- **This sequential autoregressive mechanism allows each agent to make decisions conditioned on the actions of those before it.**

This sequential autoregressive mechanism allows each agent to make decisions conditioned on the actions of those before it.

### `discrete_parallel_act` function
This function is very similar to the previous one, but the key difference is that **agent actions are not performed sequentially** in this case.

**Note:** `continuous_autoregressive_act` and `continuous_parallel_act` are similar to the previous functions. However, the key difference is that **action selection is continuous rather than categorical** in these cases. These functions perform **continuous action prediction** instead of choosing from discrete action categories.

### `DecodeBlock` Class

![encode block image](<img/image.png>)