# Portions of this file are adapted from Decision Transformer:
#   https://github.com/kzl/decision-transformer
# Original work © 2021 Decision Transformer (Decision Transformer: Reinforcement Learning via Sequence Modeling) Authors
#   https://arxiv.org/abs/2106.01345
# Licensed under the MIT License (see third_party/decision-transformer/LICENSE).

import numpy as np
import torch

from mpc_decisioner.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        state_target, action_target, reward_target, attention_mask_pred, attention_mask_tar = torch.clone(states), torch.clone(actions), torch.clone(rtg[:,:-1]), torch.clone(attention_mask), torch.clone(attention_mask)
        attention_mask_pred[:,-1] = 0
        # Find the first valid timestep for each sequence (where attention_mask is non-zero)
        first_valid_indices = (attention_mask > 0).int().argmax(dim=1)  # Shape: (batch_size,)
        # Set first valid timestep to 0 in `state_target_mask`
        for i in range(attention_mask.shape[0]):  # Loop through batch
            attention_mask_tar[i, first_valid_indices[i]] = 0  # Mask out first valid timestep
        
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        rwd_dim = reward_preds.shape[2]
        reward_preds = reward_preds.reshape(-1, rwd_dim)[attention_mask_pred.reshape(-1) > 0]
        reward_target = reward_target.reshape(-1, rwd_dim)[attention_mask_tar.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, reward_preds,
            None, action_target, reward_target,
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
