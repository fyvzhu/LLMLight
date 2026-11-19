"""TRL PPO with compact microbatches, correct forward mode and BF16."""
import torch
from trl import PPOTrainer


class CheckpointedPPOTrainer(PPOTrainer):
    """TRL 0.9.2 calls eval() inside both inference and gradient forwards.

    Set the mode at the actual model call so training forwards can use gradient
    checkpointing. Use BF16 autocast without changing prepared weight dtypes.
    Remove only columns that are padding for the entire microbatch. Delegate
    response masks, losses and optimization to TRL, keeping original positions.
    """

    def _forward_pass(self, model, *args, **kwargs):
        training_forward = torch.is_grad_enabled()
        previous_modes = [(module, module.training) for module in model.modules()]

        def set_forward_mode(module, inputs):
            module.train(training_forward)

        handle = model.register_forward_pre_hook(set_forward_mode)
        try:
            with torch.autocast(self.accelerator.device.type, dtype=torch.bfloat16):
                return super().batched_forward_pass(model, *args, **kwargs)
        finally:
            handle.remove()
            # Preserve mixed child-module modes as well as the root mode.
            for module, training in previous_modes:
                module.training = training

    def batched_forward_pass(self, model, queries, responses, model_inputs,
                             return_logits=False, response_masks=None):
        if self.is_encoder_decoder or self.config.kl_penalty == 'full':
            raise ValueError('CoLLMLight PPO requires decoder-only sampled-KL training')
        width = model_inputs['input_ids'].shape[1]
        parts = []
        for start in range(0, len(queries), self.config.mini_batch_size):
            end = min(start + self.config.mini_batch_size, len(queries))
            inputs = {key: value[start:end] for key, value in model_inputs.items()}
            active = inputs['attention_mask'].bool().any(dim=0).nonzero().flatten()
            if not active.numel():
                raise ValueError('PPO microbatch has no valid input tokens')
            left, right = int(active[0]), int(active[-1]) + 1
            inputs = {key: value[:, left:right] for key, value in inputs.items()}
            # These models normally count positions from the original padded
            # origin. Preserve that origin when removing leading padding.
            inputs.setdefault('position_ids', torch.arange(left, right,
                device=inputs['input_ids'].device).unsqueeze(0).expand(end-start, -1))
            result = self._forward_pass(model, queries[start:end], responses[start:end], inputs,
                return_logits=return_logits,
                response_masks=None if response_masks is None else response_masks[start:end])
            if return_logits:
                # TRL requests logits only for one gradient microbatch here.
                if len(queries) > self.config.mini_batch_size:
                    raise ValueError('PPO training logits require a single microbatch')
                self._loss_bounds = (left, right-1)
                return result
            logprobs, _, values, mask = result
            # Rewards/GAE still use TRL's original batch-wide coordinates.
            parts.append(tuple(torch.nn.functional.pad(value, (left, width-right))
                               for value in (logprobs, values, mask)))
        return (torch.cat([part[0] for part in parts]), None,
                torch.cat([part[1] for part in parts]), torch.cat([part[2] for part in parts]))

    def train_minibatch(self, old_logprobs, values, logprobs, logits, vpreds,
                        mask, advantages, returns):
        left, right = self._loss_bounds
        return super().train_minibatch(old_logprobs[:, left:right], values[:, left:right],
            logprobs, logits, vpreds, mask[:, left:right], advantages[:, left:right],
            returns[:, left:right])
