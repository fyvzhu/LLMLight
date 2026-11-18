"""TRL PPO with the required forward mode and BF16 mixed precision."""
import torch
from trl import PPOTrainer


class CheckpointedPPOTrainer(PPOTrainer):
    """TRL 0.9.2 calls eval() inside both inference and gradient forwards.

    Set the mode at the actual model call so training forwards can use gradient
    checkpointing. Use BF16 autocast without changing prepared weight dtypes.
    Delegate batching, masks, losses and optimization to TRL.
    """

    def batched_forward_pass(self, model, *args, **kwargs):
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
