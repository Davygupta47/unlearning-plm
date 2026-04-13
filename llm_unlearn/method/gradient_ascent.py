import torch
from torch import nn
from transformers import Trainer
from transformers.utils import is_sagemaker_mp_enabled, is_apex_available

from typing import Dict, Union, Any

if is_apex_available():
    from apex import amp

class GradientAscentTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes the negative of the mean loss for gradient ascent.
        Compatible with newer Transformers versions.
        """
        outputs = model(**inputs)

        # Get loss tensor
        loss_tensor = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else outputs[0]

        # Mean over batch
        loss = loss_tensor.mean()

        # Gradient ascent → maximize loss
        return (-loss, outputs) if return_outputs else -loss