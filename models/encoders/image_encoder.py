import os
import torch.nn as nn
from transformers import AutoModel, Dinov2Config, Dinov2Model

    

class DINOv2Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.freeze = True
        # Complete MGPC checkpoints contain all encoder weights; strict loading follows.
        self.model = (Dinov2Model(Dinov2Config(image_size=518)) if os.environ.get('MGPC_ENCODER_INIT_ONLY') == '1'
                      else AutoModel.from_pretrained('facebook/dinov2-base'))
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    

    def forward(self, images):

        outputs = self.model(images)
        image_tokens = outputs.last_hidden_state          # b, 257(16*16+1), 768
        return image_tokens
    