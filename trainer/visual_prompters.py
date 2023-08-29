"""
    source1(VP): https://github.com/hjbahng/visual_prompting
"""
import torch
import torch.nn as nn
from torchvision import transforms


# inverting RGB transform of CLIP
inv_normalize = transforms.Normalize(
                                    mean=[-0.48145466/0.26862954, 
                                          -0.4578275/0.26130258, 
                                          -0.40821073/0.27577711],
                                    std=[1/0.26862954, 
                                         1/0.26130258, 
                                         1/0.27577711]
                                    )


class PadPrompter(nn.Module):
    def __init__(self, p_eps):
        super(PadPrompter, self).__init__()
        self.pad_size = 30
        image_size = 224
        self.p_eps = p_eps

        self.base_size = image_size - self.pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 3, self.pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, self.pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - self.pad_size*2, self.pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - self.pad_size*2, self.pad_size]))

    def forward(self, x):
        n_samples = x.shape[0]
        base = torch.zeros(1, 3, self.base_size, self.base_size).to(x.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])
        x_prompted = x + self.p_eps * prompt
        return x_prompted
