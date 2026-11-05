"""model.py - Stage 2, step 6: the network.

Check it builds and see its output shape, size, and receptive field:

    python model.py
"""
import torch
import torch.nn as nn

NUM_CLASSES = 3   # person, fire_hydrant, trash_can (+1 background = 4 class scores)


# def block(c_in, c_out):
#     """3x3 convolution with same padding -> batch norm -> ReLU.

#     bias=False because batch norm's own learnable shift does the bias's job.
#     """
#     return nn.Sequential(
#         nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
#         nn.BatchNorm2d(c_out),
#         nn.ReLU(inplace=True),
#     )
# testing one less max pool;

def block(c_in, c_out, dilation=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=dilation,
                  dilation=dilation, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )

class Detector(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            block(3, 16),                        # stage 1   240x384
            nn.MaxPool2d(2),                     #        -> 120x192
            block(16, 32),                       # stage 2
            nn.MaxPool2d(2),                     #        -> 60x96
            block(32, 64), block(64, 64),        # stage 3
            nn.MaxPool2d(2),                     #        -> 30x48
            block(64, 128), block(128, 128),     # stage 4
            nn.MaxPool2d(2),                     #        -> 15x24 (the grid)
            block(128, 128), block(128, 128),    # stage 5: no pool, grows the receptive field
        )
        # Head: 1x1 convolution, one answer per cell. 4 class scores + 4 box numbers.
        self.head = nn.Conv2d(128, (num_classes + 1) + 4, kernel_size=1)
        self.num_scores = num_classes + 1

    def forward(self, x):
        out = self.head(self.features(x))              # [batch, 8, 15, 24]
        class_scores = out[:, : self.num_scores]       # [batch, 4, 15, 24]  (softmax later, in the cost)
        box_raw = out[:, self.num_scores:]             # [batch, 4, 15, 24]  (sigmoid later)
        return class_scores, box_raw


def receptive_field(module):
    """Receptive field (pixels of the input) of one output cell."""
    rf, jump = 1, 1
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            k, s = layer.kernel_size[0], layer.stride[0]
        elif isinstance(layer, nn.MaxPool2d):
            k = s = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
        else:
            continue
        rf += (k - 1) * jump
        jump *= s
    return rf


if __name__ == "__main__":
    model = Detector()
    x = torch.zeros(2, 3, 240, 384)
    scores, box = model(x)
    params = sum(p.numel() for p in model.parameters())
    print("class scores:", tuple(scores.shape), " box:", tuple(box.shape))
    print(f"learnable parameters: {params:,}")
    print(f"receptive field of one cell: {receptive_field(model.features)} px (input is 384x240)")
