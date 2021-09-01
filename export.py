import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage

from models import networks
from models.networks import define_G

netG = networks.define_G(3, 3, 64, "resnet_9blocks", "instance", False, "xavier", 0.02, False, False, [0])
netG.load_state_dict(torch.load("/media/winfried/Daten/data_external/experiments_from_gpu/CUT_second/latest_net_G.pth"))
netG.eval()
img = Image.open("imgs/2.png").convert("RGB").resize((256, 256))
img = ToTensor()(img).cuda().unsqueeze(0)


class MyTraceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.g = netG

    def forward(self, x):
        x = self.n(x)
        x = self.g(x)
        return torch.clamp(x, 0.0, 1.0)


@torch.jit.script
def execute(x):
    # x =
    x = netG(x)
    return torch.clamp(x, 0.0, 1.0)


save_path = "/media/winfried/Daten/data_external/experiments_from_gpu/CUT_second/generator_ts.pt"
script_module = torch.jit.trace(MyTraceModule(), img)
script_module.save(save_path)
m = torch.jit.load(save_path)
m.eval()


out = m(img)
out = ToPILImage()(out.squeeze(0).cpu())
plt.imshow(out)
plt.show()

print("done")
