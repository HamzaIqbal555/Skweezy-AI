import torch
import torchvision
from transformers import BlipProcessor, BlipForConditionalGeneration


print("BLIP imports successful")
print(torch.__version__)
print(torchvision.__version__)

import lzma
print(lzma.__file__)