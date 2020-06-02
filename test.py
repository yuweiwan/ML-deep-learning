import csv
import argparse as ap

import torch
import torch.nn.functional as F
import numpy as np
from skimage.transform import rotate

ran = torch.LongTensor(10).random_(1,1,5)
print(ran)