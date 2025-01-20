import gymnasium as gym
import numpy as np
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import deque
import ale_py
import time


# Prétraitement des frames
def preprocessing(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
    normalized_frame = resized_frame / 255.0
    return normalized_frame