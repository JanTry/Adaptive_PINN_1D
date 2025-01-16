import logging
import sys

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_LEVEL = logging.INFO  # DEF logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s[%(levelname)s] %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
logging.log(logging.DEBUG, f"Using device: {DEVICE}")
