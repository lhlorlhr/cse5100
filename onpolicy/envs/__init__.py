
import socket

try:
    from absl import flags
except ModuleNotFoundError:
    flags = None
else:
    FLAGS = flags.FLAGS
    FLAGS(["train_sc.py"])


