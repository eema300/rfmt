from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tz_dataset import TZ_Dataset

BATCH_SIZE = 32
DATASET_PATH = ''

dataset = TZ_Dataset(DATASET_PATH)
n = len(dataset)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)
n_test  = n - n_train - n_val

train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test],
                                            generator=torch.Generator().manual_seed(112))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) # i think i need a collate fn for these?
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=True)
# dataset info: 
#   * iterable-type
#   * [z] vectors as feature, t's as labels
#   * multiple observations for each flow matching procedure: <THIS IS NEXT BIG Q TO ANSWER>
#       * doing property guidance (at different t's or at different amounts?)
#       * this means i include a non-guided procedure too right?


