from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch

def get_dataloader(dataset, batch_size, is_train = True):
    
    if is_train:
        sampler = RandomSampler(dataset)
        batch_size = batch_size * torch.cuda.device_count()
    else:
        sampler = SequentialSampler(dataset)
        batch_size = 2*batch_size * torch.cuda.device_count()

    dataloader = DataLoader(dataset, sampler=sampler, 
            batch_size=batch_size, num_workers=4)

    return dataloader