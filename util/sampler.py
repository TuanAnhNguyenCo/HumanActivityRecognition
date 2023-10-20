from torch.utils.data import SubsetRandomSampler


def data_sampler(idx):
    return SubsetRandomSampler(idx)
    
