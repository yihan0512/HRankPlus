import torch
import sys

def print_nonzero(ckpt):
    total = 0
    nnz = 0
    for key, value in ckpt['state_dict'].items():
        if 'conv1' in key:
            print('{}: {}/{}'.format(key, value.count_nonzero(), value.numel()))
        nnz += value.count_nonzero().item()
        total += value.numel()
    
    print('total: {}/{}'.format(nnz, total))

if __name__ == "__main__":
    ckpt = torch.load(sys.argv[1])
    print_nonzero(ckpt)
