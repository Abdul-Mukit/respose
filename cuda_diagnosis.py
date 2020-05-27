import numpy as np
import torch
from timeit import default_timer as timer
from torchvision.models import resnet50


def main():
    # Define model and input data
    resnet = resnet50().cuda()
    x = torch.from_numpy(np.random.rand(1, 3, 224, 224).astype(np.float32)).cuda()

    # The first pass is always slower, so run it once
    resnet.forward(x)

    # Measure elapsed time
    passes = 20
    total_time = 0
    for _ in range(passes):
        start = timer()
        resnet.forward(x)
        delta = timer() - start

        print('Forward pass: %.3fs' % delta)
        total_time += delta
    print('Average forward pass: %.3fs' % (total_time / passes))


if __name__ == '__main__':
    main()