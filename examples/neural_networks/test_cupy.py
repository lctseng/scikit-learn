#!/usr/bin/env python3

import numpy as np
import cupy as cp

import pdb
import time

x_gpu = cp.arange(40000, dtype=np.float64).reshape(200,200)
cp.dot(x_gpu, x_gpu)

ts = time.time()
cp.dot(x_gpu, x_gpu)

print("GPU  Time elpased: %f ms \n" % ((time.time() - ts) * 1000))


x_cpu = np.arange(40000,dtype=np.float64).reshape(200,200)


ts = time.time()
np.dot(x_cpu, x_cpu)

print("CPU  Time elpased: %f ms \n" % ((time.time() - ts) * 1000))

