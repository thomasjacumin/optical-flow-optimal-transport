import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import time

import utils
import benamou_brenier
import classical

# Parse parameters
parser = argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("f0", help="first frame")
parser.add_argument("f1", help="second frame")
parser.add_argument("--out", nargs='?', help="optical flow output")
parser.add_argument("--ground-truth", nargs='?', help="optical flow ground truth")
parser.add_argument("--save-benchmark", nargs='?', help="file output of benchmark")
# Model parameters
parser.add_argument("--Nt", nargs='?', type=int, default=4, help="Discretization in time")
parser.add_argument("--epsilon", nargs='?', type=float, default=0.1, help="Stopping threshold")
parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, help="normalize the input images if enabled")

args = parser.parse_args()

np.random.seed(0)

# Create data
f1, w, h = utils.openGrayscaleImage(args.f0)
f2, w, h = utils.openGrayscaleImage(args.f1)

print("***********************************")
print("Input images: ")
print(" - f0 = "+str(args.f0))
print(" - f1 = "+str(args.f1))
if args.normalize == True:
    print(" - normalize input images")
    f1 = f1/(np.sum(f1)/(w*h))
    f2 = f2/(np.sum(f2)/(w*h))

# Start timer
start_time = time.time()
# Solve
mu, phi, q = benamou_brenier.solve(f1, f2, args.Nt, w, h, epsilon=args.epsilon, r=1.1)
u, v, m = utils.opticalflow_from_benamoubrenier(phi, args.Nt, w, h)
# stop timer
timer = time.time() - start_time

# Benchmark
if args.ground_truth:
    print("Benchmark:")
    rec = utils.apply_opticalflow(f1, u, v, w, h, m)
    rec = np.clip(rec, 0, 1)
    wGT, hGT, uGT, vGT = utils.openFlo(args.ground_truth)
    assert(wGT == w and hGT == h)
    AEE, SDEE = utils.EE(w, h, u, v, uGT, vGT)
    AAE, SDAE = utils.AE(w, h, u, v, uGT, vGT)
    IE = utils.IE(w, h, rec, f2)
    print(" - EE-mean: "+str(AEE))
    print(" - EE-stddev: "+str(SDEE))
    print(" - AE-mean: "+str(AAE))
    print(" - AE-stddev: "+str(SDAE))
    print(" - IE: "+str(IE))
    print(" - time: "+str(timer)+"s")

    if args.save_benchmark:
        f = open(args.save_benchmark, "w")
        f.write("EE-mean: "+str(AEE)+"\n")
        f.write("EE-stddev: "+str(SDEE)+"\n")
        f.write("AE-mean: "+str(AAE)+"\n")
        f.write("AE-stddev: "+str(SDAE)+"\n")
        f.write("IE: "+str(IE)+"\n")
        f.write("time: "+str(timer)+"s")
        f.close()

if args.out:
    print("saving flo file...")
    utils.saveFlo(w, h, u, v, args.out)
    
print("***********************************")