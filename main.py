# -----------------------------------------------------------------------------
# Copyright (c) 2025, Thomas Jacumin
#
# This file is part of a program licensed under the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

import numpy as np
from PIL import Image
import argparse
import time
import cv2

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
parser.add_argument("--save-reconstruction", nargs='?', help="file output of reconstruction")
parser.add_argument("--save-lum", nargs='?', help="file output of luminosity")
# Model parameters
parser.add_argument("--algo", nargs='?', help="Algorithm")
parser.add_argument("--Nt", nargs='?', type=int, default=4, help="Discretization in time")
parser.add_argument("--r", nargs='?', type=float, default=1., help="augmented langrangian parameter")
parser.add_argument("--convergence-tol", nargs='?', type=float, default=0.1, help="Stopping threshold")
parser.add_argument("--reg-epsilon", nargs='?', type=float, default=1e-3, help="Regularization for the step 1 of Benamou-Brenier")
parser.add_argument("--max-it", nargs='?', type=int, default=100, help="Maximal number of iteration")
parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, help="normalize the input images if enabled")
parser.add_argument("--alpha", nargs='?', type=float, default=0.1, help="Horn-Schunck alpha")
parser.add_argument("--lambdaa", nargs='?', type=float, default=0.2, help="Horn-Schunck lambda")

args = parser.parse_args()

np.random.seed(0)

# Create data
f1, w, h = utils.openGrayscaleImage(args.f0)
f2, w, h = utils.openGrayscaleImage(args.f1)

# w=32
# h=32

# f1 = np.zeros(w*h)
# for i in range(int(h/6),int(3*h/6)):
#     for j in range(int(h/6),int(3*h/6)):
#         f1[i*w+j] = 1
# f2 = np.zeros(w*h)
# for i in range(int(2*h/6),int(4*h/6)):
#     for j in range(int(2*h/6),int(4*h/6)):
#         f2[i*w+j] = 1  

print("***********************************")
print("Input images: ")
print(" - f0 = "+str(args.f0)+" / total mass = "+str(np.sum(f1)))
print(" - f1 = "+str(args.f1)+" / total mass = "+str(np.sum(f2)))
if args.normalize == True:
    print(" - normalize input images")
    rho1 = f1/(np.sum(f1))
    rho2 = f2/(np.sum(f2))
else:
    rho1 = f1 
    rho2 = f2
# rho1 = cv2.GaussianBlur(rho1, (3, 3), 0).flatten()
# rho2 = cv2.GaussianBlur(rho2, (3, 3), 0).flatten()  

# Start timer
start_time = time.time()
# Solve
if args.algo == 'foto':
    print(" - algorithm: FOTO")
    print(f"\t - Nt={args.Nt}")
    print(f"\t - r={args.r}")
    print(f"\t - convergence_tol={args.convergence_tol}")
    print(f"\t - reg_epsilon={args.reg_epsilon}")
    print(f"\t - max_it={args.max_it}")
    # mu, phi, q = benamou_brenier.solve(rho1, rho2, args.Nt, w, h, r=args.r, convergence_tol=args.convergence_tol, reg_epsilon=args.reg_epsilon, max_it=args.max_it)
    # u, v, m = utils.opticalflow_from_benamoubrenier(phi, args.Nt, w, h)
    u, v, m = benamou_brenier.solve(rho1, rho2, args.Nt, w, h, r=args.r, convergence_tol=args.convergence_tol, reg_epsilon=args.reg_epsilon, max_it=args.max_it)
elif args.algo == 'GN':
    print(" - algorithm: GN")
    print(f"\t - alpha={args.alpha}")
    print(f"\t - lambda={args.lambdaa}")
    classical = classical.GLLOpticalFlow(w,h)
    classical.setAlpha(args.alpha)
    classical.setLambda(args.lambdaa)
    [u, v, m] = classical.assemble(rho1, rho2).process()
else:
    assert("not implemented")
# stop timer
timer = time.time() - start_time    

# Benchmark
print("Benchmark:")
rec = utils.apply_opticalflow(f1, u, v, w, h, m)
rec = np.clip(rec, 0, 1)
IE = utils.IE(w, h, rec, f2)
print(" - time: "+str(timer)+"s")
print(" - IE: "+str(IE))

if args.ground_truth:
    wGT, hGT, uGT, vGT = utils.openFlo(args.ground_truth)
    assert(wGT == w and hGT == h)
    AEE, SDEE = utils.EE(w, h, u, v, uGT, vGT)
    AAE, SDAE = utils.AE(w, h, u, v, uGT, vGT)
    print(" - EE-mean: "+str(AEE))
    print(" - EE-stddev: "+str(SDEE))
    print(" - AE-mean: "+str(AAE))
    print(" - AE-stddev: "+str(SDAE))
    
if args.save_benchmark:
    f = open(args.save_benchmark, "w")
    if args.ground_truth:
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

if args.save_reconstruction:
    print("saving reconstruction...")
    Image.fromarray(np.uint8(255*rec.reshape([h,w])), 'L').save(args.save_reconstruction)

if args.save_lum:
    print("saving luminosity...")
    Image.fromarray(np.uint8(255*np.clip((m+1)/2, 0, 1).reshape([h,w])), 'L').save(args.save_lum)
    
print("***********************************")