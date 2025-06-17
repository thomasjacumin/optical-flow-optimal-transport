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
import argparse
from PIL import Image
import random

import utils

# Parse parameters
parser = argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("f1", help="frame 1")
parser.add_argument("f2", help="frame 1")
parser.add_argument("out1", help="output 1")
parser.add_argument("out2", help="output 2")

args = parser.parse_args()

f1, w, h = utils.openGrayscaleImage(args.f1)
f2, w, h = utils.openGrayscaleImage(args.f2)

f1 = f1/np.sum(f1)
f2 = f2/np.sum(f2)

scale = max( np.max(f1), np.max(f2) )

f1 = f1/scale
f2 = f2/scale

# print(f"normalizing by {scale} / sum1 {np.sum(f1)} / sum2 {np.sum(f2)}")

Image.fromarray(np.uint8(255*np.clip(f1, 0, 1).reshape([h,w])), 'L').save(args.out1)
Image.fromarray(np.uint8(255*np.clip(f2, 0, 1).reshape([h,w])), 'L').save(args.out2)