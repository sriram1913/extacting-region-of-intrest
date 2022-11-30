import numpy as np
from PIL import Image
import math
import time
import matplotlib.pyplot as plt

threshold = 0.95


def SAD(left, right,minr,maxr,minc,maxc, offset):
    num = 0
    for v in range(minr, maxr):
        for u in range(minc,maxc):
            num += abs(int(left[v, u]) - int(right[v, (u) - offset]))
    return num


def SSD(left, right,minr,maxr,minc,maxc, offset):
    num = 0
    for v in range(minr, maxr):
        for u in range(minc,maxc):
            num += (int(left[v, u]) - int(right[v, ( u) - offset])) ** 2
    return num


def Zm_SAD(left, right,minr,maxr,minc,maxc, offset):
    num1 = 0
    num2 = 0
    for v in range(minr, maxr):
        for u in range(minc,maxc):
            num1 += int(left[v, u])
            num2 += int(right[ v, ( u) - offset])
    num = 0
    for v in range(minr, maxr):
        for u in range(minc,maxc):
            num += abs(int(left[ v,  u]) - int(right[ v, ( u) - offset]) - num1 + num2)
    return num


def NCC(left, right,minr,maxr,minc,maxc, offset):
    ncc = 0
    ncc_num = 0
    ncc_den1 = 0
    ncc_den2 = 0
    for v in range(minr, maxr):
        for u in range(minc,maxc):
            ncc_num += int(left[ v,  u]) * int(right[ v, ( u) - offset])
            ncc_den1 += int(left[ v,  u]) ** 2
            ncc_den2 += int(right[ v, ( u) - offset]) ** 2

    ncc_den = math.sqrt(ncc_den1 * ncc_den2)
    if(ncc_den==0):return 0
    ncc = ncc_num / float(ncc_den)
    return ncc


def stereo_match(left_img, right_img, kernel, max_offset, func, name, file):
    curr = time.time()
    left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)
    w, h = left_img.size

    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w

    kernel_half = int(kernel / 2)

    max_depth = 0

    for y in range(0, h):
        print("\r" + name + '-' + str(kernel) + "Processing.. %d%% complete" % (y / (h - kernel_half) * 100), end="",
              flush=True)

        minr = max(1, y - kernel_half)
        maxr = min(h, y + kernel_half)

        for x in range(0, w):

            minc = max(1, x - kernel_half)
            maxc = min(h, x + kernel_half)
            best_offset = 0
            prev = 99999999
            if str == "NCC":
                prev = 0

            mind = max(-max_offset, 1 - minc)
            maxd = min(max_offset, h - maxc)



            for offset in range(mind,maxd):

                comp = func(left, right,minr,maxr,minc,maxc, -1*offset)

                if str == "NCC":
                    if threshold * comp > prev:
                        prev = comp
                        best_offset = offset
                else:
                    if comp < prev * threshold:
                        prev = comp
                        best_offset = offset

            # set depth output for this x,y location to the best match
            depth[y, x] = best_offset

    # Convert to PIL and save it
    # Image.fromarray(depth).convert('RGB').save(name + '-' + str(i) + '.png')
    t = time.time() - curr
    file.write(str(t) + '\t')
    plt.imshow(depth)
    plt.colorbar()
    plt.savefig(name + '-' + str(i) + '.png')


if __name__ == '__main__':
    file = open("data.txt", "w")
    file.write('\n\n\n')
    for i in [4,8, 12]:
        stereo_match("HHim1.jpg", "HHim2.jpg", i, 10, SAD, "SAD", file)
        stereo_match("HHim1.jpg", "HHim2.jpg", i, 10, SSD, "SSD", file)
        stereo_match("HHim1.jpg", "HHim2.jpg", i, 10, Zm_SAD, "Zm_SAD", file)
        stereo_match("HHim1.jpg", "HHim2.jpg", i, 10, NCC, "NCC", file)
        file.write("\n")

    file.close()
