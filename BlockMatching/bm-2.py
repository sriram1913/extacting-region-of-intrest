import numpy as np
from PIL import Image
import math
import time

threshold=0.95
def SAD(left, right, kernel_half, x, y, offset):
    num = 0
    for v in range(-kernel_half, kernel_half):
        for u in range(-kernel_half, kernel_half):
            # iteratively sum the sum of squared differences value for this block
            # left[] and right[] are arrays of uint8, so converting them to int saves
            # potential overflow
            num += abs(int(left[y + v, x + u]) - int(right[y + v, (x + u) - offset]))
    return num


def SSD(left, right, kernel_half, x, y, offset):
    num = 0
    for v in range(-kernel_half, kernel_half):
        for u in range(-kernel_half, kernel_half):
            # iteratively sum the sum of squared differences value for this block
            # left[] and right[] are arrays of uint8, so converting them to int saves
            # potential overflow
            num += (int(left[y + v, x + u]) - int(right[y + v, (x + u) - offset])) ** 2
    return num


def Zm_SAD(left, right, kernel_half, x, y, offset):
    num1 = 0
    num2 = 0
    for v in range(-kernel_half, kernel_half):
        for u in range(-kernel_half, kernel_half):
            # iteratively sum the sum of squared differences value for this block
            # left[] and right[] are arrays of uint8, so converting them to int saves
            # potential overflow
            num1 += int(left[y + v, x + u])
            num2 += int(right[y + v, (x + u) - offset])
    num = 0
    for v in range(-kernel_half, kernel_half):
        for u in range(-kernel_half, kernel_half):
            # iteratively sum the sum of squared differences value for this block
            # left[] and right[] are arrays of uint8, so converting them to int saves
            # potential overflow
            num += abs(int(left[y + v, x + u]) - int(right[y + v, (x + u) - offset]) - num1 + num2)
    return num


def Ls_SAD(left, right, kernel_half, x, y, offset):
    num1 = 0
    num2 = 0
    for v in range(-kernel_half, kernel_half):
        for u in range(-kernel_half, kernel_half):
            # iteratively sum the sum of squared differences value for this block
            # left[] and right[] are arrays of uint8, so converting them to int saves
            # potential overflow
            num1 += int(left[y + v, x + u])
            num2 += int(right[y + v, (x + u) - offset])
    num = 0
    for v in range(-kernel_half, kernel_half):
        for u in range(-kernel_half, kernel_half):
            # iteratively sum the sum of squared differences value for this block
            # left[] and right[] are arrays of uint8, so converting them to int saves
            # potential overflow
            num += abs(int(left[y + v, x + u]) - (num1 * int(right[y + v, (x + u) - offset] / num2)))
    return num


def NCC(left, right, kernel_half, x, y, offset):
    ncc = 0
    ncc_num = 0
    ncc_den1 = 0
    ncc_den2 = 0
    for v in range(-kernel_half, kernel_half):
        for u in range(-kernel_half, kernel_half):
            # iteratively sum the sum of squared differences value for this block
            # left[] and right[] are arrays of uint8, so converting them to int saves
            # potential overflow
            ncc_num += int(left[y + v, x + u]) * int(right[y + v, (x + u) - offset])
            ncc_den1 += int(left[y + v, x + u]) ** 2
            ncc_den2 += int(right[y + v, (x + u) - offset]) ** 2

    ncc_den = math.sqrt(ncc_den1 * ncc_den2)
    ncc = ncc_num / float(ncc_den)
    return ncc
def subpixel(x,y,z,dis):
    a=dis[x]
    b=dis[y]
    c=dis[z]
    num3=2*(b*x-c*x-a*y+c*y+a*z+b*z)
    if(num3==0):
        return min(a,b,c)
    else:
        #print(x,y,z,a,b,c,((b-c)*x**2+(c-a)*y**2+(a-b)*z**2)/num3 ,"\n")
        return ((b-c)*x**2+(c-a)*y**2+(a-b)*z**2)/num3

def stereo_match(left_img, right_img, kernel, max_offset, func, name,file):
    # Load in both images, assumed to be RGBA 8bit per channel images
    cam_par = 10  # it is diatance b/w cameras times distances b/w lens andimage plane
    curr=time.time()
    left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)
    w, h = left_img.size  # assume that both images are same size

    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w

    half=int(kernel / 2)
    kernel_half = max(max_offset,half)
    offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range

    max_depth = 0

    for y in range(kernel_half, h - kernel_half):
        print("\r" + name+'-'+str(kernel) + "Processing.. %d%% complete" % (y / (h - kernel_half) * 100), end="", flush=True)

        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev = 99999999
            if str == "NCC":
                prev = 0

            dis=[]

            for offset in range(-max_offset+1,max_offset):

                comp = func(left, right, half, x, y, offset)
                dis.append(comp)

            k=dis.index(min(dis))

            if k == 0:
                best_offset = subpixel(0,1,2,dis)
            if k == 2*max_offset-2 :
                best_offset = subpixel(2*max_offset-4, 2*max_offset-3, 2*max_offset-2,dis)
            else:
                best_offset = subpixel(k-1, k, k+1,dis)

            # set depth output for this x,y location to the best match
            depth[y, x] = best_offset
            if (depth[y, x] > max_depth):
                max_depth = depth[y, x]
    print(max_depth)
    for y in range(kernel_half, h - kernel_half):
        print("\rProcessing.. %d%% complete" % (y / (h - kernel_half) * 100), end="", flush=True)

        for x in range(kernel_half, w - kernel_half):
            depth[y, x] = (abs(depth[y, x]) / 0.00001)*255

    # Convert to PIL and save it
    Image.fromarray(depth).save(name + '-newwwwww-' + str(i) + '.png')
    t=time.time()-curr
    #file.write(str(t)+'\t')


if __name__ == '__main__':
    file=open("data.txt","a")
    #file.write('\n\n\n')
    for i in [4]:
        stereo_match("HHim1.jpg", "HHim2.jpg", i, 4, SAD, "SAD",file)
        #stereo_match("HHim1.jpg", "HHim2.jpg", i, 10, SSD, "SSD",file)
        #stereo_match("HHim1.jpg", "HHim2.jpg", i, 10, Zm_SAD, "Zm_SAD",file)
        # stereo_match("imleft.png", "imright.png", i, 10, Ls_SAD, "Ls_SAD")
        #stereo_match("HHim1.jpg", "HHim2.jpg", i, 10, NCC, "NCC",file)
        #file.write("\n")

    file.close()
