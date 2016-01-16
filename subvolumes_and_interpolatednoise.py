from math import sin, cos, ceil
import cv2
import numpy as np
from PIL import Image
from scipy import interpolate
from random import randint


# Use efficient nearest-neighbor interpolation to reciver an image from coordinates. Used to distort labels of training data.
# Input: 1. im: Original image data
#        2. x: x-coordinates with noise
#        3. y: y-coordinates with noise
# Output: Recovered image using distorted coordinates and nearest-neighbor interpolation.
def lookupNearest(im, x, y):
    xi = np.rint(x).astype(int)
    yi = np.rint(y).astype(int)
    xi = np.clip(xi, 0, im.shape[1]-1)
    yi = np.clip(yi, 0, im.shape[0]-1)

    return im[yi, xi]

# Use efficient bilinear interpolation to recover an image from coordinates. Used to distort training data.
# Input: 1. im: Original image data
#        2. x: x-coordinates with noise
#        3. y: y-coordinates with noise
# Output: Recovered image using distorted coordinates and bilinear interpolation
def bilinear_interpolate(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

# Apply distorted noise to a particular training image and its corresponding labels
# Input: 1. Image to apply smooth (cubic interpolated) noise, recovered using bilinear interpolat
#        2. Image to apply 
# Output: Image after noise addition
def applyInterpolatedNoise(image, targets, noise_factor=10, stride=64):

    size = image.shape[0]

    x = np.arange(0, size, stride)
    y = np.arange(0, size, stride)

    interp_size = x.shape[0]

    delta_x = np.random.normal(scale=noise_factor, size=(interp_size, interp_size))
    delta_y = np.random.normal(scale=noise_factor, size=(interp_size, interp_size))

    f_x = interpolate.interp2d(x, y, delta_x, kind='cubic')
    f_y = interpolate.interp2d(x, y, delta_y, kind='cubic')

    noise_x = f_x(np.arange(0, size), np.arange(0, size))
    noise_y = f_y(np.arange(0, size), np.arange(0, size))

    grid_x = np.asarray([range(size)]*size)
    grid_y = np.asarray([size*[i] for i in range(size)])

    x_jitter = grid_x + noise_x
    y_jitter = grid_y + noise_y

    labels = [None for t in range(targets.shape[0])]
    for t in range(targets.shape[0]):
        labels[t] = lookupNearest(targets[t, :, :], x_jitter, y_jitter)

    return bilinear_interpolate(image, x_jitter, y_jitter), labels

def applyInterpolatedNoiseToStack(images, targets, noise_factor=10, stride=256):

    data_stack, data_col, data_row = images[0].shape
    num_labels, label_stack, label_col, label_row = targets.shape

    data_noises = np.zeros([1, data_stack, data_col, data_row])
    label_noises = np.zeros([num_labels, label_stack, label_col, label_row])

    # data_noises = [None for x in range(num_images)]
    # label_noises = [None for x in range(num_images)]

    for s in range(data_stack):
        data_noises[0, s, :, :], labels = applyInterpolatedNoise(images[0, s, :, :], targets[:, s, :, :])
        for n in range(num_labels):
            label_noises[n, s, :, :] = labels[n]
        
    # for s in range(len(images)):
    #     data_noise[s], label_noise[s] = applyInterpolatedNoise(images[s], targets[s])

    return data_noises, label_noises


def rotatePoint(cx, cy, deg, px, py):
    s = sin(deg)
    c = cos(deg)

    px -= cx
    py -= cy

    xnew = px * c - py * s
    ynew = px * s + py * c

    px = xnew + cx
    py = ynew + cy

    return px, py

def isPointInside(dim, point, deg):

    x, y = point
    cenx, ceny = (dim[0]/2, dim[1]/2)
    deg = deg * 0.0174533

    # Corners are ax,ay,bx,by,dx,dy
    ax, ay = rotatePoint(cenx, ceny, deg, 0., 0.)
    bx, by = rotatePoint(cenx, ceny, deg, dim[0], 0.)
    dx, dy = rotatePoint(cenx, ceny, deg, 0., dim[1])

    bax = bx - ax
    bay = by - ay
    dax = dx - ax
    day = dy - ay

    if (x - ax) * bax + (y - ay) * bay < 0.0:
        return False
    if (x - bx) * bax + (y - by) * bay > 0.0:
        return False
    if (x - ax) * dax + (y - ay) * day < 0.0:
        return False
    if (x - dx) * dax + (y - dy) * day > 0.0:
        return False

    return True

def arePointsInside(dim, points, deg):
    for point in points:
        if not isPointInside(dim, point, deg):
            return False
    return True

def getFourCorners(corner, dim):
    return [corner, (corner[0]+dim[0], corner[1]), (corner[0], corner[1]+dim[1]), (corner[0]+dim[0], corner[1]+dim[1])]


# Output distorted volumes from a training image stack and its corresponding labeled target stack, for the purpose of training set augmentation.
# Input: 
#        1. image_stack: Image stack, as a list of numpy arrays
#        2. sub_dim: the dimensions of the desired sub-volumes
#        3. num_angles: the nunber of random angles to generate subvolumes for
#        4. num_samples: the number of samples to get from each angle
# Output:
#        Outputs sampled images into a train_set directory.
def outputSampleVolumes(image_stack, target_stack, sub_dim, num_angles=10, num_samples=5):

    stack_size = len(image_stack)
    cols, rows = image_stack[0].shape
    dim = [cols, rows]

    # First, apply noise to all the images, and store in the image_stack
    for s in range(stack_size):
        image_stack[s], target_stack[s] = applyInterpolatedNoise(image_stack[s], target_stack[s])

    # cv2.imwrite('test_image.tif', image_stack[0])
    # cv2.imwrite('test_target.tif', target_stack[0])
    # exit()

    # Loop through random angles, generating rotated versions of the images
    for i in range(num_angles):

        rot_ims = [None for s in range(stack_size)]
        rot_targs = [None for s in range(stack_size)]

        angle = randint(0, 359)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)

        for s in range(stack_size):
            rot_ims[s] = cv2.warpAffine(image_stack[s], M, (cols, rows))
            rot_targs[s] = cv2.warpAffine(target_stack[s], M, (cols, rows))

        # Loop through the number of samples, generating complete subvolumes
        j = 0
        while j != num_samples:
            x_corner = randint(0, cols - sub_dim[0] - 1)
            y_corner = randint(0, rows - sub_dim[1] - 1)
            # If selected subpatch is completely inside, then output
            if arePointsInside(dim, getFourCorners((x_corner, y_corner), sub_dim), angle):
                for s in range(stack_size):
                    im_patch = rot_ims[s][x_corner:x_corner+sub_dim[0], y_corner:y_corner+sub_dim[1]]
                    targ_patch = rot_targs[s][x_corner:x_corner+sub_dim[0], y_corner:y_corner+sub_dim[1]]
                    
                    cv2.imwrite('train_set/output_crop_rot{0}_#{1}_{2}.tif'.format(angle, j, s), im_patch)
                    cv2.imwrite('targ_set/output_target_crop_rot{0}_#{1}_{2}.tif'.format(angle, j, s), targ_patch)
                j += 1


# Get distorted volumes from a training image stack and its corresponding labeled target stack, for the purpose of training set augmentation.
# Input: 
#        1. image_stack: Image stack, as a list of numpy arrays
#        2. data_patchsize: the dimensions of the desired sub-volumes in the raw data
#        3. target_patchsize: the dimensions of the desired sub-volumes in the target data
#        4. num_angles: the nunber of random angles to generate subvolumes for
#        5. num_samples: the number of samples to get from each angle
# Output:
#        Returns the data images, label images, and all the offsets used in the data
def getSampleVolumes(image_stack, target_stack, input_padding, data_patchsize, target_patchsize, num_samples):

    data_stack, data_rows, data_cols = image_stack[0].shape
    data_dim = [data_rows, data_cols]

    num_labels, label_stack, label_rows, label_cols = target_stack.shape
    label_dim = [data_rows, data_cols]

    assert(data_stack == label_stack)
    assert(data_rows == label_rows)
    assert(data_cols == label_cols)

    
    data_samples = [None for x in range(num_samples)]
    label_samples = [None for x in range(num_samples)]
    offsets = [None for x in range(num_samples)]

    # Assume all the images already have noise (applyInterpolatedNoise should have been called already)


    # Loop through random angles, generating rotated versions of the images
    for i in range(num_samples):

        rot_ims = np.zeros([1, data_stack, data_rows, data_cols])
        rot_targs = np.zeros([num_labels, label_stack, label_rows, label_cols])

        angle = randint(0, 359)
        M = cv2.getRotationMatrix2D((data_rows/2,data_cols/2),angle,1)

        for s in range(data_stack):
            rot_ims[0, s, :, :] = cv2.warpAffine(image_stack[0, s, :, :], M, (data_cols, data_rows))
            # rot_targs[s] = [cv2.warpAffine(target_stack[0, s, :, :], M, data_dim)]
            for n in range(num_labels):
                rot_targs[n, s, :, :] = cv2.warpAffine(target_stack[n, s, :, :], M, (data_cols, data_rows))

        # Loop through until we find offsets that work
        while True:

            data_offset = [randint(0, data_stack - data_patchsize[0]), randint(0, data_cols - data_patchsize[1] - 1), randint(0, data_rows - data_patchsize[2] - 1)]
            label_offset = [data_offset[di] + int(ceil(input_padding[di] / float(2))) for di in range(0, len(input_padding))]

            # If data patch is within data size and label patch is within label size, then we have valid offsets
            # NOTE: may not need to check for label patch, since it will be smaller (and centered at data patch, at least it should be)
            # if arePointsInside(data_dim, getFourCorners(data_offset[1:], data_patchsize[1:]), angle) and arePointsInside(label_dim, getFourCorners(label_offset[1:], label_patchsize[1:]), angle):
            if arePointsInside(data_dim, getFourCorners(data_offset[1:], data_patchsize[1:]), angle):
                # Get patches of volume based on offsets and sizes of volumes
                data_patch = rot_ims[0, data_offset[0]:data_offset[0]+data_patchsize[0], data_offset[1]:data_offset[1]+data_patchsize[1], data_offset[2]:data_offset[2]+data_patchsize[2]]
                label_patch = rot_targs[:, label_offset[0]:label_offset[0]+label_patchsize[0], label_offset[1]:label_offset[1]+label_patchsize[1], label_offset[2]:label_offset[2]+label_patchsize[2]]

                data_samples[i] = data_patch
                label_samples[i] = label_patch
                offsets[i] = data_offset

                break

    return data_samples, label_samples, offsets


# # Testing code
# tif = TIFF.open('validate_raw_raw.tif', mode='r')

# image_stack = []
# for image in tif.iter_images():
#     image_stack.append(image)
# tif.close()

# tif = TIFF.open('validate_target.tif', mode='r')

# target_stack = []
# for image in tif.iter_images():
#     target_stack.append(image)
# tif.close()

# outputSampleVolumes(image_stack, target_stack, [256, 256], 2, 5)

