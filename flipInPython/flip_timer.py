#########################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#########################################################################

# FLIP: A Difference Evaluator for Alternating Images
# High Performance Graphics, 2020.
# by Pontus Andersson, Jim Nilsson, Tomas Akenine-Moller, Magnus Oskarsson, Kalle Astrom, and Mark D. Fairchild
#
# Pointer to our paper: https://research.nvidia.com/publication/2020-07_FLIP
# code by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller

import numpy as np
from scipy import signal, ndimage
import matplotlib.pyplot as plt
from functools import partial

from multiprocessing.dummy import Pool,Process, Queue

def linrgb2lab_and_hunt(input_color):
    img = linrgb2lab(input_color)
    return hunt_adjustment(img)

def ycxcz2linrgb_and_clip(input_color):
    img = ycxcz2linrgb(input_color)
    return np.clip(img, 0.0, 1.0)

def color_space_transformb(data):
    return color_space_transform(data[0], data[1])

def srgb2ycxcz(input_color):
    dim = input_color.shape
    
    limit = 0.04045
    temp1=np.power((input_color + 0.055) / 1.055, 2.4)
    temp2=input_color / 12.92
    input_color = np.where(input_color > limit, temp1, temp2 )
    
    A=np.array([[0.41238656, 0.35759149, 0.18045049],
               [0.21263682, 0.71518298, 0.0721802 ],
               [0.01933062, 0.11919716, 0.95037259]])

    input_color = np.transpose(input_color, (2, 0, 1)) # C(H*W)
    transformed_color = np.matmul(A, input_color)
    input_color = np.transpose(transformed_color, (1, 2, 0))
    
    reference_illuminant = np.ones((dim[2], dim[0], dim[1]))
        
    reference_illuminant = np.matmul(A, reference_illuminant)
    reference_illuminant = np.transpose(reference_illuminant, (1, 2, 0))
    
    input_color = np.divide(input_color, reference_illuminant)
    y = 116 * input_color[1:2, :, :] - 16
    cx = 500 * (input_color[0:1, :, :] - input_color[1:2, :, :])
    cz = 200 * (input_color[1:2, :, :] - input_color[2:3, :, :])
    transformed_color = np.concatenate((y, cx, cz), 0)
    return transformed_color

def ycxcz2linrgb(input_color):
    dim=input_color.shape
    y = (input_color[0:1, :, :] + 16) / 116
    cx = input_color[1:2, :, :] / 500
    cz = input_color[2:3, :, :] / 200

    x = y + cx
    z = y - cz
    transformed_color = np.concatenate((x, y, z), 0)

    A=np.array([[0.41238656, 0.35759149, 0.18045049],
                [0.21263682, 0.71518298, 0.0721802 ],
                [0.01933062, 0.11919716, 0.95037259]])
    reference_illuminant = np.ones((dim[2], dim[0], dim[1]))
    reference_illuminant = np.matmul(A, reference_illuminant)
    reference_illuminant = np.transpose(reference_illuminant, (1, 2, 0))
    
    input_color = np.multiply(transformed_color, reference_illuminant)
    input_color = np.transpose(input_color, (2, 0, 1)) # C(H*W)
    
    Ainv = np.array([[ 3.24100326, -1.53739899, -0.49861587],
                     [-0.96922426,  1.87592999,  0.04155422],
                     [ 0.05563942, -0.2040112,   1.05714897]])
    transformed_color = np.matmul(Ainv, input_color)
    transformed_color = np.transpose(transformed_color, (1, 2, 0))

    return transformed_color

def linrgb2lab(input_color):
    dim = input_color.shape

    A=np.array([[0.41238656, 0.35759149, 0.18045049],
                 [0.21263682, 0.71518298, 0.0721802 ],
                 [0.01933062, 0.11919716, 0.95037259]])

    input_color = np.transpose(input_color, (2, 0, 1)) # C(H*W)
    transformed_color = np.matmul(A, input_color)
    input_color = np.transpose(transformed_color, (1, 2, 0))
    
    reference_illuminant = np.ones((dim[2], dim[0], dim[1]))
    reference_illuminant = np.matmul(A, reference_illuminant)
    reference_illuminant = np.transpose(reference_illuminant, (1, 2, 0))
    
    input_color = np.divide(input_color, reference_illuminant)
    
    delta = 6 / 29
    limit = 0.00885
    
    temp1 = np.power(input_color, 1 / 3)
    temp2 = (input_color / (3 * delta * delta)) + (4 / 29)
    input_color = np.where(input_color > limit, temp1, temp2)

    l = 116 * input_color[1:2, :, :] - 16
    a = 500 * (input_color[0:1,:, :] - input_color[1:2, :, :])
    b = 200 * (input_color[1:2, :, :] - input_color[2:3, :, :])

    transformed_color = np.concatenate((l, a, b), 0)
    return transformed_color

def color_space_transform(input_color, fromSpace2toSpace):
    dim = input_color.shape

    if fromSpace2toSpace == "srgb2linrgb":
        limit = 0.04045
        transformed_color = np.where(input_color > limit, np.power((input_color + 0.055) / 1.055, 2.4), input_color / 12.92)

    elif fromSpace2toSpace == "linrgb2srgb":
        limit = 0.0031308
        transformed_color = np.where(input_color > limit, 1.055 * (input_color ** (1.0 / 2.4)) - 0.055, 12.92 * input_color)

    elif fromSpace2toSpace == "linrgb2xyz" or fromSpace2toSpace == "xyz2linrgb":
        # Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
        # Assumes D65 standard illuminant
        a11 = 10135552 / 24577794
        a12 = 8788810  / 24577794
        a13 = 4435075  / 24577794
        a21 = 2613072  / 12288897
        a22 = 8788810  / 12288897
        a23 = 887015   / 12288897
        a31 = 1425312  / 73733382
        a32 = 8788810  / 73733382
        a33 = 70074185 / 73733382
        A = np.array([[a11, a12, a13],
                        [a21, a22, a23],
                        [a31, a32, a33]])

        input_color = np.transpose(input_color, (2, 0, 1)) # C(H*W)
        if fromSpace2toSpace == "xyz2linrgb":
            A = np.linalg.inv(A)
        transformed_color = np.matmul(A, input_color)
        transformed_color = np.transpose(transformed_color, (1, 2, 0))

    elif fromSpace2toSpace == "xyz2ycxcz":
        reference_illuminant = color_space_transform(np.ones(dim), 'linrgb2xyz')
        input_color = np.divide(input_color, reference_illuminant)
        y = 116 * input_color[1:2, :, :] - 16
        cx = 500 * (input_color[0:1, :, :] - input_color[1:2, :, :])
        cz = 200 * (input_color[1:2, :, :] - input_color[2:3, :, :])
        transformed_color = np.concatenate((y, cx, cz), 0)

    elif fromSpace2toSpace == "ycxcz2xyz":
        y = (input_color[0:1, :, :] + 16) / 116
        cx = input_color[1:2, :, :] / 500
        cz = input_color[2:3, :, :] / 200

        x = y + cx
        z = y - cz
        transformed_color = np.concatenate((x, y, z), 0)

        reference_illuminant = color_space_transform(np.ones(dim), 'linrgb2xyz')
        transformed_color = np.multiply(transformed_color, reference_illuminant)

    elif fromSpace2toSpace == "xyz2lab":
        reference_illuminant = color_space_transform(np.ones(dim), 'linrgb2xyz')
        input_color = np.divide(input_color, reference_illuminant)
        delta = 6 / 29
        limit = 0.00885

        input_color = np.where(input_color > limit, np.power(input_color, 1 / 3), (input_color / (3 * delta * delta)) + (4 / 29))

        l = 116 * input_color[1:2, :, :] - 16
        a = 500 * (input_color[0:1,:, :] - input_color[1:2, :, :])
        b = 200 * (input_color[1:2, :, :] - input_color[2:3, :, :])

        transformed_color = np.concatenate((l, a, b), 0)

    elif fromSpace2toSpace == "lab2xyz":
        y = (input_color[0:1, :, :] + 16) / 116
        a =  input_color[1:2, :, :] / 500
        b =  input_color[2:3, :, :] / 200

        x = y + a
        z = y - b

        xyz = np.concatenate((x, y, z), 0)
        delta = 6 / 29
        xyz = np.where(xyz > delta,  xyz ** 3, 3 * delta ** 2 * (xyz - 4 / 29))

        reference_illuminant = color_space_transform(np.ones(dim), 'linrgb2xyz')
        transformed_color = np.multiply(xyz, reference_illuminant)

    elif fromSpace2toSpace == "srgb2xyz":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color,'linrgb2xyz')
    elif fromSpace2toSpace == "srgb2ycxcz":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
    elif fromSpace2toSpace == "linrgb2ycxcz":
        transformed_color = color_space_transform(input_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
    elif fromSpace2toSpace == "srgb2lab":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    elif fromSpace2toSpace == "linrgb2lab":
        transformed_color = color_space_transform(input_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    elif fromSpace2toSpace == "ycxcz2linrgb":
        transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
    elif fromSpace2toSpace == "lab2srgb":
        transformed_color = color_space_transform(input_color, 'lab2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2srgb')
    elif fromSpace2toSpace == "ycxcz2lab":
        transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    else:
        print('The color transform is not defined!')
        transformed_color = input_color

    return transformed_color

def generate_spatial_filter(pixels_per_degree, channel):
    a1_A = 1 
    b1_A = 0.0047
    a2_A = 0
    b2_A = 1e-5 # avoid division by 0
    a1_rg = 1
    b1_rg = 0.0053
    a2_rg = 0
    b2_rg = 1e-5 # avoid division by 0
    a1_by = 34.1
    b1_by = 0.04
    a2_by = 13.5
    b2_by = 0.025
    if channel == "A": #Achromatic CSF
        a1 = a1_A
        b1 = b1_A
        a2 = a2_A
        b2 = b2_A
    elif channel == "RG": #Red-Green CSF
        a1 = a1_rg
        b1 = b1_rg
        a2 = a2_rg
        b2 = b2_rg
    elif channel == "BY": # Blue-Yellow CSF
        a1 = a1_by
        b1 = b1_by
        a2 = a2_by
        b2 = b2_by

    # Determine evaluation domain
    max_scale_parameter = max([b1_A, b2_A, b1_rg, b2_rg, b1_by, b2_by])
    r = np.ceil(3 * np.sqrt(max_scale_parameter / (2 * np.pi**2)) * pixels_per_degree)
    r = int(r)
    deltaX = 1.0 / pixels_per_degree
    
    x0=np.arange(-r,r+1)
    z = (x0 * deltaX)**2
    # Generate weights
    g = np.sqrt(a1 * np.sqrt(np.pi / b1)) * np.exp(-np.pi**2 * z / b1)
    gg= np.sqrt(a2 * np.sqrt(np.pi / b2)) * np.exp(-np.pi**2 * z / b2)
    div=np.sqrt(np.sum(g)**2+np.sum(gg)**2)
    g = g / div
    gg= gg/ div

    return [g, gg], r

def laConvo1D(laData):
        data,u,v=laData
        return ndimage.convolve1d(ndimage.convolve1d(data, v, mode='nearest',axis=1), u, mode='nearest',axis=0)

def hunt_adjustment(img):
    # Applies Hunt adjustment to L*a*b* image img
    
    # Extract luminance component
    L = img[0:1, :, :]
    
    # Apply Hunt adjustment
    img_h = np.zeros(img.shape)
    img_h[0:1, :, :] = L
    img_h[1:2, :, :] = np.multiply((0.01 * L), img[1:2, :, :])
    img_h[2:3, :, :] = np.multiply((0.01 * L), img[2:3, :, :])

    return img_h

def hyab(reference, table):
    # Computes HyAB distance between L*a*b* images reference and table
    delta = reference - table
    return abs(delta[0:1, :, :]) + np.linalg.norm(delta[1:3, :, :], axis=0)

def redistribute_errors(power_deltaE_hyab, cmax):
    # Set redistribution parameters
    pc = 0.4
    pt = 0.95
    
    # Re-map error to 0-1 range. Values between 0 and
    # pccmax are mapped to the range [0, pt],
    # while the rest are mapped to the range (pt, 1]
    deltaE_c = np.zeros(power_deltaE_hyab.shape)
    pccmax = pc * cmax
    deltaE_c = np.where(power_deltaE_hyab < pccmax, (pt / pccmax) * power_deltaE_hyab, pt + ((power_deltaE_hyab - pccmax) / (cmax - pccmax)) * (1.0 - pt))

    return deltaE_c

def make_spatial_filter_parameters(reference, test, s_a, s_rg, s_by, masks):

    imgref1 = reference[0:1, :, :].squeeze(0)
    imgref2 = reference[1:2, :, :].squeeze(0)
    imgref3 = reference[2:3, :, :].squeeze(0)
                
    imgtest1 = test[0:1, :, :].squeeze(0)
    imgtest2 = test[1:2, :, :].squeeze(0)
    imgtest3 = test[2:3, :, :].squeeze(0)

    param=list()
    param.append( [imgref1, s_a[0], s_a[0]] )
    param.append( [imgtest1, s_a[0], s_a[0]] )

    param.append( [imgref2, s_rg[0], s_rg[0]] )
    param.append( [imgtest2, s_rg[0], s_rg[0]] )

    param.append( [imgref3, s_by[0], s_by[0]] )
    param.append( [imgtest3, s_by[0], s_by[0]] )

    ### skips computation if vector is null ###
    if not masks[0]:
        param.append( [imgref1, s_a[1], s_a[1]] )
        param.append( [imgtest1, s_a[1], s_a[1]] )
    if not masks[1]:
        param.append( [imgref2, s_rg[1], s_rg[1]] )
        param.append( [imgtest2, s_rg[1], s_rg[1]] )
    if not masks[2]:
        param.append( [imgref3, s_by[1], s_by[1]] )
        param.append( [imgtest3, s_by[1], s_by[1]] )
    return param

def get_spatial_filter_results(table, masks, dim):
    img_tilde_opponent_ref = np.zeros((dim[0], dim[1], dim[2]))
    img_tilde_opponent_test = np.zeros((dim[0], dim[1], dim[2]))
    i = 6

    img_tilde_opponent_ref[0:1, :, :]  = table[0] if masks[0] else table[0]+table[i]
    img_tilde_opponent_test[0:1, :, :] = table[1] if masks[0] else table[1]+table[i+1]
    if not masks[0]: i += 2

    img_tilde_opponent_ref[1:2, :, :]  = table[2] if masks[1] else table[2]+table[i]
    img_tilde_opponent_test[1:2, :, :] = table[3] if masks[1] else table[3]+table[i+1]
    if not masks[1]: i += 2

    img_tilde_opponent_ref[2:3, :, :]  = table[4] if masks[2] else table[4]+table[i]
    img_tilde_opponent_test[2:3, :, :] = table[5] if masks[2] else table[5]+table[i+1]
    return img_tilde_opponent_ref, img_tilde_opponent_test

def make_feature_detection_vectors(pixels_per_degree, feature_type):
    '''return u,v two normalized vectors such that u x v is the kernel for 2D convolution
    '''
        # Finds features of type feature_type in image img based on current PPD
    
    # Set peak to trough value (2x standard deviations) of human edge
    # detection filter
    w = 0.082
    
    # Compute filter radius
    sd = 0.5 * w * pixels_per_degree
    radius = int(np.ceil(3 * sd))

    # Compute 2D Gaussian
    #[x, y] = np.meshgrid(range(-radius, radius+1), range(-radius, radius+1))
    x = np.arange(-radius,radius+1)
    
    g = np.exp(-(x *x) / (2 * sd * sd))
    
    if feature_type == 'edge': # Edge detector
        # Compute partial derivative in x-direction
        Gx = np.multiply(-x, g)
    else: # Point detector
        # Compute second partial derivative in x-direction
        Gx = np.multiply(x ** 2 / (sd * sd) - 1, g)
    # Normalize positive weights to sum to 1 and negative weights to sum to -1
    negative_weights_sum = -np.sum(Gx[Gx < 0])
    positive_weights_sum = np.sum(Gx[Gx > 0])
    Gx = np.where(Gx < 0, Gx / negative_weights_sum, Gx / positive_weights_sum)
    
    negative_weights_sum = -np.sum(g[g < 0])
    positive_weights_sum = np.sum(g[g > 0])
    if not negative_weights_sum==0.:
        g = np.where(g < 0, g / negative_weights_sum, g/ positive_weights_sum)
    else:
        g /= positive_weights_sum
    
    u=Gx
    v=g
    return u,v
from time import time

def compute_flip(reference, test, pixels_per_degree):
    assert reference.shape == test.shape

    # Set color and feature exponents
    qc = 0.7
    qf = 0.5
    times=dict()
    # Transform reference and test to opponent color space
    def tim(timerName): 
        end=time() 
        times[timerName]=end-start
        return time()
    with Pool() as p:
    #if True:
        start = time()
        reference, test = p.map(srgb2ycxcz, [reference,test])
        start=tim("firstColorTransform")
        
    #reference = color_space_transform(reference, 'srgb2ycxcz')
    #test = color_space_transform(test, 'srgb2ycxcz')

    ##### * ------ Color pipeline ------ * #####
        ###* START spatial filter START *###
        s_a, radius_a = generate_spatial_filter(pixels_per_degree, 'A')
        s_rg, radius_rg = generate_spatial_filter(pixels_per_degree, 'RG')
        s_by, radius_by = generate_spatial_filter(pixels_per_degree, 'BY')    
        
        dim = reference.shape
        masks = (s_a[1]==0.).all(), (s_rg[1]==0.).all(), (s_by[1]==0.).all()

        params = make_spatial_filter_parameters(reference, test, s_a, s_rg, s_by, masks)
        start=tim("initSpatialFilter")
        table = p.map(laConvo1D, params)
        start=tim("convSpatialFilter")
        ref_ycxcz, test_ycxcz = get_spatial_filter_results(table, masks, dim)
        start=tim("getResultsSpatialFilter")
        filtered_reference, filtered_test = p.map(ycxcz2linrgb_and_clip, [ref_ycxcz, test_ycxcz])
        start=tim("spatialResultColorTransform")
        ###* END spatial filter reference END *###
        
    # Perceptually Uniform Color Space #
        param=[filtered_reference, filtered_test]
        preprocessed_reference, preprocessed_test = p.map(linrgb2lab_and_hunt,param) 
        start=tim("preprocessColorSpace")
        #preprocessed_reference = hunt_adjustment(color_space_transform(filtered_reference, 'linrgb2lab'))
        #preprocessed_test = hunt_adjustment(color_space_transform(filtered_test, 'linrgb2lab'))

    # Color metric
        deltaE_hyab = hyab(preprocessed_reference, preprocessed_test)
        hunt_adjusted_green = hunt_adjustment(linrgb2lab(np.array([[[0.0]], [[1.0]], [[0.0]]])))
        hunt_adjusted_blue = hunt_adjustment(linrgb2lab(np.array([[[0.0]], [[0.0]], [[1.0]]])))
        cmax = np.power(hyab(hunt_adjusted_green, hunt_adjusted_blue), qc)
        deltaE_c = redistribute_errors(np.power(deltaE_hyab, qc), cmax)
        start=tim("colorMetric")
    # --- Feature pipeline ---
    # Extract and normalize achromatic component
        reference_y = (reference[0:1, :, :] + 16) / 116
        test_y = (test[0:1, :, :] + 16) / 116
        start=tim("extractNormalize")
    # Edge and point detection
    # Edge and point kernels/vectors
        edge_u, edge_v = make_feature_detection_vectors( pixels_per_degree, 'edge' )
        point_u, point_v = make_feature_detection_vectors( pixels_per_degree, 'point' )
        
        param=[ [reference_y[0], edge_u, edge_v], [reference_y[0], edge_v, edge_u],
                [reference_y[0], point_u, point_v], [reference_y[0], point_v, point_u],
                [test_y[0], edge_u, edge_v], [test_y[0], edge_v, edge_u],
                [test_y[0], point_u, point_v], [test_y[0], point_v, point_u]]
        start=tim("makeFeatureVectorsAndParams")   
        reslist = p.map(laConvo1D, param)
        start=tim("convFeatureDetection")
        edges_reference,points_reference,edges_test,points_test = [np.stack(reslist[i:i+2]) for i in range(0,len(reslist), 2) ]
        start=tim("stackElements")
        #edges_reference = feature_detection(reference_y, pixels_per_degree, 'edge')
        #points_reference = feature_detection(reference_y, pixels_per_degree, 'point')
        #edges_test = feature_detection(test_y, pixels_per_degree, 'edge')
        #points_test = feature_detection(test_y, pixels_per_degree, 'point')
        
        param = [edges_reference, edges_test, points_test, points_reference]
        normiz = partial(np.linalg.norm, axis=0)
        norms = list(p.map(normiz, param))
        start=tim("normComputation")
    p.close()
    p.join()

    # Feature metric
    deltaE_f = np.maximum(abs(norms[0] - norms[1]), abs(norms[2] - norms[3]))
    #deltaE_f = np.maximum(abs(np.linalg.norm(edges_reference, axis=0) - np.linalg.norm(edges_test, axis=0)), abs(np.linalg.norm(points_test, axis=0) - np.linalg.norm(points_reference, axis=0)))
    deltaE_f = np.power(((1 / np.sqrt(2)) * deltaE_f), qf)
    
    # --- Final error ---
    error = np.power(deltaE_c, 1 - deltaE_f)
    start=tim("finalError")
    #print(times)
    return error
