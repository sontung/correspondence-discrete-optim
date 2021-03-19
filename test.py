import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from os import listdir
from os.path import isfile, join


def kmeans(img):

    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    K = 2
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_PP_CENTERS )
    # Now convert back into uint8, and make original image
    center = np.uint8(center)

    # # Now separate the data, Note the flatten()
    # A = Z[label.ravel() == 0]
    # B = Z[label.ravel() == 1]
    # # Plot the data
    # plt.scatter(A[:, 0], A[:, 1])
    # plt.scatter(B[:, 0], B[:, 1], c='r')
    # plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
    # plt.xlabel('Height'), plt.ylabel('Weight')
    # plt.show()

    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


def edge_contour_hsv(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    img = cv.convertScaleAbs(img, alpha=2., beta=1.0)

    edges = cv.Canny(img[:, :, 0], 100, 200)

    # Convert BGR to HSV and parse HSV
    h, s, v = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Plot result images
    plt.subplot("221")
    plt.imshow(h)
    plt.subplot("222")
    plt.imshow(s)
    plt.subplot("223")
    plt.imshow(v)
    plt.subplot("224")
    plt.imshow(img)
    # plt.show()

    # new_im = cv.cvtColor(new_im, cv.COLOR_BGR2GRAY)
    # im2, contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)[:70]
    # cv.drawContours(img, contours, -1, (0,255,0), 3)

    # im2, contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)[:10]
    #
    # max_contour = np.vstack([du[0] for du in contours])
    #
    # rect = cv.minAreaRect(max_contour)
    # box = cv.boxPoints(rect)
    # box = np.int0(box)
    # box = tuple([tuple(b) for b in box])
    #
    # xmin = min([b[0] for b in box])
    # xmax = max([b[0] for b in box])
    # ymin = min([b[1] for b in box])
    # ymax = max([b[1] for b in box])
    #
    # x = xmin
    # y = ymin
    # w = xmax - xmin
    # h = ymax - ymin
    #
    # mask = np.zeros(img.shape[:2], np.uint8)
    # bgdModel = np.zeros((1, 65), np.float64)
    # fgdModel = np.zeros((1, 65), np.float64)
    # rect = (x, y, w, h)
    # cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    # mask2 = np.where((mask == 3) | (mask == 1), 1, 0).astype('uint8')
    # img = img * mask2[:, :, np.newaxis]

    # img[:, :, 2] = 0.1*img[:, :, 2]

    return img


def edge_contour_rgb(img):
    edges = cv.Canny(img, 100, 200)

    # im2, contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)[:70]
    #
    # max_contour = np.vstack([du[0] for du in contours])
    #
    # rect = cv.minAreaRect(max_contour)
    # box = cv.boxPoints(rect)
    # box = np.int0(box)
    # box = tuple([tuple(b) for b in box])
    #
    # xmin = min([b[0] for b in box])
    # xmax = max([b[0] for b in box])
    # ymin = min([b[1] for b in box])
    # ymax = max([b[1] for b in box])
    #
    # x = xmin
    # y = ymin
    # w = xmax - xmin
    # h = ymax - ymin
    #
    # mask = np.zeros(img.shape[:2],np.uint8)
    # bgdModel = np.zeros((1,65),np.float64)
    # fgdModel = np.zeros((1,65),np.float64)
    # rect = (x, y, w, h)
    # cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    # mask2 = np.where((mask==3)|(mask==1),1,0).astype('uint8')
    # img = img*mask2[:,:,np.newaxis]

    return img


mypath = "/home/sontung/Downloads/all_images"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
saved_path = "debugs"
for im in onlyfiles:
    print(im)
    imdir = "%s/%s" % (mypath, im)

    ori_img = cv.imread(imdir)
    ori_img = cv.resize(ori_img, (ori_img.shape[1]//10, ori_img.shape[0]//10))

    # img = ori_img.copy()
    # img = cv.cvtColor(ori_img, cv.COLOR_BGR2HSV)
    # img = cv.resize(img, (img.shape[1]//6, img.shape[0]//6))
    # print(img.shape)
    #
    # ori_img = img.copy()
    # img = cv.edgePreservingFilter(img)
    # # img = cv.ximgproc.anisotropicDiffusion(img, 0.1, 0.1, 10)
    # edges = cv.Canny(img,100,200)
    #
    # # new_im = cv.cvtColor(new_im, cv.COLOR_BGR2GRAY)
    # im2, contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)[:70]
    #
    # max_contour = np.vstack([du[0] for du in contours])
    #
    # rect = cv.minAreaRect(max_contour)
    # box = cv.boxPoints(rect)
    # box = np.int0(box)
    # box = tuple([tuple(b) for b in box])
    #
    # xmin = min([b[0] for b in box])
    # xmax = max([b[0] for b in box])
    # ymin = min([b[1] for b in box])
    # ymax = max([b[1] for b in box])
    #
    # x = xmin
    # y = ymin
    # w = xmax - xmin
    # h = ymax - ymin
    #
    # print("running grabcut")
    # mask = np.zeros(img.shape[:2],np.uint8)
    # bgdModel = np.zeros((1,65),np.float64)
    # fgdModel = np.zeros((1,65),np.float64)
    # rect = (x, y, w, h)
    # cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    # mask2 = np.where((mask==3)|(mask==1),1,0).astype('uint8')
    # img = img*mask2[:,:,np.newaxis]
    # res = img.copy()
    #
    # rect = cv.minAreaRect(max_contour)
    # box = cv.boxPoints(rect)
    # box = np.int0(box)

    # cv.drawContours(img, contours, -1, (0,255,0), 3)
    # cv.drawContours(img,[box],0,(0,0,255),2)

    out = np.vstack([np.hstack([edge_contour_rgb(ori_img), edge_contour_hsv(ori_img)]),
                     np.hstack([kmeans(edge_contour_rgb(ori_img)),
                                kmeans(edge_contour_hsv(ori_img))])
                     ])
    cv.imwrite('%s/%s' % (saved_path, im), out)
    cv.imshow("t", out)
    cv.waitKey()
    cv.destroyAllWindows()
