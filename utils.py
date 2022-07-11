import sys
import gc
import copy
import cv2


def actualsize(input_obj):
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    return memory_size


def hair_removal(image):
    org_image = copy.copy(image)

    # kernel with size 17x17 was working well on image with size 574x765,
    # so for other resolutions it is scaled according to given image width
    scale = image.shape[1] / 765
    kernel_size = round(17 * scale)
    kernel = cv2.getStructuringElement(1, (kernel_size, kernel_size))

    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    return cv2.inpaint(org_image, thresh2, 1, cv2.INPAINT_TELEA)


def hair_removal_and_fill_image_seg_boundaries(image, seg):
    org_image = copy.copy(image)
    org_seg = copy.copy(seg)
    inv_seg = cv2.bitwise_not(org_seg)

    filled_cor1 = cv2.floodFill(inv_seg, None, (0, 0), 1)
    filled_cor2 = cv2.floodFill(inv_seg, None, (inv_seg.shape[1] - 1, 0), 1)
    filled_cor3 = cv2.floodFill(inv_seg, None, (0, inv_seg.shape[0] - 1), 1)
    filled_cor4 = cv2.floodFill(inv_seg, None, (inv_seg.shape[1] - 1, inv_seg.shape[0] - 1), 1)

    comb1 = filled_cor1[1] & filled_cor2[1]
    comb2 = comb1 & filled_cor3[1]
    comb3 = comb2 & filled_cor4[1]

    scale = image.shape[1] / 765
    kernel_size = round(17 * scale)
    kernel = cv2.getStructuringElement(1, (kernel_size, kernel_size))

    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    combined_mask = thresh2
    combined_mask[comb3 == 1] = 255

    return cv2.inpaint(org_image, combined_mask, 1, cv2.INPAINT_TELEA)
