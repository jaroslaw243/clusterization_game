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
    kernel = cv2.getStructuringElement(1, (17, 17))

    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    return cv2.inpaint(org_image, thresh2, 1, cv2.INPAINT_TELEA)
