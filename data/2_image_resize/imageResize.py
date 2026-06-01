import os
from PIL import Image
import cv2
import numpy as np

# Charger les images et leurs tailles
roots = ["../../chest_Xray/train/NORMAL", "../../chest_Xray/train/PNEUMONIA"]
allImage = []
for root in roots:
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name.endswith(".jpeg"):
                path = root + '/' + name
                im = Image.open(path)
                w, h = im.size
                if "bacteria" in name:
                    img_type = "bacteria"
                elif "virus" in name:
                    img_type = "virus"
                else:
                    img_type = "normal"
                allImage.append([path, [w, h], img_type])

# Redimensionner les images
img_id = 0
for img in allImage:
    path, size, img_type = img[0], img[1], img[2]
    img = cv2.imread(path)

    w, h = int(1320/2), int(968/2)
    nexW, nexH = w, h
    borderW, borderH = 0, 0
    originalRatio = size[0] / size[1]
    if h * originalRatio > w:
        borderH = int(((w / originalRatio) - h) / 2)
        nexH = int(w / originalRatio)
    else:
        borderW = int(((h * originalRatio) - w) / 2)
        nexW = int(h * originalRatio)

    if size[0] > w or size[1] > h:
        res = cv2.resize(img, dsize=(nexW, nexH), interpolation=cv2.INTER_AREA)
    else:
        res = cv2.resize(img, dsize=(nexW, nexH), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    canvas = np.full((h, w), 0, dtype=np.uint8)
    y_offset = (h - nexH) // 2
    x_offset = (w - nexW) // 2
    canvas[y_offset:y_offset + nexH, x_offset:x_offset + nexW] = gray
    final = canvas

    newName = "outputs/" + img_type + "-" + str(img_id) + ".jpg"
    cv2.imwrite(newName, final)
    print("Image \'" + newName + "\' redimensionnée")
    img_id += 1
