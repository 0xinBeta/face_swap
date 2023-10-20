import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

img_path = 'test-img.png'
img = cv2.imread(img_path)

# if img is None:
#     print("Image not found or could not be opened.")
# else:
face = app.get(img)
    # if len(face) == 0:
    #     print("No faces found.")
    # else:
    #     fig, axs = plt.subplots(1, 1, figsize=(12, 5))
    #     for i, face_info in enumerate(face):
    #         bbox = face_info['bbox']
    #         bbox = [int(b) for b in bbox]
    #         if i == 0:
    #             axs.imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
    #             axs.axis('off')
        # plt.show()  # Display the first face bounding box

model_path = "/Users/arefgholami/Desktop/NOAAA/face-swap-test/fasw/lib/python3.10/site-packages/insightface/model_zoo/inswapper_128.onnx"
swapper = insightface.model_zoo.get_model(model_path, download=False, download_zip=False)

source_face_img = cv2.imread('source-face.png')
source_face = app.get(source_face_img)

# if len(source_face) == 0:
#     print("No faces found.")
# else:
#     fig, axs = plt.subplots(1, 1, figsize=(12, 5))
#     for i, face_info in enumerate(source_face):
#         bbox = face_info['bbox']
#         bbox = [int(b) for b in bbox]
#         if i == 0:
#             axs.imshow(source_face_img[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])  # Use source_face_img here
#             axs.axis('off')
    # plt.show() 

target_face = face[0]
source_target_face = source_face[0]

res = img.copy()
res = swapper.get(res, target_face, source_target_face, paste_back=True)

cv2.imwrite('output_image.png', res)
# plt.imshow(res[:,:,::-1])
# plt.show()