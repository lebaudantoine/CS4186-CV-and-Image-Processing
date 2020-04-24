import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('assets/a.jpg')

background_model = np.zeros((1, 65), np.float64)
foreground_model = np.zeros((1, 65), np.float64)

# using cv2.GC_INIT_WITH_RECT
# rect = (740, 198, 250, 250)
# cv2.grabCut(image, mask, rect, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)

# using cv2.GC_INIT_WITH_MASK
custom_mask = cv2.imread('assets/mask_basketball.png', 0)

mask = np.where(((custom_mask > 0) & (custom_mask < 255)), cv2.GC_PR_FGD, 0).astype('uint8')
mask[custom_mask == 0] = cv2.GC_BGD
mask[custom_mask == 255] = cv2.GC_FGD

mask, bgdModel, fgdModel = cv2.grabCut(image, mask, None, background_model, foreground_model, 5, cv2.GC_INIT_WITH_MASK)

mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
image = image * mask[:, :, np.newaxis]
plt.imshow(image), plt.colorbar(), plt.show()
