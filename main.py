from src.registration import ImageRegistration
import matplotlib.pyplot as plt
from utils.tools import tps_warp, checkerboard
import cv2

IX_path = 'img/1a.jpg'
IY_path = 'img/1b.jpg'
IX = cv2.imread(IX_path)
IY = cv2.imread(IY_path)

image_registration = ImageRegistration()
X, Y, Z = image_registration.register(IX, IY)

registered = tps_warp(Y, Z, IY, IX.shape)
checkerboard_image = checkerboard(IX, registered, 12)

plt.subplot(141)
plt.title('Reference')
plt.imshow(cv2.cvtColor(IX, cv2.COLOR_BGR2RGB))

plt.subplot(142)
plt.title('Moving')
IY_resized = cv2.resize(IY, (IX.shape[1], IX.shape[0]))
plt.imshow(cv2.cvtColor(IY_resized, cv2.COLOR_BGR2RGB))

plt.subplot(143)
plt.title('Registered')
plt.imshow(cv2.cvtColor(registered, cv2.COLOR_BGR2RGB))

plt.subplot(144)
plt.title('Checkerboard')
plt.imshow(cv2.cvtColor(checkerboard_image, cv2.COLOR_BGR2RGB))

plt.show()
