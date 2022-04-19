import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def butterworthLP(D0, imgShape, n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 / (1 + (distance((y, x), center) / D0) ** (2 * n))
    return base


img = cv.imread(input("Enter Relative Path of Image:-  "), 0)

fourier_transform = np.fft.fft2(img)
center_shift = np.fft.fftshift(fourier_transform)

fourier_noisy = 20 * np.log(np.abs(center_shift))

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

print("Enter type of noise:- \n1.Vertical Noise\n2.Horizontal Noise\n3.Right Diagonal Noise\n4.Left Diagonal Noise\n")
val = int(input("Enter the Value:- "))

if val == 1:
    # horizontal mask
    center_shift[crow - 4:crow + 4, 0:ccol - 10] = 1
    center_shift[crow - 4:crow + 4, ccol + 10:] = 1
elif val == 2:
    # vertical mask
    center_shift[:crow - 10, ccol - 4:ccol + 4] = 1
    center_shift[crow + 10:, ccol - 4:ccol + 4] = 1
elif val == 3:
    # diagonal-1 mask
    for x in range(0, rows):
        for y in range(0, cols):
            if (x == y):
                for i in range(0, 10):
                    center_shift[x - i, y] = 1
elif val == 4:
    # diagonal-2 mask
    for x in range(0, rows):
        for y in range(0, cols):
            if (x + y == cols):
                for i in range(0, 10):
                    center_shift[x - i, y] = 1

else:
    print("Invalid Input")

filtered = center_shift * butterworthLP(80, img.shape, 10)

f_shift = np.fft.ifftshift(center_shift)
denoised_image = np.fft.ifft2(f_shift)
denoised_image = np.real(denoised_image)

f_ishift_blpf = np.fft.ifftshift(filtered)
denoised_image_blpf = np.fft.ifft2(f_ishift_blpf)
denoised_image_blpf = np.real(denoised_image_blpf)

fourier_noisy_noise_removed = 20 * np.log(np.abs(center_shift))

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(2, 3, 1)
ax1.title.set_text("Original Image")
ax1.imshow(img, cmap='gray')
ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(fourier_noisy, cmap='gray')
ax2.title.set_text("Fourier Transform")
ax3 = fig.add_subplot(2, 3, 3)
ax3.imshow(fourier_noisy_noise_removed, cmap='gray')
ax3.title.set_text("Fourier Transform with mask")
ax4 = fig.add_subplot(2, 3, 4)
ax4.imshow(denoised_image, cmap='gray')
ax4.title.set_text("Denoised and unfiltered image")
ax5 = fig.add_subplot(2, 3, 6)
ax5.imshow(denoised_image_blpf, cmap='gray')
ax5.title.set_text("Denoised and filtered image")

plt.show()
