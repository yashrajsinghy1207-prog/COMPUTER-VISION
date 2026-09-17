import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load Image
# -----------------------------
img = cv2.imread("input.jpg.jpeg")

if img is None:
    print("Image not found!")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------------------------------
# Display Function
# ---------------------------------

def show(title, image):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(image, cmap='gray' if len(image.shape)==2 else None)
    plt.title(title)
    plt.axis("off")
    plt.show()

#############################################################
# PART A : Spatial Domain Enhancement
#############################################################

# 1 Brightness Adjustment

brightness = 50

bright = cv2.convertScaleAbs(img, alpha=1, beta=brightness)

show("Brightness Adjustment", bright)


# 2 Contrast Adjustment

contrast = 1.5

contrast_img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)

show("Contrast Adjustment", contrast_img)


# 3 Image Negative

negative = 255 - img

show("Image Negative", negative)


# 4 Log Transformation

c = 255 / np.log(1 + np.max(img))

log_image = c * np.log(1 + img.astype(np.float32))

log_image = np.array(log_image, dtype=np.uint8)

show("Log Transformation", log_image)


# 5 Gamma Transformation

gamma = 2.0

gamma_corrected = np.array(255*(img/255)**gamma,dtype='uint8')

show("Gamma Transformation", gamma_corrected)


# 6 Thresholding

gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

threshold_value = 127

_,thresh=cv2.threshold(gray,threshold_value,255,cv2.THRESH_BINARY)

show("Thresholding",thresh)


# 7 Contrast Stretching

min_val=np.min(img)
max_val=np.max(img)

stretch=((img-min_val)/(max_val-min_val)*255).astype(np.uint8)

show("Contrast Stretching",stretch)

#############################################################
# PART B : Geometrical Transformations
#############################################################

rows,cols=img.shape[:2]

# 1 Translation

tx=100
ty=50

M=np.float32([[1,0,tx],[0,1,ty]])

translated=cv2.warpAffine(img,M,(cols,rows))

show("Translation",translated)


# 2 Scaling

scale=1.5

scaled=cv2.resize(img,None,fx=scale,fy=scale)

show("Scaling",scaled)


# 3 Rotation

angle=45

M=cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)

rotated=cv2.warpAffine(img,M,(cols,rows))

show("Rotation",rotated)


# 4 Horizontal Reflection

horizontal=cv2.flip(img,1)

show("Horizontal Reflection",horizontal)


# 5 Vertical Reflection

vertical=cv2.flip(img,0)

show("Vertical Reflection",vertical)


# 6 Reflection about Origin

origin=cv2.flip(img,-1)

show("Reflection about Origin",origin)


# 7 Shearing X-axis

shear_x=0.5

M=np.float32([[1,shear_x,0],
              [0,1,0]])

sheared_x=cv2.warpAffine(img,M,(int(cols+rows*shear_x),rows))

show("Shearing X-axis",sheared_x)


# 8 Shearing Y-axis

shear_y=0.5

M=np.float32([[1,0,0],
              [shear_y,1,0]])

sheared_y=cv2.warpAffine(img,M,(cols,int(rows+cols*shear_y)))

show("Shearing Y-axis",sheared_y)


# 9 Affine Transformation

pts1=np.float32([[50,50],[200,50],[50,200]])

pts2=np.float32([[10,100],[200,50],[100,250]])

M=cv2.getAffineTransform(pts1,pts2)

affine=cv2.warpAffine(img,M,(cols,rows))

show("Affine Transformation",affine)