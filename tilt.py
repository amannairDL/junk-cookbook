import cv2
import numpy as np
import matplotlib.pyplot as plt

debug = True

#Display image
def display(img, frameName="OpenCV Image"):
    if not debug:
        return
    plt.imshow(img)
    plt.show()

#rotate the image with given theta value
def rotate(img, theta):
    rows, cols = img.shape[0], img.shape[1]
    image_center = (cols/2, rows/2)

    M = cv2.getRotationMatrix2D(image_center,theta,1)

    abs_cos = abs(M[0,0])
    abs_sin = abs(M[0,1])

    bound_w = int(rows * abs_sin + cols * abs_cos)
    bound_h = int(rows * abs_cos + cols * abs_sin)

    M[0, 2] += bound_w/2 - image_center[0]
    M[1, 2] += bound_h/2 - image_center[1]

    # rotate orignal image to show transformation
    rotated = cv2.warpAffine(img,M,(bound_w,bound_h),borderValue=(255,255,255))
    return rotated


def slope(x1, y1, x2, y2):
    if x1 == x2:
        return 0
    slope = (y2-y1)/(x2-x1)
    theta = np.rad2deg(np.arctan(slope))
    return theta


def main(filePath):
    img = cv2.imread(filePath)
    textImg = img.copy()

    small = cv2.cvtColor(textImg, cv2.COLOR_BGR2GRAY)

    #find the gradient map
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    display(grad)

    #Binarize the gradient image
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    display(bw)

    #connect horizontally oriented regions
    #kernal value (9,1) can be changed to improved the text detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    display(connected)

    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)
    display(mask)
    #cumulative theta value
    cummTheta = 0
    #number of detected text regions
    ct = 0
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        #fill the contour
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        #display(mask)
        #ratio of non-zero pixels in the filled region
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        #assume at least 45% of the area is filled if it contains text
        if r > 0.45 and w > 8 and h > 8:
            #cv2.rectangle(textImg, (x1, y), (x+w-1, y+h-1), (0, 255, 0), 2)

            rect = cv2.minAreaRect(contours[idx])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(textImg,[box],0,(0,0,255),2)

            #we can filter theta as outlier based on other theta values
            #this will help in excluding the rare text region with different orientation from ususla value 
            theta = slope(box[0][0], box[0][1], box[1][0], box[1][1])
            cummTheta += theta
            ct +=1 
            #print("Theta", theta)

    #find the average of all cumulative theta value
    orientation = cummTheta/ct
    print("Image orientation in degress: ", orientation)
    finalImage = rotate(img, orientation)
#    display(textImg, "Detectd Text minimum bounding box")
    display(finalImage)

if __name__ == "__main__":


    file_path = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\total_180.jpg"
    file_path1 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\clock_90.jpg"
    file_path2 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\anti_90.jpg"
    file_path3 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\clock_05.png"
    file_path4 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\anti_05.png"
    main(file_path1)
    
    
    
    
import cv2
import numpy
from matplotlib import pyplot
import collections

QUANT_STEPS = 360*4

def quantized_angle(line, quant = QUANT_STEPS):
    theta = line[0][1]
    return numpy.round(theta / numpy.pi / 2 * QUANT_STEPS) / QUANT_STEPS * 360 % 90

def detect_rotation(monochromatic_img):
    # edges = cv2.Canny(monochromatic_img, 50, 150, apertureSize = 3) #play with these parameters
    lines = cv2.HoughLines(monochromatic_img, #input
                           1, # rho resolution [px]
                           numpy.pi/180, # angular resolution [radian]
                           200) # accumulator threshold – higher = fewer candidates
    counter = collections.Counter(quantized_angle(line) for line in lines)
    return counter


file_path = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\total_180.jpg"
file_path1 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\clock_90.jpg"
file_path2 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\anti_90.jpg"
file_path3 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\clock_05.png"
file_path4 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\anti_05.png"



img = cv2.imread(file_path1) #Image directly as grabbed from imgur.com
total_count = collections.Counter()
for channel in range(img.shape[-1]):
    total_count.update(detect_rotation(img[:,:,channel]))

most_common = total_count.most_common(5)

for angle,_ in most_common:
    pyplot.figure(figsize=(8,6), dpi=100)
    pyplot.title(f"{angle:.2f}°")
    rotation = cv2.getRotationMatrix2D((img.shape[0]/2, img.shape[1]/2), -angle, 1)
    pyplot.imshow(cv2.warpAffine(img, rotation, img.shape[:2]))
    
    
    
    
    
    
    
    
    
    
    
    
    





import cv2
import numpy as np
import matplotlib.pyplot as plt

def deskew(im, max_skew=10):
    height, width = im.shape[:2]

    # Create a grayscale image and denoise it
    im_gs = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gs = cv2.fastNlMeansDenoising(im_gs, h=3)

    # Create an inverted B&W copy using Otsu (automatic) thresholding
    im_bw = cv2.threshold(im_gs, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Detect lines in this image. Parameters here mostly arrived at by trial and error.
    lines = cv2.HoughLinesP(
        im_bw, 1, np.pi / 180, 200, minLineLength=width / 12, maxLineGap=width / 150
    )

    # Collect the angles of these lines (in radians)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angles.append(np.arctan2(y2 - y1, x2 - x1))

    # If the majority of our lines are vertical, this is probably a landscape image
    landscape = np.sum([abs(angle) > np.pi / 4 for angle in angles]) > len(angles) / 2

    # Filter the angles to remove outliers based on max_skew
    if landscape:
        angles = [
            angle
            for angle in angles
            if np.deg2rad(90 - max_skew) < abs(angle) < np.deg2rad(90 + max_skew)
        ]
    else:
        angles = [angle for angle in angles if abs(angle) < np.deg2rad(max_skew)]

    if len(angles) < 5:
        # Insufficient data to deskew
        return im

    # Average the angles to a degree offset
    angle_deg = np.rad2deg(np.median(angles))
    print(angle_deg)
    # If this is landscape image, rotate the entire canvas appropriately
    if landscape:
        if angle_deg < 0:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            angle_deg += 90
        elif angle_deg > 0:
            im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
            angle_deg -= 90

    # Rotate the image by the residual offset
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1)
    im = cv2.warpAffine(im, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return im



file_path = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\total_180.jpg"
file_path1 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\clock_90.jpg"
file_path2 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\anti_90.jpg"
file_path3 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\clock_05.png"
file_path4 = r"C:\Users\Aman.Sivaprasad\OneDrive - EY\Desktop\labcorp\results\testingdata\dob\orentation\anti_05.png"



image = cv2.imread(file_path2)

img =deskew(image)

plt.imshow(img)

















