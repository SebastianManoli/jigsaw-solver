import cv2
import numpy as np
def transform_colorspace(image):
    # convert to YUV
    YUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # split channels
    _, U, _ = cv2.split(YUV)

    return U

def transform_blur_and_thresh(U):
    # Apply blurring to U channel and thresholding [1], [2] 
    blur = cv2.GaussianBlur(U, (25, 25), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def segment_contours_cropping(mask, image):
    # extract the contours
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    #create bounding rectangle
    shark = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(shark)

    # Loop through contours
    for c in contours:
        # bound the contour with a rectangle
        x, y, w, h = cv2.boundingRect(c)
        # find the area of the contour
        A = cv2.contourArea(c)
        # cropping the image and mask 
        cropped_img = image[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
        
        return cropped_img, cropped_mask

def transform_convert_clean(cropped_img, cropped_mask):
    # convert to YUV
    cropped_img_YUV = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2YUV)
    # split channels
    Y2, U2, V2 = cv2.split(cropped_img_YUV)
    # equalise the histogram
    Y_new = cv2.equalizeHist(Y2)
    # merge new y channel and convert back
    YUV_new = cv2.merge([Y_new, U2, V2])
    I_new = cv2.cvtColor(YUV_new, cv2.COLOR_YUV2BGR)
    # Apply kernel for enhancement
    k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=float)
    I_kernel = cv2.filter2D(I_new, ddepth=-1, kernel=k)
    # extract the ROI, sets bg to black
    shark_only= cv2.bitwise_and(I_kernel, I_kernel, mask=cropped_mask)
    
    return shark_only
    