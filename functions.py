import cv2
import numpy as np

def transform_colorspace(image):
    # convert to YUV
    YUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # split channels
    _, U, _ = cv2.split(YUV)

    return U

def transform_blur_and_thresh(U, use_morphology=True):
    # Apply blurring to U channel and thresholding [1], [2]
    blur = cv2.GaussianBlur(U, (25, 25), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if use_morphology:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)

        mask = cv2.dilate(mask, kernel, iterations=2)

    return mask

def segment_contours_cropping(mask, image, min_area_ratio=0.005):

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    pieces = []

    # Calculate minimum area threshold
    image_area = mask.shape[0] * mask.shape[1]
    min_area_threshold = image_area * min_area_ratio

    # Loop through contours
    for c in contours:
        # find the area of the contour
        A = cv2.contourArea(c)

        if A < min_area_threshold:
            continue

        # bound the contour with a rectangle
        x, y, w, h = cv2.boundingRect(c)

        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2*padding)
        h = min(image.shape[0] - y, h + 2*padding)

        # cropping the image and mask
        cropped_img = image[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]

        pieces.append((cropped_img, cropped_mask))

    print(f"  Found {len(pieces)} pieces after filtering (min_area: {min_area_threshold:.0f})")

    return pieces

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
    piece_only = cv2.bitwise_and(I_kernel, I_kernel, mask=cropped_mask)

    return piece_only
