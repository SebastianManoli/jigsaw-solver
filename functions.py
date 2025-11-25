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


def piece_boundary(cropped_mask):

    # Find contours on the new mask (copped)
    contours, _ = cv2.findContours(cropped_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None
    
    piece_contour = max(contours, key=cv2.contourArea)
    
    return piece_contour

# def detect_corners(contour):

#     hull = cv2.convexHull(contour)
    
#     perimeter = cv2.arcLength(hull, closed=True)
#     epsilon = 0.05 * perimeter  # Start with 5%, adjust if needed
#     corners = cv2.approxPolyDP(hull, epsilon, closed=True)
    
#     if len(corners) != 4:
#         epsilon = 0.1 * perimeter
#         corners = cv2.approxPolyDP(hull, epsilon, closed=True)
    
#     return corners

def detect_corners(contour):
    x, y, w, h = cv2.boundingRect(contour)
    
    corners_rect = np.array([
        [x, y],           # top-left
        [x + w, y],       # top-right  
        [x + w, y + h],   # bottom-right
        [x, y + h]        # bottom-left
    ])
    
    contour_reshaped = contour.reshape(-1, 2)
    refined_corners = []
    
    # For each bounding box corner, look INWARD for the real corner
    inward_offsets = [
        (20, 20),    # top-left: look down and right
        (-20, 20),   # top-right: look down and left
        (-20, -20),  # bottom-right: look up and left
        (20, -20)    # bottom-left: look up and right
    ]
    
    for corner, offset in zip(corners_rect, inward_offsets):
        # Search region is slightly INWARD from bounding box
        search_center = corner + np.array(offset)
        
        distances = np.linalg.norm(contour_reshaped - search_center, axis=1)
        nearby_indices = np.where(distances < 60)[0]
        
        if len(nearby_indices) < 3:
            closest_idx = np.argmin(distances)
            refined_corners.append(contour_reshaped[closest_idx])
            continue
        
        best_idx = nearby_indices[0]
        best_score = float('inf')
        contour_len = len(contour_reshaped)
        
        for idx in nearby_indices:
            p1 = contour_reshaped[(idx - 15) % contour_len]
            p2 = contour_reshaped[idx]
            p3 = contour_reshaped[(idx + 15) % contour_len]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm_product > 0:
                cos_angle = np.dot(v1, v2) / norm_product
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                
                angle_diff = abs(angle - 90)
                distance = np.linalg.norm(contour_reshaped[idx] - search_center)
                
                # Heavily weight the angle (tabs have sharper angles than 90)
                score = (angle_diff * 3) + (distance / 5)
                
                if score < best_score:
                    best_score = score
                    best_idx = idx
        
        refined_corners.append(contour_reshaped[best_idx])
    
    return np.array(refined_corners, dtype=np.int32).reshape(-1, 1, 2)

def segment_sides(contour, corners):
    contour_reshaped = contour.reshape(-1, 2)
    
    corner_indices = []
    for corner in corners.reshape(-1, 2):
        idx = np.where((contour_reshaped == corner).all(axis=1))[0][0]
        corner_indices.append(idx)
    
    corner_indices = sorted(corner_indices)
    
    sides = []
    for i in range(4):
        start_idx = corner_indices[i]
        end_idx = corner_indices[(i + 1) % 4]
        
        if end_idx > start_idx:
            side = contour_reshaped[start_idx:end_idx+1]
        else:
            side = np.vstack([contour_reshaped[start_idx:], contour_reshaped[:end_idx+1]])
        
        sides.append(side)
    
    return sides

def classify_side(side_contour):

    start_point = side_contour[0]
    end_point = side_contour[-1]
    
    straight_distance = np.linalg.norm(end_point - start_point)
    
    contour_length = cv2.arcLength(side_contour.reshape(-1, 1, 2), False)
    
    curve_ratio = contour_length / straight_distance
    
    if curve_ratio < 1.1: 
        return "FLAT"
    else:

        line_vec = end_point - start_point
        
        mid_idx = len(side_contour) // 2
        mid_point = side_contour[mid_idx]
        
        cross = np.cross(line_vec, mid_point - start_point)
        
        if cross > 0:
            return "TAB"
        else:
            return "SLOT"
        
def classify_piece(side_types):

    flat_count = side_types.count('FLAT')
    
    if flat_count == 2:
        return "CORNER"
    elif flat_count == 1:
        return "EDGE"
    elif flat_count == 0:
        return "MIDDLE"
    else:
        return "UNKNOWN"
    


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


def get_normalized_contour(contour):
    # 1. Get endpoints
    p1 = contour[0]
    p2 = contour[-1]
    
    # Handle case where contour might be (N, 1, 2) or (N, 2)
    if len(p1.shape) > 1: # (1, 2)
        p1 = p1[0]
        p2 = p2[0]
    elif len(p1.shape) == 1:
        pass # p1 is (2,)
    
    # 2. Calculate angle to rotate p1-p2 to horizontal
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    # 3. Rotate all points
    # We want to align the line p1-p2 with the x-axis.
    # In image coords (y-down), positive angle is CW (down).
    # So if line is down (+angle), we need to rotate CCW (up, -angle).
    # However, cv2.getRotationMatrix2D with positive angle rotates CCW (visually Up).
    # So to correct a +45 deg (Down-Right) line to 0 deg (Right), we rotate CCW (Up) by 45 deg.
    rotation_matrix = cv2.getRotationMatrix2D(tuple(p1.astype(float)), angle_deg, 1.0)
    
    # Ensure contour is (N, 1, 2) for cv2.transform
    pts = contour.reshape(-1, 1, 2).astype(float)
    rotated_pts = cv2.transform(pts, rotation_matrix)
    
    # 4. Translate so p1 is at (0,0)
    t_x, t_y = rotated_pts[0][0]
    normalized_pts = rotated_pts - [t_x, t_y]
    
    return normalized_pts.astype(np.float32)



def calculate_match_score(side_A, side_B, type_A, type_B):
    # 1. Compatibility Check
    if type_A == "FLAT" or type_B == "FLAT":
        return float('inf')
    if type_A == type_B: # TAB-TAB or SLOT-SLOT don't match
        return float('inf')
    
    # 2. Normalize
    norm_A = get_normalized_contour(side_A)
    norm_B = get_normalized_contour(side_B)
    
    # 3. Invert Side B (Flip over X-axis)
    # We want to see if A fits into B.
    # If A is TAB (bump up) and B is SLOT (dip down),
    # Normalization makes B a dip down.
    # Inverting B makes it a bump up.
    # Then we compare bump A with bump B.
    
    inverted_B = norm_B.copy()
    inverted_B[:, 0, 1] *= -1
    
    # 4. Calculate Match Score (Hu Moments)
    # Method=1 (CV_CONTOURS_MATCH_I1) is often good. Lower is better.
    try:
        score = cv2.matchShapes(norm_A, inverted_B, cv2.CONTOURS_MATCH_I1, 0.0)
    except cv2.error:
        return float('inf') # Handle potential errors with empty contours
        
    return score