import cv2
import numpy as np


def getMasks(warpedImg):
	
	height = warpedImg.shape[0]
	width = warpedImg.shape[1]

	copyWarped = warpedImg.copy()
	
	mask_r1 = mask_r2 = mask_red = mask_white = joined_mask = np.zeros((height, width, 3), dtype=np.uint8)
	
	hsv = cv2.cvtColor(copyWarped, cv2.COLOR_BGR2HSV)

	mask_r1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
	mask_r2 = cv2.inRange(hsv, (170, 80,60), (180, 255, 255))
	mask_red = cv2.bitwise_or(mask_r1, mask_r2)

	mask_white = cv2.inRange(hsv, (0, 0, 150), (180, 30, 255))
	joined_mask = cv2.bitwise_or(mask_red, mask_white)
	joined_mask = cv2.bitwise_not(joined_mask)

	return mask_red, mask_white, joined_mask
