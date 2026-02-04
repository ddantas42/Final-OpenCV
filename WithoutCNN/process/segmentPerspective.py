import numpy as np

# Esta função vai segmentar a warped img e vai devolver uma matrix 4x4 com as imagens dos numeros
def segmentPerspective(warped_img):


	height, width, _ = warped_img.shape

	segment_height = height // 4
	segment_width = width // 4

	segmentedMatrix = np.zeros((4, 4, segment_height, segment_width, 3), dtype='uint8')

	for i in range(4):
		for j in range(4):
			y_start = i * segment_height
			y_end = (i + 1) * segment_height
			x_start = j * segment_width
			x_end = (j + 1) * segment_width

			segmentedMatrix[i, j] = warped_img[y_start:y_end, x_start:x_end]


	return segmentedMatrix