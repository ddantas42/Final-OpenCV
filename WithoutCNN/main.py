from process.segmentPerspective import segmentPerspective
from process.warpPerspective import warpPerspective
from process.getMasks import getMasks
from process.getNumberFromMasks import getNumberFromMasks
from globals import OUTPUT_FOLDER, DEBUG
import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_rgb(image, title=""):
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Para converter de BGR (cv2 working thing) para RGB (matplotlib working thing)
	cv2.imwrite(f"{OUTPUT_FOLDER}/{title.replace(' ', '_')}.png", image)
	plt.title(title)

def show_imgs(imgs_dict):
	num_imgs = len(imgs_dict)
	for i, (title, img) in enumerate(imgs_dict.items()):
		plt.subplot(1, num_imgs, i + 1)
		show_rgb(img, title)

	if DEBUG:
		plt.show()
	

# Função para mostrar imagems de matrizes N por N
def show_matrix_img(matrix, n, title=""):
	plt.figure(figsize=(8, 8))
	for i in range(n):
		for j in range(n):
			plt.subplot(n, n, i * n + j + 1)
			show_rgb(matrix[i, j], title=f'[{i},{j}]')
			plt.axis('off')

	plt.suptitle(title)

	if DEBUG:
		plt.show()


def	main():

	# Where it all begins...
	img = cv2.imread('perspective.jpg')
	linhas, colunas, canais = img.shape
	print(f'Linhas: {linhas}, Colunas: {colunas}, Canais: {canais}')

	# Warp Image 
	warpedImg, img_with_points = warpPerspective(img)

	show_imgs({
		"Original Image": img,
		"Warp Points": img_with_points,
		"Warped Perspective": warpedImg
	})


	# Split the Perspective Warped Image into 4x4 segments
	Segmented_Matrix = segmentPerspective(warpedImg)
	show_matrix_img(Segmented_Matrix, 4, "segmented Warped Image")

	final_array = np.zeros((4, 4), dtype='int8')

	for i in range(4):
		for j in range(4):
			segment = Segmented_Matrix[i, j]
			_, _, joined_mask = getMasks(segment)

			print(f"{i*4 + j + 1}:")
			# Transform each segment into a matrix and put it into final_array
			final_array[i, j] = getNumberFromMasks(joined_mask)
	
	print("Final Array:")
	print(final_array)


if __name__ == "__main__":
	main() 