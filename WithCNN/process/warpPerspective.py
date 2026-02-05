import cv2
import numpy as np
import matplotlib.pyplot as plt
from process.getMasks import getMasks 
from globals import WARP_HEIGHT, WARP_WIDTH


def order_points(pts):

	soma = pts.sum(axis=1)		 # Obter array auxiliar [x1 + y1, x2 + y2, x3 + y3, x4 + y4] 
	diff = np.diff(pts, axis=1)  # array auxiliar: [y1 - x1, y2 - x2, y3 - x3, y4 - x4]

	tl = pts[np.argmin(soma)] # A soma minima vai sempre ser Top-Left 
	br = pts[np.argmax(soma)] # A soma maxima vai sempre ser Bottom-Right
	tr = pts[np.argmin(diff)] # A diferença mínima vai ser sempre Top-Right
	bl = pts[np.argmax(diff)] # A diferença máxima vai ser sempre Bottom-Left

	return np.array([tl, tr, bl, br], dtype="float32")

def get_points(original_img):

	"""
	# Mock Values para teste
	pontos_origem = np.float32([
		[1488, 787],  # top left
		[2871, 792], # top right
		[1221, 1852],  # bottom left
		[3191, 1884]  # bottom right
	])
	"""
	pontos_origem = np.zeros((4, 2), dtype="float32")

	pontos_destino = np.float32([
		[0, 0], # Top-left
		[WARP_WIDTH, 0], # Top-right	
		[0, WARP_HEIGHT], # Bottom-left
		[WARP_WIDTH, WARP_HEIGHT] # Bottom-right
	])

	# Let's now try to get the real thing.
	# We got the real thing. Yupi

	_, _, joined_mask = getMasks(original_img)
	joined_mask = cv2.bitwise_not(joined_mask)
	joined_mask = cv2.dilate(joined_mask, np.ones((5,5), np.uint8), iterations=1)


	contours, _ = cv2.findContours(joined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	square_contour = None
	max_area = 0

	for cnt in contours:
		area = cv2.contourArea(cnt)
		
		if area < 50000:  # ignore noise
			continue

		peri = cv2.arcLength(cnt, True) # Obter perimetro da area, True para para pegar só se for curva fechada
		approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) # Obter o numero de pontos, parametro do meio = This is the maximum distance between the original curve and its approximation
		# Ou seja, distância de "folga". True for closed curve as usual

		if len(approx) == 4 and area > max_area: # Se é quadrado AND é a maior àrea encontrada até agora atualiza
			square_contour = approx
			max_area = area

	# Obter os cantos do quadrado
	pontos_origem = square_contour.reshape(4, 2)
	
	# Ordenar os pontos
	pontos_origem = order_points(pontos_origem)

	return pontos_origem, pontos_destino

def warpPerspective(original_img):
	
	new_img = np.zeros((WARP_WIDTH, WARP_HEIGHT, 3), dtype='uint8')

	# Obter warping points
	pontos_origem, pontos_destino = get_points(original_img)
	
	# Obter a matriz de transformacao
	Transformation_Matriz = cv2.getPerspectiveTransform(pontos_origem, pontos_destino)

	# Aplicar a transformacao de perspectiva na imagem original
	new_img = cv2.warpPerspective(original_img, Transformation_Matriz, (700, 700))


	return new_img