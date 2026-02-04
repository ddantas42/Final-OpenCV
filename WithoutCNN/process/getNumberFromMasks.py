import cv2
from globals import THRESHOLD
import numpy as np

def getNumberFromMasks(joined_mask):

	number = 0
	matches = np.zeros(15)

	for i in range(1, 16):
		# Get template from masks/
		template = cv2.imread(f"masks/mask{i}.png", cv2.IMREAD_GRAYSCALE)
		
		# Get the result of how much the template matches the joined_mask and get the max value (value that indicates the best match)
		result = cv2.matchTemplate(joined_mask, template, cv2.TM_CCOEFF_NORMED)
		_, max_val, _, _ = cv2.minMaxLoc(result)
		
		if (max_val > THRESHOLD):
			print(f"\tTemplate {i}: {float(max_val)}")
			matches[i - 1] = float(max_val)

	if np.all(matches == 0): # Nenhuma mask acima do threshold, logo fica -1
		print("No matches found above threshold.")
		return -1

	biggest_index = 0

	if (np.all(matches[9:16] == 0)): # Se todas as matches de numeros 10 a 15 forem 0 (under do threhold)
		biggest_index = np.argmax(matches) # Pega o maior número (most likely o número correto)
		print(f"Number found (0-9): {biggest_index + 1}")
		return biggest_index + 1
		
	# Se apatir daqui o código está a correr, significa que há pelo menos uma match de numero 10 a 15

	# Ver se apenas existe UMA match de numeros 10 a 15 
	count = 0
	for i in range(9, 15):
		if (matches[i] > 0):
			count += 1

	if (count == 1): # Se houver apenas uma match, assumimos que é o numero correto
		biggest_index = np.argmax(matches[9:16]) + 9 # Procura o maior index entre 9 e 15 (números 10 a 15)
		print(f"Only one match for 10-15: {biggest_index + 1}")
		return biggest_index + 1

	
	# pegamos os mais 2 numeros de 0 a 9 e juntamos para criar o numero final de 10 a 15
	biggest_index = np.argmax(matches[1:9]) + 1 # numero maior de 2 a 9 visto que 1 é de certeza (+ 1 porque estamos a procurar do index 1 e não do 0)

	numero_final = 10 + biggest_index + 1 # +10 porque estamos a somar 10 ao numero do index e + 1 por ser um index no array
	
	if (matches[numero_final - 1] == 0):
		print(f"Combined number {numero_final} not found in masks, returning max of 10-15")
		return np.argmax(matches[9:16]) + 10 # Se o numero combinado nao der match em nenhum, retornamos o maior numero entre 10 a 15

	print(f"Combined number found (10-15): {numero_final}")
	return numero_final
