from process.segmentPerspective import segmentPerspective
from process.warpPerspective import warpPerspective
from process.digitProcessing import classify_segment, resolve_duplicates
from process.cnnModel import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def main():
	# Obter diretório do script
	script_dir = os.path.dirname(os.path.abspath(__file__))
	
	# Carregar imagem
	img_path = os.path.join(script_dir, 'perspective.jpg')
	img = cv2.imread(img_path)
	linhas, colunas, canais = img.shape
	print(f'Linhas: {linhas}, Colunas: {colunas}, Canais: {canais}')

	# Warp Image 
	warpedImg = warpPerspective(img)

	# Segmentar em 16 partes
	Segmented_Matrix = segmentPerspective(warpedImg)

	# Carregar o modelo CNN treinado
	model_path = os.path.join(script_dir, 'mnist_cnn_model.pth')
	if not os.path.exists(model_path):
		print("Modelo não encontrado. Execute simple_CNN.py primeiro para treinar o modelo.")
		return
	
	model, device = load_model(model_path)
	print("Modelo carregado com sucesso!")

	# Classificar cada segmento (detectando números de 0 a 15)
	predictions = []
	processed_images = []
	confidences_data = []
	
	for i in range(16):
		# Converter para grayscale se necessário
		if len(Segmented_Matrix[i].shape) == 3:
			gray = cv2.cvtColor(Segmented_Matrix[i], cv2.COLOR_BGR2GRAY)
		else:
			gray = Segmented_Matrix[i].copy()
		
		# Classificar segmento
		number, binary_img, digit_images, digits, confidences, raw_digit_images = classify_segment(gray, model, device)
		
		predictions.append(number)
		processed_images.append((binary_img, digit_images, digits, confidences, raw_digit_images))
		confidences_data.append((digits, confidences))
	
	# Resolver duplicatas (substituir o de menor confiança por 0 se não houver 0)
	predictions = resolve_duplicates(predictions, confidences_data)
	
	# Criar matriz 4x4
	puzzle_matrix = np.array(predictions).reshape(4, 4)
	
	print("\nMatriz 4x4 do puzzle:")
	print(puzzle_matrix)

	# Visualizar segmentos processados
	fig1, axes1 = plt.subplots(4, 4, figsize=(12, 12))
	fig1.suptitle('Segmentos Processados', fontsize=16)
	for i in range(16):
		row = i // 4
		col = i % 4
		binary_img, _, digits, confs, _ = processed_images[i]
		conf_str = [f"{c:.0%}" for c in confs] if confs else []
		axes1[row, col].imshow(binary_img, cmap='gray')
		axes1[row, col].set_title(f'Pred: {predictions[i]} | D: {digits} | C: {conf_str}', fontsize=8)
		axes1[row, col].axis('off')
	plt.tight_layout()
	
	# Visualizar dígitos extraídos ANTES de enviar ao CNN (sem predições)
	fig2 = plt.figure(figsize=(15, 12))
	fig2.suptitle('Dígitos Extraídos ANTES de Enviar ao CNN', fontsize=16)
	plot_idx = 1
	for i in range(16):
		_, _, _, _, raw_digit_images = processed_images[i]
		for j, raw_img in enumerate(raw_digit_images):
			if plot_idx > 64:
				break
			plt.subplot(8, 8, plot_idx)
			plt.imshow(raw_img, cmap='gray')
			plt.title(f'Segmento {i+1} - Dígito {j+1}', fontsize=8)
			plt.axis('off')
			plot_idx += 1
		if plot_idx > 64:
			break
	plt.tight_layout()
	
	# Visualizar resultados DEPOIS da avaliação do CNN
	fig3 = plt.figure(figsize=(15, 12))
	fig3.suptitle('Resultados DEPOIS da Avaliação pelo CNN (com predições e confiança)', fontsize=16)
	plot_idx = 1
	for i in range(16):
		_, digit_images, digits, confs, _ = processed_images[i]
		for j, (digit_img, digit_val) in enumerate(zip(digit_images, digits)):
			if plot_idx > 64:
				break
			plt.subplot(8, 8, plot_idx)
			plt.imshow(digit_img, cmap='gray')
			conf_pct = confs[j] * 100 if j < len(confs) else 0
			plt.title(f'S{i+1}: Pred={digit_val} ({conf_pct:.0f}%)', fontsize=7)
			plt.axis('off')
			plot_idx += 1
		if plot_idx > 64:
			break
	plt.tight_layout()
	
	plt.show()


if __name__ == "__main__":
	main() 