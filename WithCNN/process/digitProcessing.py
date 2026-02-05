import cv2
import numpy as np
import torch
import scipy.ndimage as sc
from collections import Counter


def preprocess_image_for_cnn(img_segment):
	
	gray = img_segment.copy()
	
	# Aplicar erosão e dilatação para melhorar a imagem
	b_dilated = sc.binary_erosion(gray, structure=np.ones((3, 3)))
	b_eroded = sc.binary_dilation(b_dilated, structure=np.ones((4, 4)))
	binary = (b_eroded.astype(np.uint8) * 255)
	
	# Morfologia para melhorar os dígitos
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	binary = cv2.erode(binary, kernel, iterations=1)
	binary = cv2.dilate(binary, kernel, iterations=1)
	binary = cv2.erode(binary, kernel, iterations=3)
	binary = cv2.dilate(binary, kernel, iterations=2)
	
	# Redimensionar para 28x28 mantendo aspect ratio
	h, w = binary.shape
	if h > w:
		new_h = 20
		new_w = max(1, int(w * 20 / h))
	else:
		new_w = 20
		new_h = max(1, int(h * 20 / w))
	
	resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
	
	# Centralizar em canvas 28x28
	canvas = np.zeros((28, 28), dtype=np.uint8)
	y_offset = (28 - new_h) // 2
	x_offset = (28 - new_w) // 2
	canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
	
	# Normalizar e converter para tensor PyTorch
	normalized = canvas.astype(np.float32) / 255.0
	tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
	
	return tensor, canvas


def extract_digits_from_segment(gray):
	"""Extrai os dígitos individuais de um segmento"""
	h, w = gray.shape
	
	# Recortar bordas (2%)
	margin_h = int(h * 0.02)
	margin_w = int(w * 0.02)
	binary = gray[margin_h:h-margin_h, margin_w:w-margin_w]
	
	# Verificar cor do fundo usando os cantos
	
	
	# Encontrar contornos dos dígitos
	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	if len(contours) == 0:
		return [binary], binary	
	# Filtrar contornos por área e dimensões
	img_area = binary.shape[0] * binary.shape[1]
	valid_contours = []
	for cnt in contours:
		x, y, cw, ch = cv2.boundingRect(cnt)
		area = cv2.contourArea(cnt)
		if area > 100 and cw > 5 and ch > 10 and area < img_area * 0.9:
			valid_contours.append((x, y, cw, ch, area))
	
	if len(valid_contours) == 0:
		return [binary], binary
	
	# Verificar sobreposição horizontal (números como 0, 8)
	if len(valid_contours) >= 2:
		sorted_by_x = sorted(valid_contours, key=lambda c: c[0])
		x1, _, w1, _, _ = sorted_by_x[0]
		x2, _, w2, _, _ = sorted_by_x[1]
		
		overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
		
		if overlap > min(w1, w2) * 0.5:
			return [binary], binary
	
	# Ordenar da esquerda para direita e extrair ROIs
	valid_contours = sorted(valid_contours, key=lambda c: c[0])
	
	digit_rois = []
	for (x, y, cw, ch, _) in valid_contours[:2]:
		padding = 3
		x1 = max(0, x - padding)
		y1 = max(0, y - padding)
		x2 = min(binary.shape[1], x + cw + padding)
		y2 = min(binary.shape[0], y + ch + padding)
		digit_rois.append(binary[y1:y2, x1:x2])
	
	return digit_rois, binary


def classify_segment(gray, model, device):
	"""Classifica um segmento e retorna o número detectado (0-15)"""
	digit_rois, binary_cropped = extract_digits_from_segment(gray)
	
	digits, digit_images, confidences, raw_digit_images = [], [], [], []
	
	for roi in digit_rois:
		raw_digit_images.append(roi.copy())  # Guardar imagem antes do preprocessing
		tensor, processed = preprocess_image_for_cnn(roi)
		tensor = tensor.to(device)
		digit_images.append(processed)
		
		with torch.no_grad():
			output = model(tensor)
			probs = torch.nn.functional.softmax(output, dim=1)
			confidence, pred = probs.max(1)
			digits.append(pred.item())
			confidences.append(confidence.item())
	
	# Filtrar dígitos com baixa confiança
	filtered_digits = [d for d, c in zip(digits, confidences) if c >= 0.5]
	
	# Calcular número final
	if len(filtered_digits) == 0:
		number = 0
	elif len(filtered_digits) == 1:
		number = filtered_digits[0]
	else:
		number = filtered_digits[0] * 10 + filtered_digits[1]
	
	# Validar range 0-15
	if number > 15:
		best_idx = confidences.index(max(confidences))
		number = digits[best_idx]
	
	return number, binary_cropped, digit_images, digits, confidences, raw_digit_images


def resolve_duplicates(predictions, confidences_list):

	predictions = predictions.copy()
	
	# Verificar se já existe 0 na matriz
	if 0 in predictions:
		return predictions
	
	# Encontrar duplicatas
	counts = Counter(predictions)
	duplicates = [num for num, count in counts.items() if count > 1 and num != 0]
	
	if not duplicates:
		return predictions
	
	# Para cada número duplicado, encontrar qual tem menor confiança
	for duplicate_num in duplicates:
		# Encontrar índices onde este número aparece
		indices = [i for i, pred in enumerate(predictions) if pred == duplicate_num]
		
		# Calcular confiança média para cada ocorrência
		avg_confidences = []
		for idx in indices:
			digits, confs = confidences_list[idx]
			if duplicate_num in digits:
				digit_idx = digits.index(duplicate_num)
				avg_confidences.append(confs[digit_idx] if digit_idx < len(confs) else 0)
			else:
				# Se for número de 2 dígitos, usar média das confianças
				avg_confidences.append(sum(confs) / len(confs) if confs else 0)
		
		# Encontrar o índice com menor confiança
		min_conf_idx = indices[avg_confidences.index(min(avg_confidences))]
		
		# Substituir por 0
		predictions[min_conf_idx] = 0
		print(f"Duplicado detectado: número {duplicate_num}")
		print(f"Substituído por 0 no segmento {min_conf_idx+1} (confiança: {min(avg_confidences):.1%})")
	
	return predictions
