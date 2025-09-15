import cv2
import numpy as np
import os

# Função do exercício anterior
def otsu_threshold(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    total_pixels = image.shape[0] * image.shape[1]
    prob_hist = histogram.ravel() / total_pixels
    max_variance = 0
    best_threshold = 0
    for t in range(256):
        w0 = np.sum(prob_hist[:t])
        w1 = np.sum(prob_hist[t:])
        if w0 == 0 or w1 == 0: continue
        mean0 = np.sum(np.arange(t) * prob_hist[:t]) / w0
        mean1 = np.sum(np.arange(t, 256) * prob_hist[t:]) / w1
        variance = w0 * w1 * ((mean0 - mean1) ** 2)
        if variance > max_variance:
            max_variance = variance
            best_threshold = t
    _ , thresholded_image = cv2.threshold(image, best_threshold, 255, cv2.THRESH_BINARY)
    return thresholded_image, best_threshold

print("\n--- Executando Exercício 6: Metodo de Otsu ---")
script_dir = os.path.dirname(os.path.abspath(__file__))
image_files = ['harewood.jpg', 'nuts.jpg', 'snow.jpg', 'minhaimagem.webp']

for filename in image_files:
    try:
        image_path = os.path.join(script_dir, filename)
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray_img is None:
            raise FileNotFoundError(f"Imagem '{filename}' não encontrada.")
        
        otsu_img, threshold_value = otsu_threshold(gray_img)
        output_filename = f'{filename.split(".")[0]}_otsu.png'
        cv2.imwrite(os.path.join(script_dir, output_filename), otsu_img)
        
        # Exibe o resultado de cada imagem em uma janela
        print(f"Mostrando resultado para '{filename}'. Limiar de Otsu = {threshold_value}.")
        cv2.imshow(f"Ex6: Otsu em {filename}", otsu_img)
        print("Pressione qualquer tecla para a proxima imagem...")
        cv2.waitKey(0)

    except FileNotFoundError as e:
        print(e)

# Fecha a última janela aberta
cv2.destroyAllWindows()
print("\nTodos os exercícios foram concluídos!")