import cv2
import numpy as np
import os

# Função do exercício anterior
def region_growing(image, seed, threshold=10):
    height, width = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    seed_value = int(image[seed[1], seed[0]])
    queue = [seed]
    segmented[seed[1], seed[0]] = 255
    while queue:
        x, y = queue.pop(0)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and segmented[ny, nx] == 0:
                    if abs(int(image[ny, nx]) - seed_value) < threshold:
                        segmented[ny, nx] = 255
                        queue.append((nx, ny))
    return segmented

print("\n--- Executando Exercício 5: Crescimento de Região ---")
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'root.jpg')

try:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Imagem 'root.jpg' não encontrada.")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # AJUSTE A SEMENTE AQUI (x, y)
    seed_point = (250, 250)
    
    region_mask = region_growing(gray_img, seed=seed_point, threshold=20)
    
    highlight_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    highlight_img[region_mask == 255] = [0, 0, 255] # Destaca em vermelho
    cv2.imwrite(os.path.join(script_dir, 'root_region_growing.jpg'), highlight_img)

    # Exibe o resultado final em uma janela
    cv2.imshow("Ex5: Resultado Crescimento de Regiao", highlight_img)
    print("Pressione qualquer tecla para fechar a janela do Ex. 5.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except FileNotFoundError as e:
    print(e)