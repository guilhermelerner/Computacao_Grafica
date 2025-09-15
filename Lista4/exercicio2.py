import cv2
import numpy as np
import os

print("\n--- Executando Exercício 2: Detecção de Pontos ---")
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'pontos.png')

try:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Imagem 'pontos.png' não encontrada.")

    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filtered_img = cv2.filter2D(img, -1, kernel)
    _ , result_img = cv2.threshold(filtered_img, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(script_dir, 'pontos_detectados.png'), result_img)

    # Exibe o resultado final em uma janela
    cv2.imshow("Ex2: Pontos Isolados Detectados", result_img)
    print("Pressione qualquer tecla para fechar a janela do Ex. 2.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except FileNotFoundError as e:
    print(e)