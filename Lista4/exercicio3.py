import cv2
import numpy as np
import os

print("\n--- Executando Exercício 3: Detecção de Linhas ---")
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'linhas.png')

try:
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        raise FileNotFoundError("Imagem 'linhas.png' não encontrada.")

    # Kernels
    horizontal_kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
    vertical_kernel = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
    plus_45_kernel = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])
    minus_45_kernel = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
    
    # Aplicações e limiarização
    h_lines = cv2.threshold(cv2.filter2D(gray_img, -1, horizontal_kernel), 150, 255, cv2.THRESH_BINARY)[1]
    v_lines = cv2.threshold(cv2.filter2D(gray_img, -1, vertical_kernel), 150, 255, cv2.THRESH_BINARY)[1]
    p45_lines = cv2.threshold(cv2.filter2D(gray_img, -1, plus_45_kernel), 150, 255, cv2.THRESH_BINARY)[1]
    m45_lines = cv2.threshold(cv2.filter2D(gray_img, -1, minus_45_kernel), 150, 255, cv2.THRESH_BINARY)[1]
    
    # Combinação com OR
    combined_img = cv2.bitwise_or(h_lines, v_lines)
    combined_img = cv2.bitwise_or(combined_img, p45_lines)
    combined_img = cv2.bitwise_or(combined_img, m45_lines)
    cv2.imwrite(os.path.join(script_dir, 'linhas_resultado_final.png'), combined_img)

    # Exibe o resultado final em uma janela
    cv2.imshow("Ex3: Resultado Final - Todas as Linhas", combined_img)
    print("Pressione qualquer tecla para fechar a janela do Ex. 3.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except FileNotFoundError as e:
    print(e)