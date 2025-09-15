import cv2
import os

print("\n--- Executando Exercício 4: Detector de Canny ---")
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'igreja.png')

try:
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        raise FileNotFoundError("Imagem 'igreja.png' não encontrada.")

    edges = cv2.Canny(gray_img, 100, 200)
    cv2.imwrite(os.path.join(script_dir, 'igreja_canny_edges.png'), edges)

    # Exibe o resultado final em uma janela
    cv2.imshow("Ex4: Detector de Bordas Canny", edges)
    print("Pressione qualquer tecla para fechar a janela do Ex. 4.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except FileNotFoundError as e:
    print(e)