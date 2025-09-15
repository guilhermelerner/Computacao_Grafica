import cv2
import numpy as np
import os

# Pega o caminho do diretório onde o script está localizado
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'circuito.tif')

try:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"A imagem '{image_path}' não foi encontrada.")

    # 1. Converter para escala de cinza
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    median_1 = cv2.medianBlur(gray_img, 3)
    cv2.imwrite(os.path.join(script_dir, 'circuito_mediana_1.tif'), median_1)
    
    print("Mostrando resultado 1. Pressione qualquer tecla para continuar...")
    cv2.imshow("Resultado 1 - Mediana 1x", median_1)
    cv2.waitKey(0)

    median_2 = cv2.medianBlur(median_1, 3)
    cv2.imwrite(os.path.join(script_dir, 'circuito_mediana_2.tif'), median_2)

    print("Mostrando resultado 2. Pressione qualquer tecla para continuar...")
    cv2.imshow("Resultado 2 - Mediana 2x", median_2)
    cv2.waitKey(0)

    median_3 = cv2.medianBlur(median_2, 3)
    cv2.imwrite(os.path.join(script_dir, 'circuito_mediana_3.tif'), median_3)
    
    print("Mostrando resultado 3. Pressione qualquer tecla para fechar.")
    cv2.imshow("Resultado 3 - Mediana 3x", median_3)
    cv2.waitKey(0) 
    
    cv2.destroyAllWindows()
    print("\nProcesso concluído!")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"Ocorreu um erro: {e}")