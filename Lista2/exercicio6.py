import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

def detector_bordas_prewitt(imagem):
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.int16)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.int16)

    prewitt_x = cv2.filter2D(imagem, cv2.CV_16S, kernel_x)
    prewitt_y = cv2.filter2D(imagem, cv2.CV_16S, kernel_y)
    
    abs_prewitt_x = cv2.convertScaleAbs(prewitt_x)
    abs_prewitt_y = cv2.convertScaleAbs(prewitt_y)
    
    prewitt_combinado = cv2.addWeighted(abs_prewitt_x, 0.5, abs_prewitt_y, 0.5, 0)
    return prewitt_combinado

if __name__ == '__main__':
    img_lena = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    img_minha = cv2.imread('minhaimagem.webp', cv2.IMREAD_GRAYSCALE)

    if img_lena is None or img_minha is None:
        print("Erro: Não foi possível carregar uma ou ambas as imagens.")
        sys.exit()

    # Aplica o detector
    lena_prewitt = detector_bordas_prewitt(img_lena)
    minha_prewitt = detector_bordas_prewitt(img_minha)

    # Exibe os resultados
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(img_lena, cmap='gray')
    plt.title('Original (lena.png)')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(lena_prewitt, cmap='gray')
    plt.title('Detector de Prewitt')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(img_minha, cmap='gray')
    plt.title('Original (minhaimagem.webp)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(minha_prewitt, cmap='gray')
    plt.title('Detector de Prewitt')
    plt.axis('off')

    plt.tight_layout()
    plt.show()