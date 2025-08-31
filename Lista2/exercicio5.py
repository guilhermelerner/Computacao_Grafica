import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

def detector_bordas_roberts(imagem):

    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.int16)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.int16)
    
    roberts_x = cv2.filter2D(imagem, cv2.CV_16S, kernel_x)
    roberts_y = cv2.filter2D(imagem, cv2.CV_16S, kernel_y)
    
    abs_roberts_x = cv2.convertScaleAbs(roberts_x)
    abs_roberts_y = cv2.convertScaleAbs(roberts_y)
    
    roberts_combinado = cv2.addWeighted(abs_roberts_x, 0.5, abs_roberts_y, 0.5, 0)
    return roberts_combinado

if __name__ == '__main__':
    img_lena = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    img_minha = cv2.imread('minhaimagem.webp', cv2.IMREAD_GRAYSCALE)
    
    if img_lena is None or img_minha is None:
        print("Erro: Não foi possível carregar uma ou ambas as imagens.")
        sys.exit()

    # Aplica o detector
    lena_roberts = detector_bordas_roberts(img_lena)
    minha_roberts = detector_bordas_roberts(img_minha)

    # Exibe os resultados
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(img_lena, cmap='gray')
    plt.title('Original (lena.png)')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(lena_roberts, cmap='gray')
    plt.title('Detector de Roberts')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(img_minha, cmap='gray')
    plt.title('Original (minhaimagem.webp)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(minha_roberts, cmap='gray')
    plt.title('Detector de Roberts')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()