import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

def detector_bordas_sobel(imagem):
    sobel_x = cv2.Sobel(imagem, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(imagem, cv2.CV_16S, 0, 1, ksize=3)
    
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    
    sobel_combinado = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    return sobel_combinado

if __name__ == '__main__':
    img_lena = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    img_minha = cv2.imread('minhaimagem.webp', cv2.IMREAD_GRAYSCALE)
    
    if img_lena is None or img_minha is None:
        print("Erro: Não foi possível carregar uma ou ambas as imagens.")
        sys.exit()

    # Aplica o detector
    lena_sobel = detector_bordas_sobel(img_lena)
    minha_sobel = detector_bordas_sobel(img_minha)

    # Exibe os resultados
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(img_lena, cmap='gray')
    plt.title('Original (lena.png)')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(lena_sobel, cmap='gray')
    plt.title('Detector de Sobel')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(img_minha, cmap='gray')
    plt.title('Original (minhaimagem.webp)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(minha_sobel, cmap='gray')
    plt.title('Detector de Sobel')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()