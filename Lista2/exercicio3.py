import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

def filtro_suavizacao_mediana(imagem, tamanho_janela):
    imagem_filtrada = cv2.medianBlur(imagem, tamanho_janela)
    return imagem_filtrada

if __name__ == '__main__':
    img_lena = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    img_minha = cv2.imread('minhaimagem.webp', cv2.IMREAD_GRAYSCALE)
    
    if img_lena is None or img_minha is None:
        print("Erro: Não foi possível carregar uma ou ambas as imagens.")
        sys.exit()
    
    tamanho_janela_filtro = 5

    # Aplica o filtro
    lena_filtrada = filtro_suavizacao_mediana(img_lena, tamanho_janela_filtro)
    minha_filtrada = filtro_suavizacao_mediana(img_minha, tamanho_janela_filtro)

    # Exibe os resultados
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(img_lena, cmap='gray')
    plt.title('Original (lena.png)')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(lena_filtrada, cmap='gray')
    plt.title(f'Filtro da Mediana {tamanho_janela_filtro}x{tamanho_janela_filtro}')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(img_minha, cmap='gray')
    plt.title('Original (minhaimagem.webp)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(minha_filtrada, cmap='gray')
    plt.title(f'Filtro da Mediana {tamanho_janela_filtro}x{tamanho_janela_filtro}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()