import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

def image_convolution(f, w):

    N, M = f.shape
    n, m = w.shape

    a = int((n - 1) / 2)
    b = int((m - 1) / 2)
    g = np.zeros(f.shape, dtype=np.float32)

    for x in range(a, N - a):
        for y in range(b, M - b):
            sub_f = f[x - a:x + a + 1, y - b:y + b + 1]
            g[x, y] = np.sum(np.multiply(sub_f, w))
    
    g = cv2.normalize(g, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return g

def filtro_suavizacao_media(imagem, tamanho_janela):
    kernel = np.ones((tamanho_janela, tamanho_janela), np.float32)
    kernel = kernel / (tamanho_janela * tamanho_janela)
    imagem_filtrada = image_convolution(imagem, kernel)
    return imagem_filtrada

if __name__ == '__main__':
    # Carrega as imagens em tons de cinza
    img_lena = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    img_minha = cv2.imread('minhaimagem.webp', cv2.IMREAD_GRAYSCALE)

    if img_lena is None or img_minha is None:
        print("Erro: Não foi possível carregar uma ou ambas as imagens. Verifique os nomes e caminhos dos arquivos.")
        sys.exit()

    # Define o tamanho da janela do filtro
    tamanho_janela_filtro = 7

    # Aplica o filtro em ambas as imagens
    lena_filtrada = filtro_suavizacao_media(img_lena, tamanho_janela_filtro)
    minha_filtrada = filtro_suavizacao_media(img_minha, tamanho_janela_filtro)

    # Exibe os resultados
    plt.figure(figsize=(12, 12))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_lena, cmap='gray')
    plt.title('Original (lena.png)')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(lena_filtrada, cmap='gray')
    plt.title(f'Filtro da Média {tamanho_janela_filtro}x{tamanho_janela_filtro}')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(img_minha, cmap='gray')
    plt.title('Original (minhaimagem.webp)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(minha_filtrada, cmap='gray')
    plt.title(f'Filtro da Média {tamanho_janela_filtro}x{tamanho_janela_filtro}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()