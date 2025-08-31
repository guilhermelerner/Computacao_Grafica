import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

def filtro_media_k_vizinhos(imagem, tamanho_janela, k):
    N, M = imagem.shape
    imagem_filtrada = np.copy(imagem)
    a = (tamanho_janela - 1) // 2
    
    if k > tamanho_janela**2:
        raise ValueError("k não pode ser maior que o número de pixels na janela.")

    for i in range(a, N - a):
        for j in range(a, M - a):
            janela = imagem[i - a : i + a + 1, j - a : j + a + 1].flatten()
            pixel_central = imagem[i, j]
            diferencas = np.abs(janela.astype(np.float32) - pixel_central)
            indices_ordenados = np.argsort(diferencas)
            k_vizinhos_proximos = janela[indices_ordenados[:k]]
            imagem_filtrada[i, j] = np.mean(k_vizinhos_proximos)
            
    return imagem_filtrada

if __name__ == '__main__':
    img_lena = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    img_minha = cv2.imread('minhaimagem.webp', cv2.IMREAD_GRAYSCALE)

    if img_lena is None or img_minha is None:
        print("Erro: Não foi possível carregar uma ou ambas as imagens.")
        sys.exit()

    # Parâmetros do filtro
    tamanho_janela_filtro = 5
    k_vizinhos = 15

    # Aplica o filtro
    lena_filtrada = filtro_media_k_vizinhos(img_lena, tamanho_janela_filtro, k_vizinhos)
    minha_filtrada = filtro_media_k_vizinhos(img_minha, tamanho_janela_filtro, k_vizinhos)

    # Exibe os resultados
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(img_lena, cmap='gray')
    plt.title('Original (lena.png)')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(lena_filtrada, cmap='gray')
    plt.title(f'Filtro Média k-Vizinhos (k={k_vizinhos})')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(img_minha, cmap='gray')
    plt.title('Original (minhaimagem.webp)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(minha_filtrada, cmap='gray')
    plt.title(f'Filtro Média k-Vizinhos (k={k_vizinhos})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()