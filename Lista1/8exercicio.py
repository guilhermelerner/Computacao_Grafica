import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para equalizar e mostrar resultados
def equalizar_e_mostrar(nome_arquivo, titulo):
    img = cv2.imread(nome_arquivo)
    
    # Se a imagem for colorida, converte para cinza primeiro
    if len(img.shape) > 2:
        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_cinza = img
        
    # Aplica a equalização de histograma
    img_equalizada = cv2.equalizeHist(img_cinza)
    
    # Calcula histogramas para comparação
    hist_original = cv2.calcHist([img_cinza], [0], None, [256], [0, 256])
    hist_equalizado = cv2.calcHist([img_equalizada], [0], None, [256], [0, 256])
    
    # Salva a imagem equalizada
    cv2.imwrite(f'{nome_arquivo.split(".")[0]}_equalizada.png', img_equalizada)
    
    # Exibe tudo
    plt.figure(figsize=(18, 6))
    
    plt.subplot(2, 2, 1), plt.imshow(img_cinza, cmap='gray'), plt.title(f'Original - {titulo}')
    plt.subplot(2, 2, 2), plt.imshow(img_equalizada, cmap='gray'), plt.title(f'Equalizada - {titulo}')
    
    plt.subplot(2, 2, 3), plt.plot(hist_original), plt.title('Histograma Original')
    plt.subplot(2, 2, 4), plt.plot(hist_equalizado), plt.title('Histograma Equalizado')
    
    plt.tight_layout()
    plt.savefig(f'comparacao_equalizacao_{nome_arquivo.split(".")[0]}.png')
    plt.show()


# Aplica a equalização nas três imagens solicitadas 
equalizar_e_mostrar('lena.png', 'Lena')
equalizar_e_mostrar('unequalized.jpg', 'Unequalized')
equalizar_e_mostrar('minhaimagem.webp', 'Sua Imagem')