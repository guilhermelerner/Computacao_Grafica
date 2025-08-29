import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalizacao_contraste(img, novo_min, novo_max):
    # Converte para float para os cálculos
    img_float = img.astype(np.float32)
    
    min_original = np.min(img_float)
    max_original = np.max(img_float)
    
    # Aplica a fórmula da normalização
    img_normalizada = (img_float - min_original) * (novo_max - novo_min) / (max_original - min_original) + novo_min
    
    # Garante que os valores fiquem no intervalo [0, 255]
    img_normalizada = np.clip(img_normalizada, 0, 255)
    
    return img_normalizada.astype(np.uint8)

# Carrega as imagens
img_lena = cv2.imread('lena.png')
img_aluno = cv2.imread('minhaimagem.webp')

# Aplica a normalização com a faixa [0, 100]
lena_normalizada = normalizacao_contraste(img_lena, 0, 100)
aluno_normalizado = normalizacao_contraste(img_aluno, 0, 100)

# Salva os resultados
cv2.imwrite('lena_normalizada_0_100.png', lena_normalizada)
cv2.imwrite('aluno_normalizado_0_100.png', aluno_normalizado)

# Exibe os resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(lena_normalizada, cv2.COLOR_BGR2RGB)), plt.title('Lena Normalizada (0-100)')
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(aluno_normalizado, cv2.COLOR_BGR2RGB)), plt.title('Sua Imagem Normalizada (0-100)')
plt.show()