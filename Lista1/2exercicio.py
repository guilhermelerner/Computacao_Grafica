import cv2
import numpy as np
import matplotlib.pyplot as plt

def gerar_negativo(img):
    return 255 - img

# Carrega as imagens originais
img_lena = cv2.imread('lena.png')
img_aluno = cv2.imread('minhaimagem.webp')

# Aplica o negativo
lena_negativo = gerar_negativo(img_lena)
aluno_negativo = gerar_negativo(img_aluno)

# Salva os resultados
cv2.imwrite('lena_negativo.png', lena_negativo)
cv2.imwrite('aluno_negativo.png', aluno_negativo)

# Exibe os resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(lena_negativo, cv2.COLOR_BGR2RGB)), plt.title('Negativo de Lena')
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(aluno_negativo, cv2.COLOR_BGR2RGB)), plt.title('Negativo da sua Imagem')
plt.show()