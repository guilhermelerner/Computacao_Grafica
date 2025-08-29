import cv2
import numpy as np
import matplotlib.pyplot as plt

# Vamos usar as imagens em tons de cinza geradas no primeiro exercício
lena_cinza = cv2.imread('lena_cinza.png', cv2.IMREAD_GRAYSCALE)
aluno_cinza = cv2.imread('aluno_cinza.png', cv2.IMREAD_GRAYSCALE)

def fatiar_planos_de_bits(img_cinza):
    planos = []
    for i in range(8):
        # Extrai o i-ésimo bit de cada pixel
        plano = (img_cinza >> i) & 1
        # Multiplica por 255 para tornar o bit 1 visível (branco)
        planos.append(plano * 255)
    return planos

# Fatia os planos de bits para ambas as imagens 
planos_lena = fatiar_planos_de_bits(lena_cinza)
planos_aluno = fatiar_planos_de_bits(aluno_cinza)

# Salva e exibe os planos para a imagem Lena
plt.figure(figsize=(12, 6))
plt.suptitle('Planos de Bits - Lena', fontsize=16)
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(planos_lena[i], cmap='gray')
    plt.title(f'Bit {i}')
    cv2.imwrite(f'lena_plano_bit_{i}.png', planos_lena[i])
plt.show()

# Salva e exibe os planos para a sua imagem
plt.figure(figsize=(12, 6))
plt.suptitle('Planos de Bits - Sua Imagem', fontsize=16)
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(planos_aluno[i], cmap='gray')
    plt.title(f'Bit {i}')
    cv2.imwrite(f'aluno_plano_bit_{i}.png', planos_aluno[i])
plt.show()