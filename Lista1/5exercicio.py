import cv2
import numpy as np
import matplotlib.pyplot as plt

def operador_potencia(img, c, gamma):
    # Normaliza a imagem para a faixa [0, 1]
    img_norm = img.astype(np.float32) / 255.0
    
    # Aplica a correção gama
    img_gama = c * (img_norm ** gamma)
    
    # Normaliza de volta para [0, 255]
    img_final = np.clip(img_gama * 255, 0, 255)
    
    return img_final.astype(np.uint8)

# Carrega as imagens
img_lena = cv2.imread('lena.png')
img_aluno = cv2.imread('minhaimagem.webp')

# Aplica o operador de potência com c=2 e gamma=2 
lena_potencia = operador_potencia(img_lena, 2, 2)
aluno_potencia = operador_potencia(img_aluno, 2, 2)

# Salva os resultados
cv2.imwrite('lena_potencia_c2_g2.png', lena_potencia)
cv2.imwrite('aluno_potencia_c2_g2.png', aluno_potencia)

# Exibe os resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(lena_potencia, cv2.COLOR_BGR2RGB)), plt.title('Lena Potência (c=2, γ=2)')
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(aluno_potencia, cv2.COLOR_BGR2RGB)), plt.title('Sua Imagem Potência (c=2, γ=2)')
plt.show()