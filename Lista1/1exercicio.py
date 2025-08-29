import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega as imagens
try:
    img_lena = cv2.imread('lena.png')
    img_aluno = cv2.imread('minhaimagem.webp')

    # Converte de BGR (padrão do OpenCV) para RGB para exibição correta no Matplotlib
    img_lena_rgb = cv2.cvtColor(img_lena, cv2.COLOR_BGR2RGB)
    img_aluno_rgb = cv2.cvtColor(img_aluno, cv2.COLOR_BGR2RGB)
except:
    print("Erro: Verifique se os nomes e os caminhos das imagens estão corretos.")
    exit()

def para_niveis_de_cinza(img_colorida):
    # Converte a imagem para float para evitar overflow nos cálculos
    img_float = img_colorida.astype(np.float32)
    R, G, B = img_float[:,:,0], img_float[:,:,1], img_float[:,:,2]
    
    # Média ponderada: Cinza = 0.299*R + 0.587*G + 0.114*B
    img_cinza = 0.299 * R + 0.587 * G + 0.114 * B
    
    # Converte de volta para 8-bits (0-255)
    return img_cinza.astype(np.uint8)

# Aplica a conversão
lena_cinza = para_niveis_de_cinza(img_lena_rgb)
aluno_cinza = para_niveis_de_cinza(img_aluno_rgb)

# Salva as imagens para o seu PDF
cv2.imwrite('lena_cinza.png', lena_cinza)
cv2.imwrite('aluno_cinza.png', aluno_cinza)

# Exibe os resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(lena_cinza, cmap='gray'), plt.title('Lena em Níveis de Cinza')
plt.subplot(1, 2, 2), plt.imshow(aluno_cinza, cmap='gray'), plt.title('Sua Imagem em Níveis de Cinza')
plt.show()