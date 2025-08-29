import cv2
import numpy as np
import matplotlib.pyplot as plt

def operador_logaritmico(img):
    # 1. Converte a imagem para float ANTES de qualquer cálculo
    img_float = img.astype(np.float32)
    
    # 2. Agora, o cálculo de 'c' é seguro e não vai estourar o limite de 255
    c = 255 / np.log(1 + np.max(img_float))
    
    # 3. Aplica a transformação logarítmica na imagem em float
    img_log = c * np.log(1 + img_float)
    
    # Garante que os valores fiquem no intervalo [0, 255] antes de converter de volta
    img_log = np.clip(img_log, 0, 255)

    # 4. Converte o resultado de volta para o formato de imagem (uint8)
    return img_log.astype(np.uint8)

# Carrega as imagens
img_lena = cv2.imread('lena.png')
img_aluno = cv2.imread('minhaimagem.webp') # Mantive o nome que você está usando

# Aplica o operador
lena_log = operador_logaritmico(img_lena)
aluno_log = operador_logaritmico(img_aluno)

# Salva os resultados
cv2.imwrite('lena_log.png', lena_log)
cv2.imwrite('aluno_log.png', aluno_log)

# Exibe os resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(lena_log, cv2.COLOR_BGR2RGB)), plt.title('Lena Logarítmico')
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(aluno_log, cv2.COLOR_BGR2RGB)), plt.title('Sua Imagem Logarítmica')
plt.show()