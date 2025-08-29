import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Implementação dos Algoritmos ---
# A. Histograma
def calcular_histograma(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist

# B. Histograma Normalizado
def calcular_histograma_normalizado(hist, num_pixels):
    return hist / num_pixels

# C. Histograma Acumulado
def calcular_histograma_acumulado(hist):
    return hist.cumsum()

# (i) Histograma de "unequalized.jpg" em tons de cinza 
img_unequalized = cv2.imread('unequalized.jpg')
unequalized_cinza = cv2.cvtColor(img_unequalized, cv2.COLOR_BGR2GRAY)
hist_unequalized = calcular_histograma(unequalized_cinza)

plt.figure()
plt.title("Histograma (A) - unequalized.jpg")
plt.xlabel("Intensidade")
plt.ylabel("Nº de Pixels")
plt.plot(hist_unequalized)
plt.xlim([0, 256])
plt.savefig('hist_A_unequalized.png')
plt.show()


# (ii) 3 histogramas (R, G, B) para "img_aluno" 
img_aluno = cv2.imread('minhaimagem.webp')
cores = ('b', 'g', 'r')
plt.figure()
plt.title("Histogramas (A) - Canais R, G, B de img_aluno")
plt.xlabel("Intensidade")
plt.ylabel("Nº de Pixels")
for i, cor in enumerate(cores):
    hist_cor = cv2.calcHist([img_aluno], [i], None, [256], [0, 256])
    plt.plot(hist_cor, color = cor)
    plt.xlim([0, 256])
plt.savefig('hist_A_rgb_aluno.png')
plt.show()


# (iii) Histogramas A, B, C, D para "img_aluno" em cinza 
aluno_cinza = cv2.cvtColor(img_aluno, cv2.COLOR_BGR2GRAY)
num_pixels_aluno = aluno_cinza.shape[0] * aluno_cinza.shape[1]

# A. Histograma
hist_A = calcular_histograma(aluno_cinza)

# B. Histograma Normalizado
hist_B = calcular_histograma_normalizado(hist_A, num_pixels_aluno)

# C. Histograma Acumulado
hist_C = calcular_histograma_acumulado(hist_A)

# D. Histograma Acumulado Normalizado
hist_D = calcular_histograma_acumulado(hist_B)

# Plotando os 4 histogramas
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1), plt.plot(hist_A), plt.title('A. Histograma')
plt.subplot(2, 2, 2), plt.plot(hist_B), plt.title('B. Histograma Normalizado')
plt.subplot(2, 2, 3), plt.plot(hist_C), plt.title('C. Histograma Acumulado')
plt.subplot(2, 2, 4), plt.plot(hist_D), plt.title('D. Histograma Acumulado Normalizado')
plt.savefig('histogramas_A_B_C_D_aluno_cinza.png')
plt.show()