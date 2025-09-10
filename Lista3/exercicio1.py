import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os

def plot_spectrum(image_path, title):
    """Calcula e plota o espectro de Fourier de uma imagem."""
    try:
        image = io.imread(image_path, as_gray=True)
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em '{image_path}'. Verifique o nome do arquivo e a pasta.")
        return
    except Exception as e:
        print(f"Erro ao carregar a imagem {image_path}: {e}")
        return

    # Calcula a Transformada de Fourier 2D
    f_transform = np.fft.fft2(image)

    # Centraliza o componente de frequência zero
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Calcula o espectro de magnitude e aplica a escala logarítmica
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)

    # Plota a imagem original e seu espectro
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Imagem Original: {title}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(f'Espectro de Fourier: {title}')
    plt.axis('off')

    plt.show()

script_dir = os.path.dirname(os.path.realpath(__file__))

base_path = os.path.join(script_dir, "imagens")

image_paths = {
    "arara": os.path.join(base_path, "arara.png"),
    "barra1": os.path.join(base_path, "barra1.png"),
    "barra2": os.path.join(base_path, "barra2.png"),
    "barra3": os.path.join(base_path, "barra3.png"),
    "barra4": os.path.join(base_path, "barra4.png"),
    "teste": os.path.join(base_path, "teste.tif"),
}

for name, path in image_paths.items():
    plot_spectrum(path, name)