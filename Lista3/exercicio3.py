import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import os

def aplicar_filtro_por_imagem(caminho_imagem_original, caminho_filtro):
    """
    Aplica um filtro no domínio da frequência definido por uma imagem.

    Args:
        caminho_imagem_original (str): O caminho para a imagem a ser filtrada.
        caminho_filtro (str): O caminho para a imagem que servirá de máscara de filtro.
    """
    try:

        imagem_original = io.imread(caminho_imagem_original, as_gray=True)

        filtro_imagem = io.imread(caminho_filtro, as_gray=True)
    except FileNotFoundError as e:
        print(f"ERRO: Não foi possível encontrar um dos arquivos. Verifique os caminhos.")
        print(e)
        return

    if imagem_original.shape != filtro_imagem.shape:
        print("Aviso: A imagem e o filtro têm tamanhos diferentes.")
        print(f"Redimensionando o filtro de {filtro_imagem.shape} para {imagem_original.shape}.")
        filtro_imagem = transform.resize(filtro_imagem, imagem_original.shape)

    if filtro_imagem.max() > 1:
        filtro_normalizado = filtro_imagem / 255.0
    else:
        filtro_normalizado = filtro_imagem 
    
    # 1. Transformada de Fourier 2D da imagem original e centralização do espectro
    fft_imagem = np.fft.fftshift(np.fft.fft2(imagem_original))

    # 2. Aplicação do filtro (multiplicação elemento a elemento)
    fft_filtrada = fft_imagem * filtro_normalizado

    # 3. Transformada Inversa de Fourier para voltar ao domínio do espaço
    imagem_filtrada = np.fft.ifft2(np.fft.ifftshift(fft_filtrada))
    imagem_filtrada = np.abs(imagem_filtrada) 

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(imagem_original, cmap='gray')
    axes[0].set_title('Imagem Original (arara.png)')
    axes[0].axis('off')

    # Mostra a imagem original do filtro, não a normalizada, para ser fiel ao arquivo
    axes[1].imshow(filtro_imagem, cmap='gray')
    axes[1].set_title('Filtro Rejeita-Banda (arara_filtro.png)')
    axes[1].axis('off')

    axes[2].imshow(imagem_filtrada, cmap='gray')
    axes[2].set_title('Imagem Filtrada')
    axes[2].axis('off')

    plt.suptitle('Aplicação de Filtro Rejeita-Banda a partir de Imagem', fontsize=16)
    plt.show()

if __name__ == "__main__":
    # Nomes dos arquivos a serem processados
    nome_img_original = "arara.png"
    nome_img_filtro = "arara_filtro.png"

    # Constrói o caminho completo para a pasta 'imagens'
    script_dir = os.path.dirname(os.path.realpath(__file__))
    caminho_imagens = os.path.join(script_dir, "imagens")

    # Caminhos completos para os arquivos
    caminho_arara = os.path.join(caminho_imagens, nome_img_original)
    caminho_filtro_arara = os.path.join(caminho_imagens, nome_img_filtro)

    print(f"Processando '{nome_img_original}' com o filtro '{nome_img_filtro}'...")
    aplicar_filtro_por_imagem(caminho_arara, caminho_filtro_arara)
    print("Processo concluído.")