import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os


def _criar_gaussiana_passa_baixa(formato, D0_ratio):
    """Função auxiliar para criar uma máscara Gaussiana passa-baixa."""
    linhas, colunas = formato
    u = np.arange(-colunas // 2, colunas // 2)
    v = np.arange(-linhas // 2, linhas // 2)
    u, v = np.meshgrid(u, v)
    D = np.sqrt(u**2 + v**2)
    D0 = D0_ratio * min(linhas, colunas)
    if D0 == 0: D0 = 1e-6 
    
    return np.exp(-(D**2) / (2 * D0**2))

def criar_filtro_de_banda(formato, raio_interno_ratio, raio_externo_ratio, tipo='passa_banda'):
    """
    Cria um filtro passa-banda ou rejeita-banda Gaussiano.

    Args:
        formato (tuple): A forma (linhas, colunas) da imagem.
        raio_interno_ratio (float): Raio do círculo interno da banda.
        raio_externo_ratio (float): Raio do círculo externo da banda.
        tipo (str): 'passa_banda' ou 'rejeita_banda'.

    Returns:
        numpy.ndarray: A máscara do filtro de banda.
    """
    if raio_interno_ratio >= raio_externo_ratio:
        raise ValueError("O raio interno deve ser menor que o raio externo.")

    # Cria os dois filtros passa-baixa que formarão a banda
    filtro_externo = _criar_gaussiana_passa_baixa(formato, raio_externo_ratio)
    filtro_interno = _criar_gaussiana_passa_baixa(formato, raio_interno_ratio)
    
    # O filtro passa-banda é a subtração do interno do externo
    filtro_passa_banda = filtro_externo - filtro_interno

    if tipo == 'rejeita_banda':
        return 1 - filtro_passa_banda
    else: # 'passa_banda'
        return filtro_passa_banda

def aplicar_filtro_de_banda(caminho_imagem, raio_interno_ratio, raio_externo_ratio, tipo_filtro):
    """
    Carrega uma imagem, aplica um filtro de banda e exibe os resultados.
    """
    try:
        imagem_original = io.imread(caminho_imagem, as_gray=True)
    except FileNotFoundError:
        print(f"ERRO: Imagem não encontrada em '{caminho_imagem}'. Verifique o nome e a pasta.")
        return

    # 1. Transformada de Fourier e centralização
    fft_imagem = np.fft.fftshift(np.fft.fft2(imagem_original))

    # 2. Criação do filtro de banda
    filtro = criar_filtro_de_banda(imagem_original.shape, raio_interno_ratio, raio_externo_ratio, tipo=tipo_filtro)
    
    # 3. Aplicação do filtro
    fft_filtrada = fft_imagem * filtro

    # 4. Transformada Inversa de Fourier
    imagem_filtrada = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_filtrada)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(imagem_original, cmap='gray')
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')

    axes[1].imshow(filtro, cmap='gray')
    axes[1].set_title(f"Filtro {tipo_filtro.replace('_', '-').title()}")
    axes[1].axis('off')

    axes[2].imshow(imagem_filtrada, cmap='gray')
    axes[2].set_title('Imagem Filtrada')
    axes[2].axis('off')
    
    nome_arquivo = os.path.basename(caminho_imagem)
    plt.suptitle(f'Resultados para: {nome_arquivo}', fontsize=16)
    plt.show()

if __name__ == "__main__":

    raio_interno = 0.1  
    raio_externo = 0.3 

    # Nomes das imagens
    nome_img_teste = "teste.tif"
    nome_img_aluno = "minhaimagem.webp" 

    # Constrói o caminho para a pasta 'imagens'
    script_dir = os.path.dirname(os.path.realpath(__file__))
    caminho_imagens = os.path.join(script_dir, "imagens")

    imagens_para_processar = [
        os.path.join(caminho_imagens, nome_img_teste),
        os.path.join(caminho_imagens, nome_img_aluno)
    ]

    for caminho in imagens_para_processar:
        print(f"\n--- Processando a imagem: {os.path.basename(caminho)} ---")
        
        # Aplica o filtro Passa-Banda
        print("Aplicando Filtro Passa-Banda...")
        aplicar_filtro_de_banda(caminho, raio_interno, raio_externo, 'passa_banda')

        # Aplica o filtro Rejeita-Banda
        print("Aplicando Filtro Rejeita-Banda...")
        aplicar_filtro_de_banda(caminho, raio_interno, raio_externo, 'rejeita_banda')