import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os

def criar_filtro_gaussiano(formato, D0_ratio, passa_alta=False):
    """
    Cria uma máscara de filtro Gaussiano para o domínio da frequência.

    Args:
        formato (tuple): A forma (linhas, colunas) da imagem.
        D0_ratio (float): A frequência de corte como uma proporção do tamanho da imagem.
                           Valores típicos são entre 0.05 e 0.2.
        passa_alta (bool): Se True, cria um filtro passa-alta. Caso contrário, passa-baixa.

    Returns:
        numpy.ndarray: A máscara do filtro Gaussiano.
    """
    linhas, colunas = formato
    # Cria uma grade de coordenadas (u, v) centrada na origem
    u = np.arange(-colunas // 2, colunas // 2)
    v = np.arange(-linhas // 2, linhas // 2)
    u, v = np.meshgrid(u, v)

    # Calcula a distância Euclidiana D(u, v) do centro
    D = np.sqrt(u**2 + v**2)

    # Define a frequência de corte D0 em pixels
    # Usamos a menor dimensão para evitar distorções
    D0 = D0_ratio * min(linhas, colunas)

    # Evita divisão por zero se D0 for 0
    if D0 == 0:
        D0 = 1e-6

    # Equação do filtro Gaussiano Passa-Baixa
    filtro_passa_baixa = np.exp(-(D**2) / (2 * D0**2))

    if passa_alta:
        # Filtro Passa-Alta é o inverso do Passa-Baixa
        return 1 - filtro_passa_baixa
    else:
        return filtro_passa_baixa

def aplicar_filtro_frequencia(caminho_imagem, D0_ratio, tipo_filtro):
    """
    Carrega uma imagem, aplica um filtro Gaussiano e exibe os resultados.
    """
    try:
        # Carrega a imagem em escala de cinza
        imagem_original = io.imread(caminho_imagem, as_gray=True)
    except FileNotFoundError:
        print(f"ERRO: Imagem não encontrada em '{caminho_imagem}'. Verifique o nome e a pasta.")
        return

    # 1. Transformada de Fourier 2D e centralização do espectro
    fft_imagem = np.fft.fftshift(np.fft.fft2(imagem_original))

    # 2. Criação do filtro
    if tipo_filtro == 'passa_baixa':
        filtro = criar_filtro_gaussiano(imagem_original.shape, D0_ratio, passa_alta=False)
        titulo_filtro = f"Filtro Passa-Baixa (D0 = {D0_ratio*100:.0f}%)"
    elif tipo_filtro == 'passa_alta':
        filtro = criar_filtro_gaussiano(imagem_original.shape, D0_ratio, passa_alta=True)
        titulo_filtro = f"Filtro Passa-Alta (D0 = {D0_ratio*100:.0f}%)"
    else:
        print("Tipo de filtro inválido. Use 'passa_baixa' ou 'passa_alta'.")
        return
        
    # 3. Aplicação do filtro (multiplicação elemento a elemento)
    fft_filtrada = fft_imagem * filtro

    # 4. Transformada Inversa de Fourier
    imagem_filtrada = np.fft.ifft2(np.fft.ifftshift(fft_filtrada))
    # Pegamos o valor absoluto para obter a imagem real
    imagem_filtrada = np.abs(imagem_filtrada)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(imagem_original, cmap='gray')
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')

    axes[1].imshow(filtro, cmap='gray')
    axes[1].set_title(titulo_filtro)
    axes[1].axis('off')

    axes[2].imshow(imagem_filtrada, cmap='gray')
    axes[2].set_title('Imagem Filtrada')
    axes[2].axis('off')
    
    nome_arquivo = os.path.basename(caminho_imagem)
    plt.suptitle(f'Resultados para: {nome_arquivo}', fontsize=16)
    plt.show()

if __name__ == "__main__":

    nome_img_teste = "teste.tif"
    nome_img_aluno = "minhaimagem.webp" 

    # Frequência de corte (D0). 0.1 significa 10% da menor dimensão da imagem.
    # Experimente valores como 0.05 (mais forte) ou 0.2 (mais suave).
    frequencia_de_corte = 0.1

    # Constrói o caminho para a pasta 'imagens'
    script_dir = os.path.dirname(os.path.realpath(__file__))
    caminho_imagens = os.path.join(script_dir, "imagens")

    imagens_para_processar = [
        os.path.join(caminho_imagens, nome_img_teste),
        os.path.join(caminho_imagens, nome_img_aluno)
    ]

    # --- PROCESSAMENTO ---
    for caminho in imagens_para_processar:
        print(f"\n--- Processando a imagem: {os.path.basename(caminho)} ---")
        
        # Aplica o filtro Passa-Baixa
        print("Aplicando Filtro Passa-Baixa...")
        aplicar_filtro_frequencia(caminho, frequencia_de_corte, 'passa_baixa')

        # Aplica o filtro Passa-Alta
        print("Aplicando Filtro Passa-Alta...")
        aplicar_filtro_frequencia(caminho, frequencia_de_corte, 'passa_alta')