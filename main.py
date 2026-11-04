from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from huffmancodec import HuffmanCodec

# Alínea 1 - Histograma
def alinea_1_histograma(P, nome_ficheiro="fonte"):
    # "A" sendo o alfabeto que vai conter todos os valores únicos distintos (os símbolos) encontrados nos dados brutos (P)
    # Contagens sendo a frequência dos símbolos. Quantas vezes cada símbolo de A apareceu em P
    A, contagens = np.unique(P, return_counts=True)
    probabilidades = contagens / len(P)
    
    plt.figure(figsize=(10, 5))
    plt.bar(A, probabilidades, width=1.0, edgecolor="black")    
    plt.title(f"Histograma da fonte ({nome_ficheiro})")
    plt.xlabel("Símbolos")
    plt.ylabel("Probabilidade")

    # Salva a imagem no repositório e fecha a figura para liberar memória
    plt.savefig(f"histograma_{nome_ficheiro}.png", bbox_inches='tight')
    plt.close()
    
    return A, probabilidades

# Alínea 2 - Entropia
def alinea_2_entropia(probabilidades):
    # Fórmula da entropia de Shannon
    entropia = -np.sum(probabilidades * np.log2(probabilidades))
    return entropia

# Alínea 4 - Número médio de bits/símbolo (L) e a variância
def alinea_4_huffman(P, probabilidades):
    codec = HuffmanCodec.from_data(P)
    symbols, lengths = codec.get_code_len()

    # Convertendo para array do NumPy
    lengths = np.array(lengths)

    L = np.sum(probabilidades * lengths)
    variancia = np.sum(probabilidades * ((lengths - L) ** 2))

    return variancia, L

# Alínea 5 - Agrupamento de símbolos (extensão da fonte)
def alinea_5_agrupamento(P, nome_ficheiro="fonte"):
    # Se o número total de elementos for ímpar, removemos o último para garantir que não sobram elementos ao formar os pares
    if len(P) % 2 != 0:
        P = P[:-1]
        
    # Agrupando símbolos em pares contíguos (os que estão lado a lado na seq. original) e remodelando para "n" linhas(-1) e 2 colunas
    # reshape() está fazendo o contrário do que flatten() faz, está transformando em uma matriz
    pares = P.reshape(-1, 2)
    
    # Exatamente o mesmo processo da linha 1 mas com axis=0 para que cada linha inteira (o par) seja interpretada como um símbolo único
    pares_unicos, contagens = np.unique(pares, axis=0, return_counts=True)
    prob_pares = contagens / len(pares)
    
    # Visualização do histograma da alínea 3
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(pares_unicos)), prob_pares, width=1.0, edgecolor="black")    
    plt.title(f"Histograma da fonte com os símbolos agrupados em pares ({nome_ficheiro})")
    plt.xlabel("Índice do par de símbolos")
    plt.ylabel("Probabilidade")
    plt.savefig(f"histograma_pares_{nome_ficheiro}.png", bbox_inches='tight')
    plt.close()
    
    # Cálculo da entropia do bloco, a entropia conjunta do par, agora nosso alfabeto tem 256x256
    entropia_bloco = -np.sum(prob_pares * np.log2(prob_pares))
    
    # A entropia por símbolo original ajustada. Quanto "custou" cada pixel da dupla
    entropia_simbolo_agrupado = entropia_bloco / 2
    
    return entropia_bloco, entropia_simbolo_agrupado

# Alínea 3 - Distribuição estatística
if __name__ == "__main__":
    ficheiros = ["lena.bmp", "ct1.bmp", "binaria.bmp", "saxriff.wav"]
    
    base = Path(__file__).resolve().parent / "data"
    
    for ficheiro in ficheiros:
        caminho = base / ficheiro
        
        print("\n")
        print(f"{'-'*70}")
        print(f"A analisar a fonte: {ficheiro}")
        print(f"{'-'*70}")
        
        # 1. Carregando a fonte e distinguindo entre .bmp e .wav
        if ficheiro.endswith(".bmp"):
            dados = plt.imread(caminho)
            # 2 dimensões da imagem: altura e largura
            if dados.ndim > 2:
                # Pega todas as linhas(:), todas as colunas(:) mas apenas a primeira fatia de profundidade(0=Red)
                dados = dados[:, :, 0]
            # flatten() vai pegar a matriz de linhas e colunas e achar para apenas 1 dimensão
            fonte = dados.flatten()
            
        elif ficheiro.endswith(".wav"):
            _, dados = wav.read(caminho)
            # Se tiver mais de 1 dimensão (áudio estéreo com lado esq e dir) vai transformar em mono
            if dados.ndim > 1:
                dados = dados[:, 0]
            fonte = dados.flatten()
            
        else:
            print(f"Formato de {ficheiro} não suportado")
            continue
            
        # 2. Calculando o Histograma
        # Guarda "A" em alfabeto e "probabilidades" em probs
        alfabeto, probs = alinea_1_histograma(fonte, ficheiro)
        
        # 3. Calculando a Entropia
        entropia = alinea_2_entropia(probs)
        
        print(f"A entropia de {ficheiro}: é de {entropia:.6f} bits/símbolo")
        
        # 4. Calculando redundância e a taxa de compressão
        # redundancia = 8 - entropia
        # taxa_compressao = 8 / entropia
        
        # print(f"A redundância de {ficheiro}: é de {redundancia:.6f} bits/símbolo")
        # print(f"A taxa de compressão de {ficheiro}: é de {taxa_compressao:.6f} bits/símbolo")

        # Alinea 4 - Média de bits por símbolo
        variancia, media_bits_simbolo = alinea_4_huffman(fonte, probs)
        
        print(f"A média de bits por símbolo de {ficheiro} é de: {media_bits_simbolo:.6f} bits/símbolo")
        print(f"A variância de {ficheiro} é de: {variancia:.6f} bits^2")
        print(f"{'-'*70}")
        
        # Alínea 5 - Agrupamento
        entropia_bloco, entropia_agrupada = alinea_5_agrupamento(fonte, ficheiro)
        
        print(f"A entropia do bloco em pares de {ficheiro}: é de {entropia_bloco:.6f} bits/bloco")
        print(f"A entropia agrupada por símbolo original: é de {entropia_agrupada:.6f} bits/símbolo")
        print(f"{'-'*70}")