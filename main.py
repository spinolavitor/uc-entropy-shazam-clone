from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from huffmancodec import HuffmanCodec

# Função genérica para fazer o histograma e evitar repetição
def histograma(valores_x, probabilidades, titulo, legenda_x, caminho_salvar):
    plt.figure(figsize=(10, 5))
    plt.bar(valores_x, probabilidades, width=1.0, edgecolor="black")    
    plt.title(titulo)
    plt.xlabel(legenda_x)
    plt.ylabel("Probabilidade")
    # Salvando a imagem no caminho especificado e fechando a figura
    plt.savefig(caminho_salvar, bbox_inches='tight')
    plt.close()
    
# Função genérica para calcular alfabeto e probabilidades, adaptado para suportar 1D e 2D com o "eixo"
def calcular_probabilidades(dados, eixo=None):
    # O alfabeto contêm todos os valores únicos distintos (os símbolos) encontrados em "dados"
    # Contagens sendo a frequência dos símbolos. Quantas vezes cada símbolo do alfabeto apareceu em dados
    alfabeto, contagens = np.unique(dados, axis=eixo, return_counts=True)
    probabilidades = contagens / len(dados)
    return alfabeto, probabilidades

# Se tiver mais de 1 dimensão (áudio estéreo com lado esq e dir) vai transformar em mono
def carregar_audio_mono(caminho_ficheiro):
    _, dados = wav.read(caminho_ficheiro)
    if dados.ndim > 1:
        dados = dados[:, 0]
    return dados

# Alínea 1 - Histograma
def alinea_1_histograma(P, nome, pasta_destino):
    # Aqui utilizamos a função genérica para calcular as probabilidades com o "unique"
    A, probabilidades = calcular_probabilidades(P)
    
    # Definimos o caminho completo de onde a imagem será salva
    caminho_salvar = pasta_destino / f"histograma_{nome}.png"
    
    # Aqui, utilizamos a função genérica para montar e salvar o histograma
    histograma(A, probabilidades, f"Histograma da fonte ({nome})", "Símbolos", caminho_salvar)
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
def alinea_5_agrupamento(P, nome, pasta_destino):
    # Se o número total de elementos for ímpar, removemos o último para garantir que não sobram elementos ao formar os pares
    if len(P) % 2 != 0:
        P = P[:-1]
        
    # Agrupando símbolos em pares contíguos (os que estão lado a lado na seq. original) e remodelando para "n" linhas(-1) e 2 colunas
    # reshape() está fazendo o contrário do que flatten() faz, está transformando em uma matriz
    pares = P.reshape(-1, 2)
    
    # Com "eixo=0" sendo o axis=0, o NumPy identifica cada linha inteira (o par) como um símbolo único
    pares_unicos, prob_pares = calcular_probabilidades(pares, eixo=0)
    
    caminho_salvar = pasta_destino / f"histograma_pares_{nome}.png"
    
    # Visualização e salvamento do histograma
    histograma(
        range(len(pares_unicos)), 
        prob_pares,  
        f"Histograma da fonte agrupada - ({nome})", 
        "Índice do par de símbolos",
        caminho_salvar
    )
    
    # Cálculo da entropia do bloco, a entropia conjunta do par, agora nosso alfabeto tem 256x256
    entropia_bloco = -np.sum(prob_pares * np.log2(prob_pares))
    
    # A entropia por símbolo original ajustada. Quanto "custou" cada pixel da dupla
    entropia_simbolo_agrupado = entropia_bloco / 2
    
    return entropia_bloco, entropia_simbolo_agrupado

# Alínea 6a - Informação mútua com janela deslizante
def alinea_6a_informacao_mutua(query, target, alfabeto=None, passo=1):
    numero_query = len(query)
    numero_target = len(target)
    
    info_mutua = []
    
    # 1. H(X): Entropia da query
    _, probs_query = calcular_probabilidades(query)
    entropia_query= alinea_2_entropia(probs_query)
    
    # 2. Deslizamos a janela sobre o target
    # range(início, parada, salto)
    # A parada é uma trava de segurança para que o "for" não tente ler um pedaço que não existe da música
    for i in range(0, numero_target - numero_query + 1, passo):
        # Recorte de target na posição i + o tamanho da query
        janela = target[i : i + numero_query]
        
        # H(Y): Entropia da janela atual
        _, probs_janela = calcular_probabilidades(janela)
        entropia_janela = alinea_2_entropia(probs_janela)
        
        # H(X, Y): Entropia conjunta par a par (junta a query com a janela para formar o par)
        # Emparelha a amostra "k" da query com a amostra "k" da janela
        pares = np.column_stack((query, janela))
        # Passamos eixo=0 para os pares
        _, probs_pares = calcular_probabilidades(pares, eixo=0)
        entropia_pares = alinea_2_entropia(probs_pares)
        
        # I(X; Y) = H(X) + H(Y) - H(X, Y)
        # I(X; Y): Informação mútua
        I_XY = entropia_query + entropia_janela - entropia_pares
        info_mutua.append(I_XY)
        
    return np.array(info_mutua)

# Alínea 6b - Localização de sinal via informação mútua
def alinea_6b_localizacao(caminho_base, query_audio, passo, pasta_destino):
    print("\n")
    print(f"{'-'*70}")
    print("Iniciando a alínea 6b - localização de sinal")

    # 1. Carregando apenas os targets, a query já vem pronta nos parâmetros
    caminho_target1 = caminho_base / "target01 - repeat.wav"
    caminho_target2 = caminho_base / "target02 - repeatNoise.wav"

    # Aplicando a função para transformar os áudios em mono
    target1_audio = carregar_audio_mono(caminho_target1)
    target2_audio = carregar_audio_mono(caminho_target2)

    # 2. Cálculo da informação mútua para cada target
    print("Calculando a informação mútua para target 01 (repeat.wav)")
    info_mutua_target1 = alinea_6a_informacao_mutua(query_audio, target1_audio, passo=passo)
    
    print("Calculando a informação mútua para target 02 (repeatNoise.wav)")
    info_mutua_target2 = alinea_6a_informacao_mutua(query_audio, target2_audio, passo=passo)
    print(f"{'-'*70}")

    # 3. Visualização e salvamento gráfico
    plt.figure(figsize=(12, 5))
    janelas1 = np.arange(len(info_mutua_target1))
    janelas2 = np.arange(len(info_mutua_target2))
    plt.plot(janelas1, info_mutua_target1, label="Target 01 (repeat.wav)", color="blue", linewidth=1.5)
    plt.plot(janelas2, info_mutua_target2, label="Target 02 (repeatNoise.wav)", color="red", linestyle="--", alpha=0.8, linewidth=1.5)
    plt.title("Evolução da informação mútua ao longo do tempo")
    plt.xlabel("Índice da janela deslizante")
    plt.ylabel("Informação mútua (bits)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    caminho_salvar = pasta_destino / "evolucao_informacao_mutua_targets.png"
    plt.savefig(caminho_salvar, dpi=150)
    plt.close()
    
    return info_mutua_target1, info_mutua_target2

# Alínea 6c - Simulador de identificação de música (Shazam)
def alinea_6c_shazam(caminho_base, query_audio, passo, pasta_destino):
    print("\n")
    print(f"{'-'*70}")
    print("Iniciando a alínea 6c - simulador de identificação de música (Shazam)")

    # 1. Encontrando todos os arquivos "Song*.wav" na pasta
    ficheiros_song = list(caminho_base.glob("Song*.wav"))
    
    if not ficheiros_song:
        print("Nenhum arquivo 'Song*.wav' encontrado no diretório especificado")
        return

    resultados_shazam = []
    plt.figure(figsize=(12, 6))

    # 2. Iterando sobre cada música encontrada
    for caminho_song in ficheiros_song:
        nome_ficheiro = caminho_song.name
        
        target_audio = carregar_audio_mono(caminho_song)
        
        # Envia saxriff e a música em questão para a função
        info_mutua_song = alinea_6a_informacao_mutua(query_audio, target_audio, passo=passo)
        # Vasculha o array de números gerado anteriormente e extrai o maior deles
        info_mutua_max = np.max(info_mutua_song)
        resultados_shazam.append((nome_ficheiro, info_mutua_max))
        
        plt.plot(info_mutua_song, label=nome_ficheiro, alpha=0.8, linewidth=1.5, marker='o')

    # 3. Finalização visual e salvamento do gráfico
    plt.title("Simulador Shazam: evolução da informação mútua nas músicas")
    plt.xlabel("Índice da janela deslizante")
    plt.ylabel("Informação mútua (bits)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    caminho_salvar = pasta_destino / "ranking_shazam_informacao_mutua.png"
    plt.savefig(caminho_salvar, dpi=150)
    plt.close()

    # 4. Ordenação e apresentação do ranking
    # "key=lambda" serve para ordenar o tuplo pelo número (índice [1]) ao invés do nome (índice [0])
    # "reverse=True" ordena de maneira decrescente como o enunciado pede
    resultados_shazam.sort(key=lambda x: x[1], reverse=True)
    
    print("Resultados da pesquisa (ranking)")
    print(f"{'-'*70}")
    for i, (ficheiro, info_mutua_max) in enumerate(resultados_shazam, start=1):
        print(f"{i}º lugar: {ficheiro} -> informação mútua máxima = {info_mutua_max:.6f} bits")
    print(f"{'-'*70}")
    
    return resultados_shazam

# Programa principal e a alínea 3 - Distribuição estatística
if __name__ == "__main__":
    ficheiros = ["lena.bmp", "ct1.bmp", "binaria.bmp", "saxriff.wav"]
    
    # Caminho dinâmico do Codespaces
    diretorio_atual = Path(__file__).resolve().parent
    base = diretorio_atual / "data"
    
    # Criação das pastas de destino para organizar as imagens geradas
    pasta_simples = diretorio_atual / "histogramas_simples"
    pasta_pares = diretorio_atual / "histogramas_pares"
    pasta_graficos = diretorio_atual / "graficos_shazam"
    
    pasta_simples.mkdir(exist_ok=True)
    pasta_pares.mkdir(exist_ok=True)
    pasta_graficos.mkdir(exist_ok=True)
    
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
            fonte = carregar_audio_mono(caminho)
            
        else:
            print(f"Formato de {ficheiro} não suportado")
            continue
            
        # 2. Calculando o Histograma (Passando a pasta simples)
        alfabeto, probs = alinea_1_histograma(fonte, ficheiro, pasta_simples)
        
        # 3. Calculando a Entropia
        entropia = alinea_2_entropia(probs)
        
        print(f"A entropia de {ficheiro}: é de {entropia:.6f} bits/símbolo")

        # Alinea 4 - Média de bits por símbolo
        variancia, media_bits_simbolo = alinea_4_huffman(fonte, probs)
        
        print(f"A média de bits por símbolo de {ficheiro} é de: {media_bits_simbolo:.6f} bits/símbolo")
        print(f"A variância de {ficheiro} é de: {variancia:.6f} bits^2")
        print(f"{'-'*70}")
        
        # Alínea 5 - Agrupamento (Passando a pasta pares)
        entropia_bloco, entropia_agrupada = alinea_5_agrupamento(fonte, ficheiro, pasta_pares)
        
        print(f"A entropia do bloco em pares de {ficheiro}: é de {entropia_bloco:.6f} bits/bloco")
        print(f"A entropia agrupada por símbolo original: é de {entropia_agrupada:.6f} bits/símbolo")
        print(f"{'-'*70}")
        
    # Simulação da alínea 6a
    query_sim = np.array([2, 6, 4, 10, 5, 9, 5, 8, 0, 8])
    target_sim = np.array([
        6, 8, 9, 7, 2, 4, 9, 9, 4, 9, 1, 4, 8, 0, 1, 2, 2, 6, 3, 2, 0, 7, 4, 9, 5, 
        4, 8, 5, 2, 7, 8, 0, 7, 4, 8, 5, 7, 4, 3, 2, 2, 7, 3, 5, 2, 7, 4, 9, 9, 6
    ])
    # De 0 a 10
    alfabeto_sim = np.arange(11)
    passo_sim = 1

    res_info_mutua = alinea_6a_informacao_mutua(query_sim, target_sim, alfabeto_sim, passo_sim)
    
    print("\n")
    print(f"{'-'*70}")
    print("Resultados da simulação da alínea 6a:")
    print(f"Número de janelas: {len(res_info_mutua)}")
    print(f"Informação mútua: {np.round(res_info_mutua, 4)}")
    print(f"{'-'*70}")
    
    
    # Encontrando o caminho da query principal que é a "saxriff.wav"
    caminho_query_principal = base / "saxriff.wav"
    # Calculando o tamanho da query e o passo de 1/4
    # Esses valores são úteis de saber para a alínea 6b e 6c
    query_audio_principal = carregar_audio_mono(caminho_query_principal)
    passo_principal = int(round(len(query_audio_principal) / 4))
    
    print("\n")
    print(f"{'-'*70}")
    print(f"Tamanho da query: {len(query_audio_principal)} amostras")
    print(f"Passo calculado (1/4): {passo_principal} amostras")
    print(f"{'-'*70}")

    # Chamando a alínea 6b passando a query, o passo e a pasta de gráficos
    info_mutua_target1, info_mutua_target2 = alinea_6b_localizacao(base, query_audio_principal, passo_principal, pasta_graficos)
    
    # Chamando a alínea 6c passando a query, o passo e a pasta de gráficos
    ranking = alinea_6c_shazam(base, query_audio_principal, passo_principal, pasta_graficos)