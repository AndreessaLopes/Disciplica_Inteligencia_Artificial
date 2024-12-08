import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Configurações do ACO
FEROMONIO_INICIAL = 0.01
TAXA_EVAPORACAO = 0.07
MAX_ITER = 500
N_ANTS = 40
ALPHA = 1.0
BETA = 8.0

def carregar_grafo(arquivo):
    df = pd.read_csv(arquivo, sep='\t')
    df.columns = df.columns.str.strip()
    print("Colunas do DataFrame carregado:", df.columns)
    G = nx.from_pandas_edgelist(df, source='origem', target='destino', edge_attr='custo', create_using=nx.Graph())
    return G, df

def inicializar_feromonio(G):
    return {tuple(sorted(edge)): FEROMONIO_INICIAL for edge in G.edges}

def calcular_visibilidade(G):
    return {tuple(sorted(edge)): G.edges[edge]['custo'] for edge in G.edges}

def atualizar_feromonio(G, feromonio, caminhos, custos):
    for edge in feromonio:
        feromonio[edge] *= (1 - TAXA_EVAPORACAO)

    for caminho, custo in zip(caminhos, custos):
        for i in range(len(caminho) - 1):
            edge = tuple(sorted((caminho[i], caminho[i + 1])))
            feromonio[edge] += custo
    return feromonio

def calcular_probabilidades(G, feromonio, visibilidade, no_atual, caminho):
    visitados = set(caminho)
    vizinhos = [v for v in G.neighbors(no_atual) if v not in visitados]
    if not vizinhos:
        return None, None

    probabilidades = []
    for vizinho in vizinhos:
        edge = tuple(sorted((no_atual, vizinho)))
        p = (feromonio[edge] ** ALPHA) * (visibilidade[edge] ** BETA)
        probabilidades.append(p)

    soma_probabilidades = sum(probabilidades)
    probabilidades = [p / soma_probabilidades for p in probabilidades]
    return vizinhos, probabilidades

def executar_aco(arquivo, nodo_final):
    G, df = carregar_grafo(arquivo)
    feromonio = inicializar_feromonio(G)
    visibilidade = calcular_visibilidade(G)
    melhor_caminho_global, maior_custo_global = None, float('-inf')

    for iteracao in range(MAX_ITER):
        caminhos, custos = [], []
        for _ in range(N_ANTS):
            caminho, v_atual = [1], 1
            while v_atual != nodo_final:
                vizinhos, probabilidades = calcular_probabilidades(G, feromonio, visibilidade, v_atual, caminho)
                if not vizinhos:
                    caminho = []
                    break
                v_atual = np.random.choice(vizinhos, p=probabilidades)
                caminho.append(v_atual)

            if len(caminho) > 1:
                custo = sum(G.edges[tuple(sorted((caminho[i], caminho[i + 1])))]["custo"] for i in range(len(caminho) - 1))
                caminhos.append(caminho)
                custos.append(custo)

        feromonio = atualizar_feromonio(G, feromonio, caminhos, custos)

        #menor custo: inverte o sinal no if
        for caminho, custo in zip(caminhos, custos):
            if custo > maior_custo_global:
                melhor_caminho_global, maior_custo_global = caminho, custo

    return melhor_caminho_global, maior_custo_global

def plotar_grafo(G, caminho):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'custo')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    if caminho:
        path_edges = list(zip(caminho, caminho[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)
    plt.show()

if __name__ == "__main__":
    base = input("Escolha a base (A, B, C, D): ").strip().upper()
    if base == "A":
        arquivo = "exemplo_slides.csv"
        nodo_final = 4
    elif base == "B":
        arquivo = "grafo1.csv"
        nodo_final = 12
    elif base == "C":
        arquivo = "grafo2.csv"
        nodo_final = 20
    elif base == "D":
        arquivo = "grafo3.csv"
        nodo_final = 100
    else:
        print("Base inválida!")
        exit()

    pior_caminho, custo = executar_aco(arquivo, nodo_final)
    print(f"Base escolhida: {base}")
    print("Pior caminho encontrado:", pior_caminho)
    print("Custo do pior caminho:", custo)

    G, _ = carregar_grafo(arquivo)
    plotar_grafo(G, pior_caminho)
