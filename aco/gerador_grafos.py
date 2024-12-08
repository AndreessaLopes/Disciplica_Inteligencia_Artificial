import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

def gerar_grafo(file_name='exemplo_sala_novo.csv', num_nodes=10, num_arcs=30):
    # Configurações iniciais
    max_connections = (num_nodes * (num_nodes - 1)) // 2  # Máximo de conexões possíveis

    # Cria uma matriz para rastrear as conexões entre os nós (de 1 até num_nodes)
    connections = [[0] * (num_nodes + 1) for _ in range(num_nodes + 1)]

    # Criação inicial do DataFrame para armazenar origem, destino e custo
    df = pd.DataFrame(columns=['origem', 'destino', 'custo'])

    # Adiciona as arestas de forma aleatória
    for _ in range(num_arcs):
        while True:
            origem = random.randint(1, num_nodes)  # Nós começam de 1
            destino = random.randint(1, num_nodes)

            if origem != destino and connections[origem][destino] == 0:
                custo = round(random.uniform(1, 10), 2)
                df = pd.concat([df, pd.DataFrame({'origem': [origem], 'destino': [destino], 'custo': [custo]})], ignore_index=True)
                connections[origem][destino] = 1
                connections[destino][origem] = 1
                break

    # Assegura que cada nó tenha pelo menos duas conexões
    for node in range(1, num_nodes + 1):
        if sum(connections[node]) < 2:
            while True:
                dest_node = random.randint(1, num_nodes)
                if node != dest_node and connections[node][dest_node] == 0:
                    custo = round(random.uniform(1, 10), 2)
                    df = pd.concat([df, pd.DataFrame({'origem': [node], 'destino': [dest_node], 'custo': [custo]})], ignore_index=True)
                    connections[node][dest_node] = 1
                    connections[dest_node][node] = 1
                    break

    # Salva o DataFrame como CSV
    df.to_csv(file_name, index=False)

    # Criação do grafo usando NetworkX
    graph = nx.from_pandas_edgelist(df, 'origem', 'destino', edge_attr='custo', create_using=nx.Graph())

    # Configura a cor dos nós
    vinicial, vfinal = 1, num_nodes
    node_colors = ['orange' if node in [vinicial, vfinal] else 'white' for node in graph.nodes()]

    # Plotagem do grafo
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color='black', node_size=700)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={(u, v): d['custo'] for u, v, d in graph.edges(data=True)})
    plt.show()
