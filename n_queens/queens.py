import random
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

def gerar_solucao(n_rainhas):  # , coluna_fixa=None, linha_fixa=None):
    solucao = list(range(1, n_rainhas + 1))
    random.shuffle(solucao)
    return solucao

def funcobj(bestsol):
    n = len(bestsol)
    diagonal_pos = set()
    diagonal_neg = set()
    conflitos = 0
    
    for i in range(n):
        diag_pos = i - bestsol[i]
        diag_neg = i + bestsol[i]
        
        if diag_pos in diagonal_pos:
            conflitos += 1
        if diag_neg in diagonal_neg:
            conflitos += 1
            
        diagonal_pos.add(diag_pos)
        diagonal_neg.add(diag_neg)
    
    return conflitos

def gerar_vizinhos(solucao):
    vizinhos = []
    n = len(solucao)
    
    for i in range(n):
        for j in range(i + 1, n):
            vizinho = solucao[:]
            vizinho[i], vizinho[j] = vizinho[j], vizinho[i]
            vizinhos.append(vizinho)
    return vizinhos

def busca_tabu(n_rainhas, max_iter, tamanho_tabu):  # , coluna_fixa=None, linha_fixa=None):
    solucao_atual = gerar_solucao(n_rainhas)
    melhor_solucao = solucao_atual[:]
    melhor_fitness = funcobj(solucao_atual)
    
    lista_tabu = deque(maxlen=tamanho_tabu)
    lista_tabu.append(solucao_atual)
    
    iteracao = 0
    conflitos_por_iteracao = []
    
    while iteracao < max_iter and melhor_fitness > 0:
        vizinhos = gerar_vizinhos(solucao_atual)
        
        vizinho_aceito = None
        melhor_vizinho_fitness = float('inf')
        
        for vizinho in vizinhos:
            if vizinho not in lista_tabu:
                fitness_vizinho = funcobj(vizinho)
                if fitness_vizinho < melhor_vizinho_fitness:
                    melhor_vizinho_fitness = fitness_vizinho
                    vizinho_aceito = vizinho
        
        if vizinho_aceito:
            solucao_atual = vizinho_aceito
            lista_tabu.append(solucao_atual)
        
            if melhor_vizinho_fitness < melhor_fitness:
                melhor_solucao = solucao_atual[:]
                melhor_fitness = melhor_vizinho_fitness
        
        conflitos_por_iteracao.append(melhor_fitness)
        iteracao += 1
    
    return melhor_solucao, melhor_fitness, conflitos_por_iteracao

# Executando a busca tabu
n_rainhas = 100
max_iter = 5000
tamanho_tabu = 50

inicio = time.time()
melhor_solucao, melhor_fitness, conflitos_por_iteracao = busca_tabu(n_rainhas=n_rainhas, max_iter=max_iter, tamanho_tabu=tamanho_tabu)
fim = time.time()
tempo_execucao = fim - inicio

# Estatísticas
media_conflitos = np.mean(conflitos_por_iteracao)
mediana_conflitos = np.median(conflitos_por_iteracao)
desvio_padrao_conflitos = np.std(conflitos_por_iteracao)
melhor_caso = min(conflitos_por_iteracao)
pior_caso = max(conflitos_por_iteracao)

print("Melhor solução encontrada:", melhor_solucao)
print("Número de conflitos:", melhor_fitness)
print(f"Tempo de execução: {tempo_execucao:.2f} segundos")
print(f"Média de conflitos: {media_conflitos}")
print(f"Mediana de conflitos: {mediana_conflitos}")
print(f"Desvio padrão de conflitos: {desvio_padrao_conflitos}")
print(f"Melhor caso (menor número de conflitos): {melhor_caso}")
print(f"Pior caso (maior número de conflitos): {pior_caso}")

# Gerando o gráfico de convergência
plt.figure(figsize=(10, 6))
plt.plot(conflitos_por_iteracao, label="Conflitos por Iteração")
plt.xlabel("Iteração")
plt.ylabel("Número de Conflitos")
plt.title("Convergência do Número de Conflitos nas Iterações")
plt.legend()
plt.grid(True)
plt.show()
