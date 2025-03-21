import numpy as np
import random
import matplotlib.pyplot as plt

# Configurações do PSO
maxIter = 100          # Numero máximo de iterações
W_max, W_min = 0.9, 0.4  # Valores dinâmicos para a inércia
C1 = 1.5              # Constante cognitiva (aumentada para melhorar a exploração)
C2 = 1.5              # Constante social (aumentada para melhorar a convergência)
qtdParticulas = 40     # Quantidade de partículas

# Parametros do espaco de busca
dimensao = 5             # Dimensao do problema
limite_inferior = -5.12  # Limite inferior do espaco de busca
limite_superior = 5.12   # Limite superior do espaco de busca

def inicializar_particulas(qtdParticulas, dimensao, limite_inferior, limite_superior):
    posicoes = [
        [random.uniform(-0.5, 0.5) for _ in range(dimensao)]  # Inicialize proximo de -1/3
        for _ in range(qtdParticulas)
    ]
    velocidades = [
        [random.uniform(-0.1, 0.1) for _ in range(dimensao)]  # Velocidades iniciais pequenas
        for _ in range(qtdParticulas)
    ]
    return posicoes, velocidades

# Funcao Rastrigin com penalizacao para aproximar de -1/3
def fitness(x):
    rastrigin = 10 * len(x) + sum(xi ** 2 - 10 * np.cos(2 * np.pi * xi) for xi in x)
    penalizacao = sum((xi + 1/3) ** 2 for xi in x)  # Penalizacao para valores longe de -1/3
    return rastrigin + 1000 * penalizacao  # Peso da penalizacao ajustavel

# Execucao do PSO
def executar_pso():
    posicoes, velocidades = inicializar_particulas(qtdParticulas, dimensao, limite_inferior, limite_superior)
    pbest = posicoes[:]
    gbest = min(pbest, key=fitness)

    ftmedia = []  # Fitness media por geracao
    ftstd = []    # Desvio padrão da fitness por geracao
    ftgbest = []  # Fitness do melhor individuo por geracao

    for iteracao in range(maxIter):
        # Atualizar valor dinamico de W
        W = W_max - (W_max - W_min) * (iteracao / maxIter)

        fitness_values = [fitness(p) for p in posicoes]

        # Atualizar estatisticas por geracao
        ftmedia.append(np.mean(fitness_values))
        ftstd.append(np.std(fitness_values))
        ftgbest.append(fitness(gbest))

        for i in range(qtdParticulas):
            # Atualizar velocidade (inercia + cognitivo + social)
            velocidades[i] = [
                W * v + C1 * random.random() * (p - x) + C2 * random.random() * (g - x)
                for v, p, g, x in zip(velocidades[i], pbest[i], gbest, posicoes[i])
            ]

            # Atualizar posicao
            posicoes[i] = [x + v for x, v in zip(posicoes[i], velocidades[i])]

            # Restringir as particulas ao espaco de busca
            posicoes[i] = [
                min(max(x, limite_inferior), limite_superior) for x in posicoes[i]
            ]

            # Atualizar pbest e gbest
            if fitness(posicoes[i]) < fitness(pbest[i]):
                pbest[i] = posicoes[i]
            if fitness(posicoes[i]) < fitness(gbest):
                gbest = posicoes[i]

        # Debug: Imprimir progresso a cada 10 iteracoes
        if iteracao % 10 == 0 or iteracao == maxIter - 1:
            print(f"Iteração {iteracao}: Melhor Fitness = {fitness(gbest):.6f}, Melhor Posição = {gbest}")

    return ftmedia, ftstd, ftgbest, gbest

# Executar o PSO múltiplas vezes
resultados = []  # Armazenar o melhor fitness de cada execucao
n_execucoes = 20

for execucao in range(n_execucoes):
    ftmedia, ftstd, ftgbest, gbest = executar_pso()
    resultados.append(fitness(gbest))

# Estatisticas finais
media_resultados = np.mean(resultados)
std_resultados = np.std(resultados)

print(f"\nMédia do melhor fitness nas execuções: {media_resultados:.6f}")
print(f"Desvio padrão do melhor fitness nas execuções: {std_resultados:.6f}")
print(f"Melhor posição encontrada: {gbest}")
print(f"Diferença da solução para -1/3 em cada dimensão: {[x + 1/3 for x in gbest]}")
print(f"Fitness do melhor: {fitness(gbest):.6f}")

# Melhorando a visualização dos gráficos

# Gráfico 1: Evolução do Fitness Médio e Melhor Fitness ao longo das gerações
plt.figure(figsize=(8, 5))
plt.plot(ftmedia, label='Fitness Média', color='blue', linestyle='-', linewidth=2, marker='o', markersize=4)
plt.plot(ftgbest, label='Melhor Fitness', color='red', linestyle='--', linewidth=2, marker='s', markersize=4)
plt.xlabel('Gerações', fontsize=12)
plt.ylabel('Fitness', fontsize=12)
plt.title('Evolução do Fitness - Função Rastrigin', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Gráfico 2: Variação do Desvio Padrão ao longo das Gerações
plt.figure(figsize=(8, 5))
plt.plot(ftstd, label='Desvio Padrão do Fitness', color='green', linestyle='-', linewidth=2, marker='^', markersize=4)
plt.xlabel('Gerações', fontsize=12)
plt.ylabel('Desvio Padrão', fontsize=12)
plt.title('Variação do Desvio Padrão do Fitness', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Gráfico 3: Representação 3D da Função Rastrigin
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(limite_inferior, limite_superior, 100)
y = np.linspace(limite_inferior, limite_superior, 100)
x, y = np.meshgrid(x, y)
z = 10 * 2 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)

ax.plot_surface(x, y, z, cmap='plasma', alpha=0.8, edgecolor='k', linewidth=0.3)
ax.scatter(gbest[0], gbest[1], fitness(gbest), color='r', s=50, label='Melhor Solução')
ax.set_xlabel("X", fontsize=12)
ax.set_ylabel("Y", fontsize=12)
ax.set_zlabel("Fitness", fontsize=12)
ax.set_title("Função Rastrigin - Representação 3D", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.show()
