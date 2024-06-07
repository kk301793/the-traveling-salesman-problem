import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_random_points(num_points, min_coord, max_coord):
    points = []
    for _ in range(num_points):
        x = random.uniform(min_coord, max_coord)
        y = random.uniform(min_coord, max_coord)
        points.append((x, y))
    return points

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def total_distance(path, points):
    total = 0
    for i in range(len(path)):
        total += distance(points[path[i]], points[path[(i + 1) % len(path)]])

    total += distance(points[path[-1]], points[path[0]])
    return total

def initial_population(size, num_points):
    return [random.sample(range(num_points), num_points) for _ in range(size)]

def selection(population, scores, k=5):
    selected = [population[random.choice(range(len(population)))] for _ in range(k)]
    return min(selected, key=lambda x: scores[population.index(x)])

def crossover(parent1, parent2):
    start = random.randint(0, len(parent1) - 1)
    end = random.randint(0, len(parent1) - 1)
    if start > end:
        start, end = end, start

    child = [x for x in parent1[start:end]]
    remaining = [x for x in parent2 if x not in child]

    child.extend(remaining[:len(parent1) - len(child)])
    return child

def mutate(path, probability):
    if random.random() < probability:
        index1, index2 = random.sample(range(len(path) - 1), 2)
        path[index1], path[index2] = path[index2], path[index1]
    return path

def genetic_algorithm(points, population_size, generations, mutation_prob, crossover_prob):
    population = initial_population(population_size, len(points))
    best_path = []
    best_distance = float('inf')
    all_distances = []
    all_paths = []

    for gen in range(generations):
        scores = [total_distance(path, points) for path in population]
        new_population = [selection(population, scores) for _ in range(population_size)]
        for i in range(population_size):
            parent1 = new_population[i]
            parent2 = new_population[(i + 1) % population_size]
            if random.random() < crossover_prob: #decyzja o krzyżowaniu
                child = crossover(parent1, parent2)
            else:
                child = parent1
            child = mutate(child, mutation_prob)
            new_population[i] = child
            distance_child = total_distance(child, points)
            if distance_child < best_distance:
                best_distance = distance_child
                best_path = child
        population = new_population
        all_distances.append(best_distance)
        all_paths.append(best_path)

    return all_paths, all_distances
"""
# Parametry
num_points = 20
min_coord = 0
max_coord = 100
population_size = 100
generations = 100
mutation_prob = 0.01
crossover_probs = [0.1, 0.3, 0.5, 0.7, 0.9]  # wartości progu krzyżowania

# Generowanie punktów
points = generate_random_points(num_points, min_coord, max_coord)

# Przechowywanie wyników dla różnych wartości progu krzyżowania
results = {}

for cp in crossover_probs:
    all_paths, all_distances = genetic_algorithm(points, population_size, generations, mutation_prob, cp)
    results[cp] = (all_paths, all_distances)

# Rysowanie wykresów porównawczych
fig, ax = plt.subplots()
for cp in crossover_probs:
    _, all_distances = results[cp]
    ax.plot(range(1, generations + 1), all_distances, label=f'Crossover Prob: {cp}')

ax.set_title('Porównanie najlepszej odległości na przestrzeni pokoleń dla różnych prawdopodobieństw krzyżowania')
ax.set_xlabel('Generation')
ax.set_ylabel('Best Distance')
ax.legend()
ax.grid(True)
plt.show()
"""


# Parametry
num_points = 20
min_coord = 0
max_coord = 100
population_size = 100
generations = 200
mutation_prob = 0.01
crossover_prob = 0.7  # Prawdopodobieństwo krzyżowania

# Generowanie punktów
points = generate_random_points(num_points, min_coord, max_coord)

# Algorytm genetyczny
all_paths, all_distances = genetic_algorithm(points, population_size, generations, mutation_prob, crossover_prob)

# Inicjalizacja wykresu
fig, ax = plt.subplots()
scatter = ax.scatter(*zip(*points), color='blue')
line, = ax.plot([], [], color='red', linewidth=2)
ax.set_title(f'Generacja 0, Najlepszy Dystans: {all_distances[0]:.2f}')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.grid(True)

def update(frame):
    if frame < len(all_paths):
        best_path = all_paths[frame]
        best_distance = all_distances[frame]
        closed_path = best_path + [best_path[0]]
        line.set_data([points[i][0] for i in closed_path], [points[i][1] for i in closed_path])
        ax.set_title(f'Generacja {frame+1}, Najlepszy Dystans: {best_distance:.2f}')
    else:
        ani.event_source.stop()
        best_path = all_paths[-1]
        best_distance = all_distances[-1]
        closed_path = best_path + [best_path[0]]
        line.set_data([points[i][0] for i in closed_path], [points[i][1] for i in closed_path])
        ax.set_title(f'Wynik Końcowy, Najlepszy Dystans: {best_distance:.2f}')

# Animacja
ani = FuncAnimation(fig, update, frames=len(all_paths) + 10, interval=200)
plt.show()

# Wyświetlenie statystyk
plt.figure()
plt.plot(range(1, generations+1), all_distances)
plt.title('Zmiana Najlepszego Dystansu na Przestrzeni Generacji')
plt.xlabel('Generacja')
plt.ylabel('Najlepszy Dystans')
plt.grid(True)
plt.show()
