import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import choices, randint, randrange, random


# function to calculate the mean standard error. 
# mean((predicted - actual outcome)^2)
def MSE(w, x, y):
    w = np.array(w)  
    prediction = x @ w
    return np.mean((prediction - y) ** 2)

# read in credit card data and change all values to integers
def load_data(filename="CreditCard.csv"):
    df = pd.read_csv(filename)

    # encode string values to integers
    df["Gender"] = df["Gender"].map({"M": 1, "F": 0})
    df["CarOwner"] = df["CarOwner"].map({"Y": 1, "N": 0})
    df["PropertyOwner"] = df["PropertyOwner"].map({"Y": 1, "N": 0})

    # Features (X) and labels (y)
    X = df[["Gender", "CarOwner", "PropertyOwner", "#Children", "WorkPhone", "Email_ID"]].values
    y = df["CreditApprove"].values

    return X, y



Genome = list
Population = list

# return a genome (array of 1s or -1s randomly) ex. [1,1,1,-1]
def generate_genome(length: int):
    return choices([-1, 1], k=length)

# return a population, array of genomes for the length of size
def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]

# split both genonmes and swap the tails of each
def single_point_crossover(a: Genome, b: Genome):
    length = len(a)

    # cant take the first or last valeus
    p = randint(1, length - 1)
    # child 1 = a (start to p) + b (p to end)
    # child 2 flipped
    return a[0:p] + b[p:], b[0:p] + a[p:]

def mutation(genome: Genome, num: int = 1, probability: float = 0.2) -> Genome:
    # for how many elements you want to consider for mutation (default 1)
    for _ in range(num):
        index = randrange(len(genome)) #randomly select an index
        if random() < probability:
            genome[index] *= -1
    return genome

def fitness_func(genome: Genome, x, y):
    error = MSE(genome, x, y)
    return 1 / (1 + error)   # maximize fitness, minimize error


def run_evolution(x, y, population_size, generations):
    genome_length = x.shape[1]
    population = generate_population(population_size, genome_length)

    history = []
    best_genome = None
    best_error = float("inf")

    for gen in range(generations):
        # Sort by fitness putting highest fitness at the start of the array
        population = sorted(population, key=lambda g: fitness_func(g, x, y), reverse=True)

        # Track best
        current_best = population[0]
        current_error = MSE(current_best, x, y)
        history.append(current_error)

        if current_error < best_error:
            best_error = current_error
            best_genome = current_best

        # keep top 2
        next_gen = population[0:2]

        # Breed rest (each loop takes 2 children and already took the top 2 so -1)
        for _ in range(int(len(population) / 2) - 1):
            # select 2 at random from population
            parents = choices(population, k=2)

            # combine the two
            offspring_a, offspring_b = single_point_crossover(parents[0], parents[1])
            next_gen.append(mutation(offspring_a))
            next_gen.append(mutation(offspring_b))

        population = next_gen

    return best_genome, best_error, history



if __name__ == "__main__":
    x, y = load_data()

    Ws, erW, history = run_evolution(x, y, population_size=20, generations=50)

    plt.plot(range(len(history)), history, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Genetic Algorithm Error Over Time')

    print("Best weights:", Ws)
    print("Final error:", erW)
    plt.show()
