from typing import List
import numpy as np
from ioh import get_problem, logger, ProblemClass
from GA import s4018907_s4168216_GA, create_problem

budget = 1000000
hyperparameter_space = {
    "population_size": [50, 100, 200],
    "mutation_rate": [0.01, 0.05, 0.1],
    "crossover_rate": [0.5, 0.7, 0.9]
}

def tune_hyperparameters() -> List:
    # You should decide/engineer the `score` youself, which is the tuning objective
    best_score = float('inf')
    best_params = None
    # create the LABS problem and the data logger
    F18, _logger = create_problem(dimension=50, fid=18)
    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)

    for pop_size in hyperparameter_space['population_size']:
        for mutation_rate in hyperparameter_space['mutation_rate']:
            for crossover_rate in hyperparameter_space['crossover_rate']:
                score_f18 = s4018907_s4168216_GA(F18, pop_size, mutation_rate, crossover_rate, budget)
                print(score_f18)
                F18.reset()
                if score_f18 < best_score:
                    best_score = score_f18
                    best_params = [pop_size, mutation_rate, crossover_rate]
                score_f23 = s4018907_s4168216_GA(F23, pop_size, mutation_rate, crossover_rate, budget)
                print(score_f23)
                F23.reset()
                if score_f23 < best_score:
                    best_score = score_f23
                    best_params = [pop_size, mutation_rate, crossover_rate]
    return best_params


if __name__ == "__main__":
    # Hyperparameter tuning to determine the best parameters for both problems
    population_size, mutation_rate, crossover_rate = tune_hyperparameters()
    print(population_size)
    print(mutation_rate)
    print(crossover_rate)
