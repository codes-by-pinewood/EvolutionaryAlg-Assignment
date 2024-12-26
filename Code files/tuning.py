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
    best_score = float('-inf')
    best_params = None
    for pop_size in hyperparameter_space['population_size']:
        for mutation_rate in hyperparameter_space['mutation_rate']:
            for crossover_rate in hyperparameter_space['crossover_rate']:

                F18, _logger18 = create_problem(dimension=50, fid=18, name=f"pop_size={pop_size}_mr={mutation_rate}_cr={crossover_rate}")
                F23, _logger23 = create_problem(dimension=49, fid=23, name=f"pop_size={pop_size}_mr={mutation_rate}_cr={crossover_rate}")

                print("running hyperparameter tuning for ", pop_size, mutation_rate, crossover_rate)

                print("Running GA on F18")
                score_f18_1, score_f18_2 = s4018907_s4168216_GA(F18, pop_size, mutation_rate, crossover_rate, budget)
                score_f18 = np.mean([score_f18_1, score_f18_2])
                print("score for F18: ", score_f18)

                F18.reset()

                if score_f18 > best_score:
                    best_score = score_f18
                    print("best score for F18: ", best_score)
                    best_params = [pop_size, mutation_rate, crossover_rate]

                print("Running GA on F23")
                score_f23_1, score_f23_2 = s4018907_s4168216_GA(F23, pop_size, mutation_rate, crossover_rate, budget)
                score_f23 = np.mean([score_f23_1, score_f23_2])
                print("score for F23: ", score_f23)

                F23.reset()

                if score_f23 > best_score:
                    best_score = score_f23
                    print("best score for F23: ", best_score)
                    best_params = [pop_size, mutation_rate, crossover_rate]

    return best_params


if __name__ == "__main__":

    # Call the hyperparameter tuning function
    best_params = tune_hyperparameters()
    population_size, mutation_rate, crossover_rate = best_params
    print(population_size)
    print(mutation_rate)
    print(crossover_rate)