import numpy as np
import ioh
from ioh import get_problem, logger, ProblemClass
from es_modules import ES_Modules
from typing import List, Tuple

# Setting Budget and Dimensions
budget = 50000
dimension = 10

def create_problem(dimension: int, fid: int, name:str) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.BBOB)
    print("creating problem")
    l = logger.Analyzer(
        root="data", 
        folder_name="run", 
        algorithm_name= name, 
        algorithm_info="Practical assignment of the EA course",
    )
    problem.attach_logger(l)
    return problem, l

# ES Algorithm
def s4018907_s4168216_ES(problem: ioh.problem.BBOB, pop_size: int, mutation_rate: float, crossover_rate: float, budget: int) -> None:
    mutated_fitness = []
    es_module = ES_Modules(problem, mu=pop_size, lambda_=pop_size, budget=budget)
    population, pop_fitness = es_module.set_population(problem, pop_size)
    while problem.state.evaluations < budget:
        children = es_module.intermediate_recombination(population, crossover_rate)
        mutated_children = es_module.mutation(children, mutation_rate)
        for child in mutated_children:
            muta_child = problem(child)
            mutated_fitness.append((muta_child, child))
        pop_fitness.extend(mutated_fitness)
        pop_fitness.sort(key=lambda x: x[0])
        pop_fitness = pop_fitness[:pop_size]
        #print("Best fitness: ", pop_fitness)
        population = np.array([ind[1] for ind in pop_fitness])
        best_solution = pop_fitness[0]
    return best_solution[0]

# Tuning Hyperparameters
hyperparameter_space = {
    "population_size": [10, 20, 50],
    "mutation_rate": [0.01, 0.05, 0.1],
    "crossover_rate": [0.5, 0.7, 0.9] #0.5
}

# Hyperparameter tuning function
def tune_hyperparameters() -> List:
    # You should decide/engineer the `score` youself, which is the tuning objective
    best_score = float('-inf')
    print("Running the problem")
    best_params = None
    
    for pop_size in hyperparameter_space['population_size']:
        for mutation_rate in hyperparameter_space['mutation_rate']:
            for crossover_rate in hyperparameter_space['crossover_rate']:
                F23, _logger = create_problem(dimension=10, fid=23, name=f"pop_size={pop_size}_mr={mutation_rate}_cr={crossover_rate}")
                print("running hyperparameter tuning for ", pop_size, mutation_rate, crossover_rate)
                score_f23 = s4018907_s4168216_ES(F23, pop_size, mutation_rate, crossover_rate, budget)
                F23.reset()
                print("F23 score is", score_f23)
                if score_f23 > best_score:
                    best_score = score_f23
                    print("best score for F23: ", best_score)
                    best_params = [pop_size, mutation_rate, crossover_rate]
    return best_params


if __name__ == "__main__":
    # Hyperparameter tuning to determine the best parameters for both problems
    tuning_results = tune_hyperparameters()

    if tuning_results is not None:
        # Corrected unpacking of tuning_results to ensure no TypeError for non-subscriptable types
        if isinstance(tuning_results, list) and len(tuning_results) == 3:
            population_size, mutation_rate, crossover_rate = tuning_results[0], tuning_results[1], tuning_results[2]
            print(population_size)
            print(mutation_rate)
            print(crossover_rate)
        else:
            print("Unexpected format of tuning results:", tuning_results)
    else:
        print("No optimal parameters could be determined.")