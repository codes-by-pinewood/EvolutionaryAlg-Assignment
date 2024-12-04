from typing import Tuple 
import numpy as np
import ioh
from ioh import get_problem, logger, ProblemClass


budget = 5000

def n_crossover(p1, p2, size, n=2, crossover_rate = 0.5):

    if np.random.rand() > crossover_rate:
        # If not, just return the parents as children (no crossover)
        return p1, p2

    split_positions = sorted(np.random.choice(range(size), n, replace=False))
    c1 = []
    c2 = []
    subarrays_p1 = np.split(p1, split_positions)
    subarrays_p2 = np.split(p2, split_positions)
    
    for i in range(len(subarrays_p1)):
        if (i % 2 == 0):
            for j in range(len(subarrays_p1[i])):
                c1.append(subarrays_p1[i][j])
                c2.append(subarrays_p2[i][j])
        else: 
            for j in range(len(subarrays_p2[i])):
                c1.append(subarrays_p2[i][j])
                c2.append(subarrays_p1[i][j])
    return c1,c2

def mating_selection(population, pop_fitness):
    # Using the tournament selection
    # select_parent = []
    # for i in range(len(parent)) :
    #     pre_select = np.random.choice(len(parent_f),tournament_k,replace = False)
    #     max_f = sys.float_info.min
    #     for p in pre_select:
    #         if parent_f[p] > max_f:
    #             index = p
    #             max_f = parent_f[p]
    #     select_parent.append(parent[index].copy())
    # return select_parent

    # Using the proportional selection

    # Plusing 0.001 to avoid dividing 0
    f_min = min(pop_fitness)
    f_sum = sum(pop_fitness) - (f_min - 0.001) * len(pop_fitness)
    
    rw = [(pop_fitness[0] - f_min + 0.001)/f_sum]
    for i in range(1,len(pop_fitness)):
        rw.append(rw[i-1] + (pop_fitness[i] - f_min + 0.001) / f_sum)
    
    select_parent = []
    for i in range(2) :
        r = np.random.uniform(0,1)
        index = 0
        # print(rw,r)
        while(r > rw[index]) :
            index = index + 1
        
        select_parent.append(population[index].copy())
    return select_parent

def mutate(c, mutation_rate):
    ind_length = len(c)
    for j in range(ind_length):  
        if np.random.uniform(0, 1) < mutation_rate:
            swap_idx = np.random.randint(0, ind_length)
            c[j], c[swap_idx] = c[swap_idx], c[j]  # Swap mutation

    return c

def s4018907_s4168216_GA(problem: ioh.problem.PBO, init_pop_size: int, mutation_rate: float, crossover_rate: float, budget: int) -> None:
    # initial_pop = ... make sure you randomly create the first population
    #initial_pop_size = pop_size
    # mutation_rate = 
    #crossover_rate = 0.5
    population = []
    pop_fitness = []

    for i in range(init_pop_size):
        # Initialization
        population.append(np.random.randint(2, size = problem.meta_data.n_variables))
        pop_fitness.append(problem(population[i]))

    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:
        parents = mating_selection(population, pop_fitness)
        p1 = parents[0]
        p2 = parents[1]
        c1, c2 = n_crossover(p1, p2, problem.meta_data.n_variables, crossover_rate = crossover_rate)
        mutated_c1 = mutate(c1, mutation_rate)
        mutated_c2 = mutate(c2, mutation_rate)
        f1 = problem(mutated_c1)
        f2 = problem(mutated_c2)
        population.append(mutated_c1)
        population.append(mutated_c2)



def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    # create the LABS problem and the data logger
    # F18, _logger = create_problem(dimension=50, fid=18)
    # for run in range(20): 
    #     studentnumber1_studentnumber2_GA(F18)
    #     F18.reset() # it is necessary to reset the problem after each independent run
    # _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20): 
        s4018907_s4168216_GA(F23)
        F23.reset()
    _logger.close()