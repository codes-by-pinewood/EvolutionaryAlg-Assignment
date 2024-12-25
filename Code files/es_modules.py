import ioh
import numpy as np

class ES_Modules(object):
    def __init__(self, problem: ioh.problem.PBO, mu: int, lambda_: int, budget: int):
        self.problem = problem
        self.population_size = np.zeros(mu)
        self.population_fitness = None

    def intermediate_recombination(self, population, crossover_rate=0.5): # Slide 47 from 6 - ES Basics
        lambda_selection = len(population)
        children = []
        for i in range(lambda_selection):
            # print(population)
            p1 = np.random.choice(population)
            p2 = np.random.choice(population)
            while (p1 == p2):
                p2 = np.random.choice(population)
            
            if np.random.rand() > crossover_rate:
                children.append(p1)
                children.append(p2)
            else:
                child = np.sum([p1, p2]) / 2
                children.append(child)
        return children
    
    def mutation(self, children, mutation_rate=0.1): # Slide 8 from 6.2 - ES Self Adaptation
        mutated_children = []
        for i in range(len(children)):
            if np.random.rand() > mutation_rate:
                mutated_child = children[i] 
                noise = np.random.normal(0, 1, size=children[i].shape) 
                mutated_child = children[i] + noise 
                mutated_children.append(mutated_child)
            else: 
                mutated_child = children[i]
                mutated_children.append(mutated_child)
        return mutated_children
    
    def set_population(self, problem: ioh.problem.BBOB, pop_size:int):
        lower_bound = problem.bounds.lb[0]
        upper_bound = problem.bounds.ub[0]
        self.population_size = pop_size
        self.population = np.random.uniform(low=lower_bound, high=upper_bound, size=(self.population_size, problem.meta_data.n_variables))
        self.population_fitness = [(problem(ind), ind) for ind in self.population]
        return self.population, self.population_fitness