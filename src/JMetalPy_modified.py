import copy
import random
import threading
import time
from typing import TypeVar, List
import functools
import numpy

from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.operator import Mutation
from jmetal.core.problem import Problem
from jmetal.util.comparator import DominanceComparator
from jmetal.core.solution import Solution
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.constraint_handling import overall_constraint_violation_degree
S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: simulated_annealing
   :platform: Unix, Windows
   :synopsis: Implementation of Simulated Annealing.
.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class SimulatedAnnealing2(Algorithm[S, R], threading.Thread):

    def __init__(self,
                 problem: Problem[S],
                 mutation: Mutation,
                 termination_criterion: TerminationCriterion,
                 solution_generator: Generator = store.default_generator):
        super(SimulatedAnnealing2, self).__init__()
        self.problem = problem
        self.mutation = mutation
        self.termination_criterion = termination_criterion
        self.solution_generator = solution_generator
        self.observable.register(termination_criterion)
        self.temperature = 1.0
        self.minimum_temperature = 0.000001
        self.alpha = 0.95
        self.counter = 0
        self.comparator = DominanceComparator()
        
        
    def create_initial_solutions(self) -> List[S]:
        return [self.solution_generator.new(self.problem)]

    def evaluate(self, solutions: List[S]) -> List[S]:
        return [self.problem.evaluate(solutions[0])]

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def init_progress(self) -> None:
        self.evaluations = 0

    def step(self) -> None:
        mutated_solution = copy.deepcopy(self.solutions[0])
        mutated_solution: Solution = self.mutation.execute(mutated_solution)
        mutated_solution = self.evaluate([mutated_solution])[0]
        
        acceptance_probability = self.compute_acceptance_probability(
            self.solutions[0].objectives[0],
            mutated_solution.objectives[0],
            self.temperature)

        if self.comparator.compare(mutated_solution,self.solutions[0]) == 1:
            acceptance_probability *= 0.001
        elif self.comparator.compare(mutated_solution,self.solutions[0]) == 0:
            acceptance_probability *= 0.5
            
        if acceptance_probability > random.random() :
            self.solutions[0] = mutated_solution
        #self.solutions.sort(key=functools.cmp_to_key(self.comparator.compare),reverse=True)
        self.temperature *= self.alpha
      
        
    def compute_acceptance_probability(self, current: float, new: float, temperature: float) -> float:
        if new < current:
            return 1.0
        else:
            t = temperature if temperature > self.minimum_temperature else self.minimum_temperature
            value = (new - current) / t
            return numpy.exp(-1.0 * value)

    def update_progress(self) -> None:
        self.evaluations += 1

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def get_observable_data(self) -> dict:
        ctime = time.time() - self.start_computing_time
        return {'PROBLEM': self.problem, 'EVALUATIONS': self.evaluations, 'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': ctime}

    def get_result(self) -> R:
        return self.solutions[0]

    def get_name(self) -> str:
        return 'Simulated Annealing'
    
    
from typing import TypeVar, List

from jmetal.config import store
from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: genetic_algorithm
   :platform: Unix, Windows
   :synopsis: Implementation of Genetic Algorithms.
.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class GeneticAlgorithm2(EvolutionaryAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator):
        super(GeneticAlgorithm2, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size)
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.mating_pool_size = \
            self.offspring_population_size * \
            self.crossover_operator.get_number_of_parents() // self.crossover_operator.get_number_of_children()

        if self.mating_pool_size < self.crossover_operator.get_number_of_children():
            self.mating_pool_size = self.crossover_operator.get_number_of_children()

    def create_initial_solutions(self) -> List[S]:
        return [self.population_generator.new(self.problem)
                for _ in range(self.population_size)]

    def evaluate(self, population: List[S]):
        return self.population_evaluator.evaluate(population, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def selection(self, population: List[S]):
        mating_population = []

        for i in range(self.mating_pool_size):
            solution = self.selection_operator.execute(population)
            mating_population.append(solution)

        return mating_population

    def reproduction(self, mating_population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        if len(mating_population) % number_of_parents_to_combine != 0:
            raise Exception('Wrong number of parents')

        offspring_population = []
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(mating_population[i + j])

            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:
                self.mutation_operator.execute(solution)
                offspring_population.append(solution)
                if len(offspring_population) >= self.offspring_population_size:
                    break

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        population.extend(offspring_population)
        
        population.sort(key=lambda s: (s.objectives[0],overall_constraint_violation_degree(s)))
        
        return population[:self.population_size]

    def get_result(self) -> R:
        return self.solutions[0]

    def get_name(self) -> str:
        return 'Genetic algorithm'