from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII
from jmetal.algorithm.multiobjective.nsgaiii import *
from jmetal.config import store
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import Comparator, MultiComparator
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.core.solution import Solution
import random
import copy 

class UNSGAIII(NSGAIII):
    
    def __init__(self,
                 reference_directions,
                 problem: Problem,
                 mutation: Mutation,
                 crossover: Crossover,
                 population_size: int = None,
                 selection: Selection = BinaryTournamentSelection(
                     MultiComparator([FastNonDominatedRanking.get_comparator(),
                                      CrowdingDistance.get_comparator()])),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator):
        self.reference_directions = reference_directions.compute()

        if not population_size:
            population_size = len(self.reference_directions)
        if self.reference_directions.shape[1] != problem.number_of_objectives:
            raise Exception('Dimensionality of reference points must be equal to the number of objectives')

        super(NSGAIII, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            dominance_comparator=dominance_comparator
        )

        self.extreme_points = None
        self.ideal_point = np.full(self.problem.number_of_objectives, np.inf)
        self.worst_point = np.full(self.problem.number_of_objectives, -np.inf)
        
    def niching_parent_selection(self,parent1_index, parent2_index, population ,fronts,niche_of_individuals,dist_to_niche):
        chosen_parent = None
        def subfinder(mylist, value):
            for index, sublist in enumerate(mylist):
                if value in sublist:
                    return index
            return None
                
                
        if niche_of_individuals[parent1_index] == niche_of_individuals[parent2_index]:
            
                parent1_front = subfinder(fronts,parent1_index)
                parent2_front = subfinder(fronts,parent2_index)
                
                if parent1_front < parent2_front:
                    chosen_parent = population[parent1_index]
                elif parent2_front < parent1_front:
                    chosen_parent = population[parent2_index]
                    
                else:
                    if dist_to_niche[parent1_index] < dist_to_niche[parent2_index]:
                        chosen_parent = population[parent1_index]
                    else:
                        chosen_parent = population[parent2_index]
        else:
            chosen_parent = population[random.sample([parent1_index,parent2_index], k = 1)][0]
        return chosen_parent
                
    def niching_selection(self, population: List[Solution]) -> List[Solution]:
        # find or usually update the new ideal point - from feasible solutions
        # note that we are assuming minimization here!
        F = np.array([s.objectives for s in population])
        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

        # calculate the fronts of the population
        ranking = FastNonDominatedRanking(self.dominance_comparator)
        ranking.compute_ranking(population, k=self.population_size)

        fronts, non_dominated = ranking.ranked_sublists, ranking.get_subfront(0)

        # find the extreme points for normalization
        self.extreme_points = get_extreme_points(F=np.array([s.objectives for s in non_dominated]),
                                                 n_objs=self.problem.number_of_objectives,
                                                 ideal_point=self.ideal_point,
                                                 extreme_points=self.extreme_points)

        # find the intercepts for normalization and do backup if gaussian elimination fails
        worst_of_population = np.max(F, axis=0)
        worst_of_front = np.max(np.array([s.objectives for s in non_dominated]), axis=0)

        nadir_point = get_nadir_point(extreme_points=self.extreme_points,
                                      ideal_point=self.ideal_point,
                                      worst_point=self.worst_point,
                                      worst_of_population=worst_of_population,
                                      worst_of_front=worst_of_front)

        #  consider only the population until we come to the splitting front
        pop = np.concatenate(ranking.ranked_sublists)
        F = np.array([s.objectives for s in pop])

        # update the front indices for the current population
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1
        last_front = np.array(fronts[-1])

        # associate individuals to niches
        niche_of_individuals, dist_to_niche = associate_to_niches(F=F,
                                                                  niches=self.reference_directions,
                                                                  ideal_point=self.ideal_point,
                                                                  nadir_point=nadir_point)
    
        selected_population = []
        
        current_population = copy.copy(pop)
        current_population = np.append(current_population,random.sample(list(pop),k=len(pop)))
                
        for index in range(0,len(current_population),2):
            ind = index % len(pop)
            
            selected_parent = self.niching_parent_selection(ind,ind+1,pop,fronts,niche_of_individuals,dist_to_niche)
            selected_population.append(selected_parent)
            
        return selected_population
        
    
    def step(self):
        mating_population = self.selection(self.solutions)
        mating_population = self.niching_selection(mating_population)
        offspring_population = self.reproduction(mating_population)
        offspring_population = self.evaluate(offspring_population)

        self.solutions = self.replacement(self.solutions, offspring_population)
    
    def get_name(self) -> str:
        return 'U-NSGAIII'
        
from typing import TypeVar, List

import numpy as np

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Mutation, Crossover
from jmetal.core.problem import Problem
from jmetal.core.quality_indicator import EpsilonIndicator
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import SolutionAttributeComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')


class Adaptive_IBEA(GeneticAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 kappa: float,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator):
        """  Epsilon IBEA implementation as described in

        * Zitzler, Eckart, and Simon Künzli. "Indicator-based selection in multiobjective search."
        In International Conference on Parallel Problem Solving from Nature, pp. 832-842. Springer,
        Berlin, Heidelberg, 2004.

        https://link.springer.com/chapter/10.1007/978-3-540-30217-9_84

        IBEA is a genetic algorithm (GA), i.e. it belongs to the evolutionary algorithms (EAs)
        family. The multi-objective search in IBEA is guided by a fitness associated to every solution,
        which is in turn controlled by a binary quality indicator. This implementation uses the so-called
        additive epsilon indicator, along with a binary tournament mating selector.

        :param problem: The problem to solve.
        :param population_size: Size of the population.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        :param kappa: Weight in the fitness computation.
        """

        selection = BinaryTournamentSelection(
            comparator=SolutionAttributeComparator(key='fitness', lowest_is_best=False))
        self.kappa = kappa

        super(Adaptive_IBEA, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator
        )

    def compute_fitness_values(self, population: List[S], kappa: float) -> List[S]:
        population_objectives= np.array([p.objectives for p in population])
        objectives_max = np.max(population_objectives, axis= 0)
        objectives_min = np.min(population_objectives, axis= 0)
        
        
        for i in range(len(population)):
            population[i].attributes['fitness'] = 0
            adapted_objective_value_i = (population[i].objectives - objectives_min)/(objectives_max - objectives_min)
            
            
            c_obj = 0

            for j in range(len(population)):
                if j != i:
                    adapted_objective_value_j = (population[j].objectives - objectives_min)/(objectives_max - objectives_min)
                    
                    indicator_value_obj = abs(np.exp(
                        -EpsilonIndicator([adapted_objective_value_i]).compute([adapted_objective_value_j]) / self.kappa))
                    
                    if indicator_value_obj > c_obj:
                        c_obj = indicator_value_obj     

            for j in range(len(population)):
                if j != i:
                    adapted_objective_value_j = (population[j].objectives - objectives_min)/(objectives_max - objectives_min)
                    
                    population[i].attributes['fitness'] += -np.exp(
                        -EpsilonIndicator([adapted_objective_value_i]).compute([adapted_objective_value_j]) / (c_obj * self.kappa))
                    
        return population

    def create_initial_solutions(self) -> List[S]:
        population = [self.population_generator.new(self.problem) for _ in range(self.population_size)]
        population = self.compute_fitness_values(population, self.kappa)

        return population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        join_population = population + offspring_population
        join_population_size = len(join_population)
        join_population = self.compute_fitness_values(join_population, self.kappa)

        while join_population_size > self.population_size:
            current_fitnesses = [individual.attributes['fitness'] for individual in join_population]
            index_worst = current_fitnesses.index(min(current_fitnesses))
            
            population_objectives= np.array([p.objectives for p in join_population])
    
            objectives_max = np.max(population_objectives, axis= 0)
            objectives_min = np.min(population_objectives, axis= 0)
            
            
            adapted_index_worst = (join_population[index_worst].objectives - objectives_min)/(objectives_max - objectives_min)
                    
            population_constraints = np.array([p.constraints for p in join_population])
            constraints_max = np.max(population_constraints, axis= 0)
            constraints_min = np.min(population_constraints, axis= 0)
            

            #adapted_constraint_worst = np.abs((join_population[index_worst].constraints - constraints_min)/(constraints_max - constraints_min))
            #adapted_constraint_worst = np.nan_to_num(adapted_constraint_worst)
            
            #test = np.zeros(len(adapted_constraint_worst))
            
            c = 0
            for i in range(join_population_size):
                adapted_objective_value_i = (join_population[i].objectives - objectives_min)/(objectives_max - objectives_min)
                
                indicator_value = abs(np.exp(
                    -EpsilonIndicator([adapted_objective_value_i]).compute([adapted_index_worst]) / self.kappa))
                if indicator_value > c:
                    c = indicator_value  
            
            
            
            for i in range(join_population_size):
                adapted_objective_value_i = (join_population[i].objectives - objectives_min)/(objectives_max - objectives_min)
                adapted_constraint_value_i = np.abs((join_population[i].constraints - constraints_min)/(constraints_max - constraints_min))
                adapted_constraint_value_i = np.nan_to_num(adapted_constraint_value_i)
                
                
                join_population[i].attributes['fitness'] += np.exp(
                    - EpsilonIndicator([adapted_objective_value_i]).compute([adapted_index_worst]) / (c*self.kappa))
                
                #if np.max(adapted_constraint_value_i) != 0:
                    #testjoin_population[i].constraints
                    #join_population[i].attributes['fitness'] += np.exp(
                        #- EpsilonIndicator([test]).compute([join_population[i].constraints]) / (self.kappa)) 
                join_population[i].attributes['fitness'] += sum(join_population[i].constraints) #/ self.kappa     
                
                
                
            join_population.pop(index_worst)
            join_population_size = join_population_size - 1

        return join_population

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return 'Adaptive-Epsilon-IBEA'