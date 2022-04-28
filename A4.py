# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:43:27 2022

@author: erfan
"""
from pdp_utils import load_problem, feasibility_check, cost_function
from operators import one_ins_best, one_ins_first_better, multi_ins_rand
from auxiliary_functions import copy_costs, copy_features
import numpy as np
from time import time

def SA(init_sol, init_cost, probability, operators, prob, rng, T_f = 0.1, warm_up = 100):
    
    incumbent = init_sol
    best_sol = init_sol
    cost_incumb = init_cost
    costs = [0 for i in range(prob['n_vehicles'])]+ [init_cost]
    LoadSize = [[] for i in range(prob['n_vehicles'])]
    Timewindows = [[[],[]] for i in range(prob['n_vehicles'])]
    PortIndex = [[] for i in range(prob['n_vehicles'])]
    LU_Time = [[] for i in range(prob['n_vehicles'])]
    features = [LoadSize, Timewindows, PortIndex, LU_Time]
    last_improvement = 0
    best_cost = cost_incumb 
    delta = [0]
    for itr in range(10000):
        if itr == warm_up and np.mean(delta) == 0:
            warm_up += 100
        if itr < warm_up:
            op_id = rng.choice(range(len(operators)), replace=True, p=probability )
            operator = operators[op_id]
            new_sol, new_costs, new_features = operator(incumbent, copy_costs(costs), copy_features(features), rng, prob)
            new_cost = sum(new_costs)

            delta_E = new_cost - cost_incumb

            feasiblity, c = True, 'Feasible'
            if feasiblity and delta_E < 0:
                incumbent = new_sol

                cost_incumb = new_cost
                costs = copy_costs(new_costs)
                features = copy_features(new_features)
                if cost_incumb < best_cost:
                    best_sol = incumbent
                    best_cost = cost_incumb
                    last_improvement = itr
                
            elif feasiblity:
                if rng.uniform() < 0.8:
                    incumbent = new_sol
                    cost_incumb = new_cost
                    costs = copy_costs(new_costs)
                    features = copy_features(new_features)
                if delta_E>0:
                    delta.append(delta_E)
        else:
            if itr == warm_up:
                delta_avg = np.mean(delta[1:])
                T_0 = -delta_avg / np.log(0.8)
                alpha = 0.9995
                T = T_0

            op_id = rng.choice(range(len(operators)), replace=True, p=probability )
            operator = operators[op_id]
            new_sol, new_costs, new_features = operator(incumbent, copy_costs(costs), copy_features(features), rng, prob)
            new_cost = sum(new_costs)
            delta_E = new_cost - cost_incumb
            feasiblity, c = True, 'Feasible'
            
            if feasiblity and delta_E < 0:

                incumbent = new_sol
                cost_incumb = new_cost
                costs = copy_costs(new_costs)
                features = copy_features(new_features)
                if cost_incumb < best_cost:
                    best_sol = incumbent
                    best_cost = cost_incumb
                    last_improvement = itr 
                    
            elif feasiblity:
                prbb = np.exp(-delta_E/T)
                if rng.uniform() < prbb:
                    incumbent = new_sol
                    cost_incumb = new_cost
                    costs = copy_costs(new_costs)
                    features = copy_features(new_features)
                
                
                
            T *= alpha
    return best_sol, best_cost, last_improvement

if __name__ == '__main__':
    problems = [
                'Call_7_Vehicle_3',
                'Call_18_Vehicle_5',
                'Call_35_Vehicle_7',
                'Call_80_Vehicle_20',
                'Call_130_Vehicle_40',
                'Call_300_Vehicle_90'
                ]
    operators = [
        one_ins_best,
        one_ins_first_better,
        multi_ins_rand
        ]
    probabilities = [
        [1/len(operators) for i in operators],
        [1/4,1/4,1/2]
        ]
    
    repeat = 1
    print('problem \t', 'probability \t', 'average_objective \t', 'best_objective \t', 'improvement \t', 'running_time')
    for j, p in enumerate(problems):
        
        for prb in probabilities:
            start = time()
            prob = load_problem( "Data//" +p+ ".txt")
            initial_sol = [0]*prob['n_vehicles'] + [i for i in range(1,prob['n_calls']+1) for j in range(2)]
            init_cost = prob['Cargo'][:,3].sum()
            best_sol, best_cost, last_improvement = [[] for i in range(repeat)], [0 for i in range(repeat)], [0 for i in range(repeat)]
            Ps = [[] for i in range(repeat)]
            Ts = [[] for i in range(repeat)]
            deltas = [[] for i in range(repeat)]
            for i in range(repeat ):
                rng = np.random.default_rng(31+i)
                best_sol[i], best_cost[i], last_improvement[i] = SA(initial_sol, init_cost, prb, operators, prob, rng, warm_up = 100)

            running_time = (time()-start)/repeat
            minidx = np.argmin(best_cost)
            print(p,'\t', str(prb), '\t', str(np.mean(best_cost)), '\t', str(best_cost[minidx]), '\t', 100*((init_cost-best_cost[minidx])/init_cost),
                  '\t', running_time)
            print('Solution: ', str(best_sol[minidx]))

