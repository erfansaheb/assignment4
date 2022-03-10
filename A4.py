# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:43:27 2022

@author: erfan
"""
from pdp_utils import *
import numpy as np
import pandas as pd
from time import time
from copy import deepcopy

def cost4V(Vnum, Vcalls, problem):
    Cargo = problem['Cargo']
    FirstTravelCost = problem['FirstTravelCost']
    TravelCost = problem['TravelCost']
    PortCost = problem['PortCost']
    
    Vcalls = Vcalls - 1
    sortRout = np.sort(Vcalls, kind='mergesort')
    I = np.argsort(Vcalls, kind='mergesort')
    Indx = np.argsort(I, kind='mergesort')

    PortIndex = Cargo[sortRout, 1].astype(int)
    PortIndex[::2] = Cargo[sortRout[::2], 0]
    PortIndex = PortIndex[Indx] - 1

    Diag = TravelCost[Vnum, PortIndex[:-1], PortIndex[1:]]

    FirstVisitCost = FirstTravelCost[Vnum, int(Cargo[Vcalls[0], 0] - 1)]
    RouteTravelCost = np.sum(np.hstack((FirstVisitCost, Diag.flatten())))
    CostInPorts = np.sum(PortCost[Vnum, Vcalls]) / 2
    return CostInPorts + RouteTravelCost

def SA(init_sol, probability, operators, cost_func, prob, T_f = 0.1, warm_up = 100):
    incumbent = init_sol
    best_sol = init_sol
    cost_incumb = cost_func(incumbent, prob)
    best_cost = cost_incumb 
    delta = []
    for w in range(warm_up):
        operator = np.random.choice(operators, replace=True, p=probability )
        new_sol = operator(incumbent, prob)
        new_cost = cost_func(new_sol, prob)
        delta_E = new_cost - cost_incumb
        feasiblity, c = feasibility_check(new_sol, prob)
        if feasiblity and delta_E < 0:
            incumbent = new_sol
            cost_incumb = new_cost
            if cost_incumb < best_cost:
                best_sol = incumbent
                best_cost = cost_incumb
        elif feasiblity:
            if np.random.uniform() < 0.8:
                incumbent = new_sol
                cost_incumb = new_cost
            delta.append(delta_E)
    delta_avg = np.mean(delta)
    T_0 = -delta_avg / np.log(0.8)
    alpha = np.power((T_f/T_0), (1/(10000-warm_up)))
    T = T_0
    for itr in range(10000-warm_up):
        operator = np.random.choice(operators, replace=True, p=probability )
        new_sol = operator(incumbent, prob)
        new_cost = cost_func(new_sol, prob)
        delta_E = new_cost - cost_incumb
        feasiblity, c = feasibility_check(new_sol, prob)
        if feasiblity and delta_E < 0:
            incumbent = new_sol
            cost_incumb = new_cost
            if cost_incumb < best_cost:
                best_sol = incumbent
                best_cost = cost_incumb
        elif feasiblity and np.random.uniform() < np.exp(-delta_E/T):
            incumbent = new_sol
            cost_incumb = new_cost
        T *= alpha
    return best_sol, best_cost

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
        1,2,3
        # one_ins,
        # two_ex, 
        # three_ex
        ]
    probabilities = [
        [1/3,1/3,1/3],
        [1/2,1/4,1/4]
        ]
    
    repeat = 10
    for j, p in enumerate(problems):
        info = pd.DataFrame(columns=['problem', 'operator', 'solution', 'average_objective', 'best_objective',
                                      'improvement', 'running_time'])
        for prb in probabilities:
            start = time()
# =============================================================================
#             update this later before submission
# =============================================================================
            prob = load_problem( "..//..//Data//" +p+ ".txt")
            initial_sol = np.array([0]*prob['n_vehicles'] + [i for i in range(1,prob['n_calls']+1) for j in range(2)])
            init_cost = cost_function(initial_sol, prob)
            best_sol, best_cost = np.empty((repeat ,len(initial_sol)),dtype=int), np.empty(repeat )
            for i in range(repeat ):
                np.random.seed(23+i)
                # print('seed', 23+i)
                best_sol[i], best_cost[i] = SA(initial_sol, prb, operators, cost_function, prob, warm_up = 100)
                # print(best_sol[i],best_cost[i])
            info = info.append({'problem': str(j)+str(alg),
                                'operator': str(o),
                                'solution': best_sol[np.argmin(best_cost)],
                                'average_objective': np.mean(best_cost),
                                'best_objective':np.min(best_cost),
                                'improvement': 100*((init_cost-np.min(best_cost))/init_cost),
                                'running_time': (time()-start)/10}, ignore_index=True)
                # print(info)
        info = info.astype({'average_objective': float,
                            'best_objective':int,
                            'improvement': float,
                            'running_time': float}).set_index('problem').to_csv('results'+str(j)+'.csv')
