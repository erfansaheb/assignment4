# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:43:27 2022

@author: erfan
"""
from pdp_utils import load_problem, feasibility_check, cost_function
from operators import one_ins_best, one_ins_first_best, multi_ins_new
from auxiliary_functions import copy_costs, copy_features
import numpy as np
import pandas as pd
from time import time
from copy import deepcopy


def SA(init_sol, init_cost, probability, operators, prob, T_f = 0.1, warm_up = 100):
    
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
    delta = []
    deltas = []
    for w in range(warm_up):
        if w == 38:
            print('here')
        operator = np.random.choice(operators, replace=True, p=probability )
        new_sol, new_costs, new_features = operator(incumbent, copy_costs(costs), copy_features(features), prob)
        new_cost = sum(new_costs)
        delta_E = new_cost - cost_incumb
        deltas.append(delta_E)
        feasiblity, c = True, 'Feasible'
        if feasiblity and delta_E < 0:
            incumbent = new_sol
            cost_incumb = new_cost
            costs = copy_costs(new_costs)
            features = copy_features(new_features)
            if cost_incumb < best_cost:
                best_sol = incumbent
                best_cost = cost_incumb
                last_improvement = w
                
        elif feasiblity:
            if np.random.uniform() < 0.8:
                incumbent = new_sol
                cost_incumb = new_cost
                costs = copy_costs(new_costs)
                features = copy_features(new_features)
            delta.append(delta_E)
    delta_avg = np.mean(delta)
    if delta_avg == 0.0:
        delta_avg = new_cost/warm_up
    T_0 = -delta_avg / np.log(0.8)
    alpha = 0.9995#np.power((T_f/T_0), (1/(10000-warm_up)))
    T = T_0
    Ts = [T]
    Ps = [np.exp(-delta_avg/T)]
    for itr in range(10000-warm_up):
        operator = np.random.choice(operators, replace=True, p=probability )
        new_sol, new_costs, new_features = operator(incumbent, copy_costs(costs), copy_features(features), prob)
        new_cost = sum(new_costs)
        delta_E = new_cost - cost_incumb
        deltas.append(delta_E)
        feasiblity, c = True, 'Feasible'
        
        if feasiblity and delta_E < 0:
            incumbent = new_sol
            cost_incumb = new_cost
            costs = copy_costs(new_costs)
            features = copy_features(new_features)
            if cost_incumb < best_cost:
                best_sol = incumbent
                best_cost = cost_incumb
                last_improvement = itr + warm_up
                
        elif feasiblity:
            prbb = np.exp(-delta_E/T)
            Ps.append(np.exp(-delta_E/T))
            if np.random.uniform() < prbb:
        # and np.random.uniform() < np.exp(-delta_E/T):
                incumbent = new_sol
                cost_incumb = new_cost
                costs = copy_costs(new_costs)
                features = copy_features(new_features)
            
            
            
        T *= alpha
        Ts.append(T)
    return best_sol, best_cost, last_improvement, Ps, Ts,deltas

def localSearch(init_sol, probability, operators, cost_func, prob, warm_up = None):
    best_sol = init_sol
    best_cost = cost_func(best_sol, prob)
    for i in range(10000):
        operator = np.random.choice(operators, replace=True, p=probability )
        new_sol = operator(best_sol,prob)
        feasiblity, c = feasibility_check(new_sol, prob)
        new_cost = cost_func(new_sol, prob)
        if feasiblity and new_cost < best_cost:
            best_sol = new_sol
            best_cost = new_cost
    return best_sol, best_cost

if __name__ == '__main__':
    problems = [
                # 'Call_7_Vehicle_3',
                'Call_18_Vehicle_5',
                # 'Call_35_Vehicle_7',
                # 'Call_80_Vehicle_20',
                # 'Call_130_Vehicle_40',
                # 'Call_300_Vehicle_90'
                ]
    operators = [
        one_ins_best,
        one_ins_first_best, 
        multi_ins_new
        ]
    probabilities = [
        # [1/len(operators) for i in operators],
        [0,0,1]
        ]
    
    repeat = 1
    info = pd.DataFrame(columns=['problem','probability', 'solution', 'average_objective', 'best_objective',
                                  'improvement', 'best_objs', 'last_improvement', 'running_time', 'Ps'])
    for j, p in enumerate(problems):
        
        for prb in probabilities:
            start = time()
            prob = load_problem( "..//..//Data//" +p+ ".txt")
            initial_sol = [0]*prob['n_vehicles'] + [i for i in range(1,prob['n_calls']+1) for j in range(2)]
            init_cost = prob['Cargo'][:,3].sum()
            best_sol, best_cost, last_improvement = [[] for i in range(repeat)], [0 for i in range(repeat)], [0 for i in range(repeat)]
            Ps = [[] for i in range(repeat)]
            Ts = [[] for i in range(repeat)]
            deltas = [[] for i in range(repeat)]
            for i in range(repeat ):
                np.random.seed(23+i)
                # print('seed', 23+i)
                best_sol[i], best_cost[i], last_improvement[i], Ps[i], Ts[i],deltas[i] = SA(initial_sol, init_cost, prb, operators, prob, warm_up = 100)
                # print(best_sol[i],best_cost[i])
            info = info.append({'problem': str(j),
                                'probability': prb,
                                'solution': best_sol[np.argmin(best_cost)],
                                'average_objective': np.mean(best_cost),
                                'best_objective':np.min(best_cost),
                                'improvement': 100*((init_cost-np.min(best_cost))/init_cost),
                                'best_objs': best_cost,
                                'last_improvement': last_improvement,
                                'running_time': (time()-start)/repeat,
                                'Ps': Ps}, ignore_index=True,
                               )
                # print(info)
# =============================================================================
#         info.astype({'average_objective': float,
#                             'best_objective':int,
#                             'improvement': float,
#                             'running_time': float}).set_index('problem').to_csv('results'+str(j)+'.csv')
# =============================================================================
# =============================================================================
# np.random.seed(23)
# prob = load_problem( "..//..//Data//" +'Call_18_Vehicle_5'+ ".txt")
# # # # # findBestPosForDel(2,1,np.array([2, 9, 9]),prob)
# # sol = np.array([4, 4, 2, 2, 0, 7, 7, 0, 1, 5, 5, 3, 3, 1, 0, 6, 6 ])
# sol = [0]*prob['n_vehicles'] + [i for i in range(1,prob['n_calls']+1) for j in range(2)]
# init_cost = prob['Cargo'][:,3].sum()#cost_function(sol, prob)
# # a = one_ins_best(sol, prob)
# start = time()
# # a = SA(sol,init_cost, 
# #         # [1/2,1/2,0],
# #         [1/3,1/3,1/3],
# #         [
# #         one_ins_best,
# #                 one_ins_first_best, 
# #                 multi_ins_new], prob)
# # # for i in range(288):
# # #     np.random.randint(1)
# b = []
# for i in range(10):
#     b.append(SA(sol,init_cost, [1/3,1/3,1/3],[
#             one_ins_best,
#                     one_ins_first_best, 
#                     multi_ins_new], prob))
# print(time()-start)
# # two_ex(sol, prob)
# =============================================================================


# =============================================================================
# 
# =============================================================================
