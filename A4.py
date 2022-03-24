# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:43:27 2022

@author: erfan
"""
from pdp_utils import load_problem, feasibility_check, cost_function
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

def cap_TW_4V(Vnum, Vcalls, problem): #Capacity and timewindow check for a vehicle
    if Vnum == problem['n_vehicles']:
        return True
    Vcalls = Vcalls - 1
    VesselCapacity = problem['VesselCapacity']
    Cargo = problem['Cargo']
    UnloadingTime = problem['UnloadingTime']
    LoadingTime = problem['LoadingTime']
    TravelTime = problem['TravelTime']
    FirstTravelTime = problem['FirstTravelTime']
    NoDoubleCallOnVehicle = len(Vcalls)
    LoadSize = 0
    currentTime = 0
    sortRout = np.sort(Vcalls, kind='mergesort')
    I = np.argsort(Vcalls, kind='mergesort')
    Indx = np.argsort(I, kind='mergesort')
    LoadSize -= Cargo[sortRout, 2]
    LoadSize[::2] = Cargo[sortRout[::2], 2]
    LoadSize = LoadSize[Indx]
    if np.any(VesselCapacity[Vnum] - np.cumsum(LoadSize) < 0):
        return False
    Timewindows = np.zeros((2, NoDoubleCallOnVehicle))
    Timewindows[0] = Cargo[sortRout, 6]
    Timewindows[0, ::2] = Cargo[sortRout[::2], 4]
    Timewindows[1] = Cargo[sortRout, 7]
    Timewindows[1, ::2] = Cargo[sortRout[::2], 5]

    Timewindows = Timewindows[:, Indx]

    PortIndex = Cargo[sortRout, 1].astype(int)
    PortIndex[::2] = Cargo[sortRout[::2], 0]
    PortIndex = PortIndex[Indx] - 1

    LU_Time = UnloadingTime[Vnum, sortRout]
    LU_Time[::2] = LoadingTime[Vnum, sortRout[::2]]
    LU_Time = LU_Time[Indx]
    Diag = TravelTime[Vnum, PortIndex[:-1], PortIndex[1:]]
    FirstVisitTime = FirstTravelTime[Vnum, int(Cargo[Vcalls[0], 0] - 1)]

    RouteTravelTime = np.hstack((FirstVisitTime, Diag.flatten()))

    ArriveTime = np.zeros(NoDoubleCallOnVehicle)
    for j in range(NoDoubleCallOnVehicle):
        ArriveTime[j] = np.max((currentTime + RouteTravelTime[j], Timewindows[0, j]))
        if ArriveTime[j] > Timewindows[1, j]:
            return False
        currentTime = ArriveTime[j] + LU_Time[j]
    return True

def findBestPosForDel(call, Vnum, Vcalls, prob):
    PickIndex = np.array(np.where(Vcalls == call)[0], dtype=int)[0]
    best_sol = np.insert(Vcalls, PickIndex+1, call)
    if cap_TW_4V(Vnum, Vcalls, prob): 
        best_cost = cost4V(Vnum, best_sol, prob)
    else:
        best_cost = np.inf
    i_best = PickIndex+1
    if i_best == len(Vcalls):
        return i_best, best_cost
    for i in range(PickIndex+2, len(Vcalls)+1):
        new_sol = np.insert(Vcalls, i, call)
        new_cost = cost4V(Vnum, new_sol, prob)
        if new_cost < best_cost:
            best_sol = new_sol
            best_cost = new_cost
            i_best = i
    return i_best, best_cost

def one_ins_first_best(sol, prob):
    Solution = np.append(sol, [0])
# =============================================================================
#     if np.random.random()< 0.30:
#         selection_prob = np.sum(1-prob['VesselCargo'], axis = 0)/np.sum(1-prob['VesselCargo'])
#     else:
#         selection_prob = np.sum(prob['VesselCargo'], axis = 0)/np.sum(prob['VesselCargo'])
#     selected_call = np.random.choice(np.arange(1,prob['n_calls']+1), p = selection_prob)
# =============================================================================
    selected_call = np.random.choice(np.arange(1,prob['n_calls']+1))
    call_locs = np.where(Solution==selected_call)[0]
    ZeroIndexBef = np.array(np.where(Solution == 0)[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = len(ZeroIndexBef[ZeroIndexBef<call_locs[0]])
    if cur_v == prob['n_vehicles']:
        best_cost = prob['Cargo'][selected_call-1,3]
    else:
        sol_2 = np.delete(Solution, call_locs)
        best_cost = cost4V(cur_v, Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]], prob) - \
            (cost4V(cur_v, sol_2[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]-2], prob) if 
             ZeroIndexBef[cur_v]-2 >ZeroIndexBef[cur_v]-len_v[cur_v] else 0 )
    # best_cost = np.inf    
    best_sol = Solution
    Solution = np.delete(Solution, call_locs)
    ZeroIndex = np.array(np.where(Solution == 0)[0], dtype=int)
    avail_vehs = np.append(np.where(prob['VesselCargo'][:,selected_call-1] == 1)[0], prob['n_vehicles'])
    len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]

    for v in avail_vehs:
        if v == prob['n_vehicles']:
            if best_cost > prob['Cargo'][selected_call-1,3]:
                best_sol = np.insert(Solution,-1, [selected_call,selected_call])
                break
            continue
        for pos in range(len_v[v]+1):
            if len_v[v] == 0:
                cost_v = 0
            else:
                cost_v = cost4V(v, Solution[ZeroIndex[v]-len_v[v]:ZeroIndex[v]], prob)
            new_sol = np.insert(Solution,ZeroIndex[v]+ pos - len_v[v] , selected_call)
            del_idx, new_cost = findBestPosForDel(selected_call, v, new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1] ,prob)
            if new_cost - cost_v <= best_cost:
                best_sol = np.insert(new_sol,ZeroIndex[v]-len_v[v]+ del_idx, selected_call)
                best_cost = new_cost
                break
            else:
               Solution = np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
    # if best_sol == Solution:
    #     best_sol = np.append(one_ins_first_best(sol, prob),[0])
    return best_sol [:-1]

def one_ins_best(sol, prob):
    Solution = np.append(sol, [0])
# =============================================================================
#     if np.random.random()< 0.30:
#         selection_prob = np.sum(1-prob['VesselCargo'], axis = 0)/np.sum(1-prob['VesselCargo'])
#     else:
#         selection_prob = np.sum(prob['VesselCargo'], axis = 0)/np.sum(prob['VesselCargo'])
#     selected_call = np.random.choice(np.arange(1,prob['n_calls']+1), p = selection_prob)
# =============================================================================
    selected_call = np.random.choice(np.arange(1,prob['n_calls']+1))
    call_locs = np.where(Solution==selected_call)[0]
    ZeroIndexBef = np.array(np.where(Solution == 0)[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = len(ZeroIndexBef[ZeroIndexBef<call_locs[0]])
    if cur_v == prob['n_vehicles']:
        best_cost = prob['Cargo'][selected_call-1,3]
    else:
        sol_2 = np.delete(Solution, call_locs)
        best_cost = cost4V(cur_v, Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]], prob) - \
            (cost4V(cur_v, sol_2[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]-2], prob) if 
              ZeroIndexBef[cur_v]-2 >ZeroIndexBef[cur_v]-len_v[cur_v] else 0 )
    # best_cost = np.inf    
    best_sol = Solution
    Solution = np.delete(Solution, call_locs)
    ZeroIndex = np.array(np.where(Solution == 0)[0], dtype=int)
    avail_vehs = np.append(np.where(prob['VesselCargo'][:,selected_call-1] == 1)[0], prob['n_vehicles'])
    len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]

    for v in avail_vehs:
        if v == prob['n_vehicles']:
            if best_cost > prob['Cargo'][selected_call-1,3]:
                best_sol = np.insert(Solution,-1, [selected_call,selected_call])
            continue
        for pos in range(len_v[v]+1):
            if len_v[v] == 0:
                cost_v = 0
            else:
                cost_v = cost4V(v, Solution[ZeroIndex[v]-len_v[v]:ZeroIndex[v]], prob)
            new_sol = np.insert(Solution,ZeroIndex[v]+ pos - len_v[v] , selected_call)
            del_idx, new_cost = findBestPosForDel(selected_call, v, new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1] ,prob)
            if new_cost - cost_v <= best_cost:
                best_sol = np.insert(new_sol,ZeroIndex[v]-len_v[v]+ del_idx, selected_call)
                best_cost = new_cost
            else:
               Solution = np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
    # if (best_sol[:-1] == sol).all():
    #     best_sol = np.append(one_ins_best(sol, prob),[0])
    return best_sol [:-1]

def multi_ins_new(sol, prob, rm_size = 3):
    Solution = np.append(sol, [0])
    selection_prob = np.sum(prob['VesselCargo'], axis = 0)/np.sum(prob['VesselCargo'])
    selected_calls = np.random.choice(np.arange(1,prob['n_calls']+1),
                                     size = rm_size,
                                     replace = False,
                                     p = selection_prob)
    ZeroIndexBef = np.array(np.where(Solution == 0)[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    for k in range(rm_size):
        for selected_call in selected_calls:
            call_locs = np.where(Solution==selected_call)[0]
            cur_v = len(ZeroIndexBef[ZeroIndexBef<call_locs[0]])
            if cur_v == prob['n_vehicles']:
                best_cost = prob['Cargo'][selected_call-1,3]
            else:
                best_cost = cost4V(cur_v, Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]], prob)
            best_sol = Solution
            Solution = np.delete(Solution, call_locs)
            ZeroIndex = np.array(np.where(Solution == 0)[0], dtype=int)
            avail_vehs = np.where(prob['VesselCargo'][:,selected_call-1] == 1)[0]
            len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
            
            for v in avail_vehs:
                for pos in range(len_v[v]+1):
                    if len_v[v] == 0:
                        cost_v = 0
                    else:
                        cost_v = cost4V(v, Solution[ZeroIndex[v]-len_v[v]:ZeroIndex[v]], prob)
                    new_sol = np.insert(Solution,ZeroIndex[v]+ pos - len_v[v] , selected_call)
                    del_idx, new_cost = findBestPosForDel(selected_call, v, new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1] ,prob)
                    if new_cost - cost_v < best_cost:
                        best_sol = np.insert(new_sol,ZeroIndex[v]-len_v[v]+ del_idx, selected_call)
                        best_cost = new_cost
                    else:
                       Solution = np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
            Solution = best_sol
    return best_sol [:-1]
    

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
    if delta_avg == 0.0:
        delta_avg = new_cost/warm_up
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
# =============================================================================
# if __name__ == '__main__':
#     problems = [
#                 'Call_7_Vehicle_3',
#                 'Call_18_Vehicle_5',
#                 'Call_35_Vehicle_7',
#                 'Call_80_Vehicle_20',
#                 'Call_130_Vehicle_40',
#                 'Call_300_Vehicle_90'
#                 ]
#     operators = [
#         1,2,3
#         # one_ins_best,
#         # two_ex, 
#         # three_ex
#         ]
#     probabilities = [
#         [1/3,1/3,1/3],
#         [1/2,1/4,1/4]
#         ]
#     
#     repeat = 10
#     for j, p in enumerate(problems):
#         info = pd.DataFrame(columns=['problem', 'operator', 'solution', 'average_objective', 'best_objective',
#                                       'improvement', 'running_time'])
#         for prb in probabilities:
#             start = time()
# # =============================================================================
# #             update this later before submission
# # =============================================================================
#             prob = load_problem( "..//..//Data//" +p+ ".txt")
#             initial_sol = np.array([0]*prob['n_vehicles'] + [i for i in range(1,prob['n_calls']+1) for j in range(2)])
#             init_cost = cost_function(initial_sol, prob)
#             best_sol, best_cost = np.empty((repeat ,len(initial_sol)),dtype=int), np.empty(repeat )
#             for i in range(repeat ):
#                 np.random.seed(23+i)
#                 # print('seed', 23+i)
#                 best_sol[i], best_cost[i] = SA(initial_sol, prb, operators, cost_function, prob, warm_up = 100)
#                 # print(best_sol[i],best_cost[i])
#             info = info.append({'problem': str(j)+str(alg),
#                                 'operator': str(o),
#                                 'solution': best_sol[np.argmin(best_cost)],
#                                 'average_objective': np.mean(best_cost),
#                                 'best_objective':np.min(best_cost),
#                                 'improvement': 100*((init_cost-np.min(best_cost))/init_cost),
#                                 'running_time': (time()-start)/10}, ignore_index=True)
#                 # print(info)
#         info = info.astype({'average_objective': float,
#                             'best_objective':int,
#                             'improvement': float,
#                             'running_time': float}).set_index('problem').to_csv('results'+str(j)+'.csv')
# =============================================================================
np.random.seed(23)
prob = load_problem( "..//..//Data//" +'Call_7_Vehicle_3'+ ".txt")
# # # # findBestPosForDel(2,1,np.array([2, 9, 9]),prob)
sol = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
# sol = np.array([0]*prob['n_vehicles'] + [i for i in range(1,prob['n_calls']+1) for j in range(2)])
# a = one_ins_best(sol, prob)
# start = time()
a = SA(sol, [1],[one_ins_best], cost_function, prob)
# # for i in range(288):
# #     np.random.randint(1)
# # for i in range(10):
# #     b = SA(sol, three_ex, cost_function, prob)
# print(time()-start)
# two_ex(sol, prob)