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
    FirstTravelCost = problem['FirstTravelCost']
    TravelCost = problem['TravelCost']
    PortCost = problem['PortCost']
    VesselCapacity = problem['VesselCapacity']
    Cargo = problem['Cargo']
    UnloadingTime = problem['UnloadingTime']
    LoadingTime = problem['LoadingTime']
    TravelTime = problem['TravelTime']
    FirstTravelTime = problem['FirstTravelTime']

    Vcalls = Vcalls - 1
    
    NoDoubleCallOnVehicle = len(Vcalls)
    LoadSize = 0
    currentTime = 0
    sortRout = np.sort(Vcalls, kind='mergesort')
    I = np.argsort(Vcalls, kind='mergesort')
    Indx = np.argsort(I, kind='mergesort')
    LoadSize -= Cargo[sortRout, 2]
    LoadSize[::2] = Cargo[sortRout[::2], 2]
    LoadSize = LoadSize[Indx]
    # if np.any(VesselCapacity[Vnum] - np.cumsum(LoadSize) < 0):
    #     return False, 'Capacity exceeded', 0
    Timewindows = np.zeros((2, NoDoubleCallOnVehicle))
    Timewindows[0] = Cargo[sortRout, 6]
    Timewindows[0, ::2] = Cargo[sortRout[::2], 4]
    Timewindows[1] = Cargo[sortRout, 7]
    Timewindows[1, ::2] = Cargo[sortRout[::2], 5]
    Timewindows = Timewindows[:, Indx]
    
    PortIndex = Cargo[sortRout, 1].astype(int)
    PortIndex[::2] = Cargo[sortRout[::2], 0]
    PortIndex = PortIndex[Indx] - 1
    
    Diag_cost = TravelCost[Vnum, PortIndex[:-1], PortIndex[1:]]
    
    FirstVisitCost = FirstTravelCost[Vnum, int(Cargo[Vcalls[0], 0] - 1)]
    RouteTravelCost = np.sum(np.hstack((FirstVisitCost, Diag_cost.flatten())))
    CostInPorts = np.sum(PortCost[Vnum, Vcalls]) / 2
    
    LU_Time = UnloadingTime[Vnum, sortRout]
    LU_Time[::2] = LoadingTime[Vnum, sortRout[::2]]
    LU_Time = LU_Time[Indx]
    Diag = TravelTime[Vnum, PortIndex[:-1], PortIndex[1:]]
    FirstVisitTime = FirstTravelTime[Vnum, int(Cargo[Vcalls[0], 0] - 1)]
    
    RouteTravelTime = np.hstack((FirstVisitTime, Diag.flatten()))
    
    ArriveTime = np.zeros(NoDoubleCallOnVehicle)
    for j in range(NoDoubleCallOnVehicle):
        if VesselCapacity[Vnum] - np.cumsum(LoadSize)[j] < 0:
            return False, 'Capacity exceeded at call {}'.format(j), np.inf
        ArriveTime[j] = np.max((currentTime + RouteTravelTime[j], Timewindows[0, j]))
        if ArriveTime[j] > Timewindows[1, j]:
            return False, 'Time window exceeded at call {}'.format(j), np.inf
        currentTime = ArriveTime[j] + LU_Time[j]
    return True, 'Feasible', CostInPorts + RouteTravelCost

def findBestPosForDel(call, Vnum, Vcalls, prob):
    PickIndex = np.array(np.where(Vcalls == call)[0], dtype=int)[0]
    best_cost = np.inf
    i_best = 0
    for i in range(PickIndex+1, len(Vcalls)+1):
        new_sol = np.insert(Vcalls, i, call)
        new_feasibility, new_c, new_cost = cap_TW_4V(Vnum, new_sol, prob)
        if new_c=='Feasible':
            c = new_c
            feasibility = new_feasibility
            if i == len(Vcalls):
                return i, new_cost, feasibility, new_c
            elif new_cost < best_cost:
                # best_sol = new_sol
                best_cost = new_cost
                i_best = i
        else:
            if (int(new_c[-1]) == PickIndex or int(new_c[-1]) == i) or \
                (int(new_c[-1]) > PickIndex and int(new_c[-1]) < i):
                break
    try:
        return i_best, best_cost, feasibility, c
    except:
        return i_best, best_cost, new_feasibility, new_c

def one_ins_first_best(sol, costs, prob):
    Solution = np.append(sol, [0])
# =============================================================================
#     if np.random.random()< 0.30:
#         selection_prob = np.sum(1-prob['VesselCargo'], axis = 0)/np.sum(1-prob['VesselCargo'])
#     else:
#         selection_prob = np.sum(prob['VesselCargo'], axis = 0)/np.sum(prob['VesselCargo'])
# =============================================================================
    # selection_prob = np.sum(prob['VesselCargo'], axis = 0)/np.sum(prob['VesselCargo'])    
    # selected_call = np.random.choice(np.arange(1,prob['n_calls']+1), p = selection_prob)
    selected_call = np.random.choice(np.arange(1,prob['n_calls']+1))
    call_locs = np.where(Solution==selected_call)[0]
    ZeroIndexBef = np.array(np.where(Solution == 0)[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = len(ZeroIndexBef[ZeroIndexBef<call_locs[0]])
    if cur_v == prob['n_vehicles']:
        cur_cost = prob['Cargo'][selected_call-1,3]
    else:
        sol_2 = np.delete(Solution, call_locs)
        cur_cost = cost4V(cur_v, Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]], prob) - \
            (cost4V(cur_v, sol_2[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]-2], prob) if 
             ZeroIndexBef[cur_v]-2 >ZeroIndexBef[cur_v]-len_v[cur_v] else 0 )
    # best_cost = np.inf    
    costs[cur_v] -= cur_cost
    costs[-1] += prob['Cargo'][selected_call-1,3]
    best_sol = Solution
    Solution = np.delete(Solution, call_locs)
    ZeroIndex = np.array(np.where(Solution == 0)[0], dtype=int)
    avail_vehs = np.append(np.where(prob['VesselCargo'][:,selected_call-1] == 1)[0], prob['n_vehicles'])
    len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
    best_v = prob['n_vehicles']
    best_cost = np.inf
    for v in avail_vehs:
        if v == prob['n_vehicles']:
            if best_cost >= prob['Cargo'][selected_call-1,3]:
                best_sol = np.insert(Solution,-1, [selected_call,selected_call])
                break
            continue
        for pos in range(len_v[v]+1):
            cost_v = costs[v]
# =============================================================================
#             if len_v[v] == 0:
#                 cost_v = 0
#             else:
#                 cost_v = cost4V(v, Solution[ZeroIndex[v]-len_v[v]:ZeroIndex[v]], prob)
# =============================================================================
            new_sol = np.insert(Solution,ZeroIndex[v]+ pos - len_v[v] , selected_call)
            del_idx, new_cost, feasibility, c = findBestPosForDel(selected_call, v, new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1] ,prob)
            if not c.startswith('F'):
                if int(c[-1]) == pos or int(c[-1]) == del_idx:
                    Solution = np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
                    if c.startswith('T'):
                        break
                    else:
                        continue
                else:
                    Solution = np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
                    continue
            if new_cost - cost_v <= best_cost:
                best_sol = np.insert(new_sol,ZeroIndex[v]-len_v[v]+ del_idx, selected_call)
                best_cost = new_cost - cost_v
                best_v = v
                break
            else:
               Solution = np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
# =============================================================================
#         if best_v != prob['n_vehicles']:
#             break
# =============================================================================
    # if( best_sol[:-1] == sol).all():
    #     best_sol, costs = one_ins_first_best(sol, costs, prob)
    #     best_sol = np.append(best_sol,[0])
    if best_v != prob['n_vehicles']:
        costs[best_v] += best_cost
        costs[-1] -= prob['Cargo'][selected_call-1,3]
    return best_sol [:-1], costs

def one_ins_best(sol, costs, prob):
    Solution = np.append(sol, [0])
# =============================================================================
#     if np.random.random()< 0.30:
#         selection_prob = np.sum(1-prob['VesselCargo'], axis = 0)/np.sum(1-prob['VesselCargo'])
#     else:
#         selection_prob = np.sum(prob['VesselCargo'], axis = 0)/np.sum(prob['VesselCargo'])
# =============================================================================
    # selection_prob = np.sum(prob['VesselCargo'], axis = 0)/np.sum(prob['VesselCargo'])
    # selected_call = np.random.choice(np.arange(1,prob['n_calls']+1), p = selection_prob)
    selected_call = np.random.choice(np.arange(1,prob['n_calls']+1))
    call_locs = np.where(Solution==selected_call)[0]
    ZeroIndexBef = np.array(np.where(Solution == 0)[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = len(ZeroIndexBef[ZeroIndexBef<call_locs[0]])
    if cur_v == prob['n_vehicles']:
        cur_cost = prob['Cargo'][selected_call-1,3]
    else:
        sol_2 = np.delete(Solution, call_locs)
        cur_cost = cost4V(cur_v, Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]], prob) - \
            (cost4V(cur_v, sol_2[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]-2], prob) if 
              ZeroIndexBef[cur_v]-2 >ZeroIndexBef[cur_v]-len_v[cur_v] else 0 )
    # best_cost = np.inf    
    costs[cur_v] -= cur_cost
    costs[-1] += prob['Cargo'][selected_call-1,3]
    best_sol = Solution
    Solution = np.delete(Solution, call_locs)
    ZeroIndex = np.array(np.where(Solution == 0)[0], dtype=int)
    avail_vehs = np.append(np.where(prob['VesselCargo'][:,selected_call-1] == 1)[0], prob['n_vehicles'])
    len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
    best_v = prob['n_vehicles']
    best_cost = np.inf
    for v in avail_vehs:
        if v == prob['n_vehicles']:
            if best_cost >= prob['Cargo'][selected_call-1,3]:
                best_sol = np.insert(Solution,-1, [selected_call,selected_call])
            continue
        for pos in range(len_v[v]+1):
# =============================================================================
#             if len_v[v] == 0:
#                 cost_v = 0
#             else:
#                 cost_v = cost4V(v, Solution[ZeroIndex[v]-len_v[v]:ZeroIndex[v]], prob)
# =============================================================================
            cost_v = costs[v]
            new_sol = np.insert(Solution,ZeroIndex[v]+ pos - len_v[v] , selected_call)
            del_idx, new_cost, feasibility, c = findBestPosForDel(selected_call, v, new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1] ,prob)
            if not c.startswith('F'):
                if int(c[-1]) == pos or int(c[-1]) == del_idx:
                    Solution = np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
                    if c.startswith('T'):
                        break
                    else:
                        continue
                else:
                    Solution = np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
                    continue
                
            if new_cost - cost_v <= best_cost:
                best_sol = np.insert(new_sol,ZeroIndex[v]-len_v[v]+ del_idx, selected_call)
                best_cost = new_cost - cost_v
                best_v = v
            else:
               Solution = np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
# =============================================================================
#         if not c.startswith('F'):
#             if int(c[-1]) == pos or int(c[-1]) == del_idx:
#                 break
# =============================================================================
    # if (best_sol[:-1] == sol).all():
    #     best_sol, costs = one_ins_best(sol, costs, prob)
    #     best_sol= np.append(best_sol,[0])
    
    if best_v != prob['n_vehicles']:
        costs[best_v] += best_cost
        costs[-1] -= prob['Cargo'][selected_call-1,3]
    return best_sol [:-1], costs

def multi_ins_new(sol, costs, prob, rm_size = 3):
    Solution = np.append(sol, [0])
    
    # selection_prob = np.sum(prob['VesselCargo'], axis = 0)/np.sum(prob['VesselCargo'])
    selected_calls = np.random.choice(np.arange(1,prob['n_calls']+1),
                                     size = rm_size,
                                     replace = False,
                                      # p = selection_prob
                                     )
    # print(Solution)
    # print(selected_calls)
    sel_loc = np.array([np.where(Solution == selected_calls[i])[0] for i in range(rm_size)])
    ZeroIndexBef = np.array(np.where(Solution == 0)[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    for c_id, call in enumerate(selected_calls):
        # call_locs = np.where(Solution==call)[0]
        # ZeroIndexBef = np.array(np.where(Solution == 0)[0], dtype=int)
        # len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
        cur_v = len(ZeroIndexBef[ZeroIndexBef<sel_loc[c_id][0]])
        #cur_v = len(ZeroIndexBef[ZeroIndexBef<call_locs[0]])
        if cur_v == prob['n_vehicles']:
            best_cost = prob['Cargo'][call-1,3]
            Solution = np.delete(Solution, sel_loc[c_id])
            ZeroIndexBef[cur_v:] -= 2
            len_v[cur_v] -= 2
            sel_loc[sel_loc>sel_loc[c_id][1]]-=1
            sel_loc[sel_loc>sel_loc[c_id][0]]-=1
        else:
            sol_2 = np.delete(Solution, sel_loc[c_id])
            best_cost = cost4V(cur_v, Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]], prob) - \
                (cost4V(cur_v, sol_2[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]-2], prob) if 
                  ZeroIndexBef[cur_v]-2 >ZeroIndexBef[cur_v]-len_v[cur_v] else 0 )
            Solution = sol_2
            ZeroIndexBef[cur_v:] -= 2
            len_v[cur_v] -= 2
            sel_loc[sel_loc>sel_loc[c_id][1]]-=1
            sel_loc[sel_loc>sel_loc[c_id][0]]-=1
        costs[cur_v] -= best_cost
        costs[-1] += prob['Cargo'][call-1,3]
# =============================================================================
#         
# =============================================================================
    # print(sel_loc)
    # Solution = np.delete(Solution,sel_loc)
    # ZeroIndexBef = np.array(np.where(Solution == 0)[0], dtype=int)
    # len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    for k in range(rm_size):
        best_costs = np.ones(len(selected_calls))*np.inf
        best_sols = np.empty((len(selected_calls),len(Solution)+2), dtype=int)
        best_vs = np.ones((len(selected_calls)), dtype=int)*prob['n_vehicles']
        ZeroIndex = np.array(np.where(Solution == 0)[0], dtype=int)
        len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
        for s, selected_call in enumerate(selected_calls):
            avail_vehs = np.append(np.where(prob['VesselCargo'][:,selected_call-1] == 1)[0], prob['n_vehicles'])
            # best_costs[s] = prob['Cargo'][selected_call-1,3]
            for v in avail_vehs:
                if v == prob['n_vehicles']:
                    if best_costs[s] >= prob['Cargo'][selected_call-1,3]:
                        best_sols[s] = np.insert(Solution,-1, [selected_call,selected_call])
                    continue
                for pos in range(len_v[v]+1):
                    cost_v = costs[v]
                    new_sol = np.insert(Solution,ZeroIndex[v]+ pos - len_v[v] , selected_call)
                    del_idx, new_cost, feasibility, c = findBestPosForDel(selected_call, v, new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1] ,prob)
                    if not c.startswith('F'):
                        if int(c[-1]) == pos or int(c[-1]) == del_idx:
                            Solution = np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
                            if c.startswith('T'):
                                break
                            else:
                                continue
                        else:
                            Solution = np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
                            continue
                    if new_cost - cost_v < best_costs[s]:
                        best_sols[s] = np.insert(new_sol,ZeroIndex[v]-len_v[v]+ del_idx, selected_call)
                        best_costs[s] = new_cost - cost_v
                        best_vs[s] = v
                    else:
                       Solution = np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
# =============================================================================
#                 if not c.startswith('F'):
#                     if int(c[-1]) == pos or int(c[-1]) == del_idx:
#                         break
# =============================================================================
            # if best_costs[s] == prob['Cargo'][selected_call-1,3]:
            #     best_sols[s] = np.insert(Solution,-1, [selected_call,selected_call])
        best_idx = np.argmin(best_costs)
        Solution = best_sols[best_idx]        
        if best_vs[best_idx] != prob['n_vehicles']:
            costs[best_vs[best_idx]] += best_costs[best_idx]
            costs[-1] -= prob['Cargo'][selected_calls[best_idx]-1,3]
        selected_calls = np.delete(selected_calls, best_idx)
# =============================================================================
#     if (Solution[:-1] == sol).all():
#         Solution, costs = multi_ins_new(sol, costs, prob)
#         Solution = np.append(Solution,[0])    
# =============================================================================
    # print(Solution)
    return Solution [:-1], costs
    

def SA(init_sol, init_cost, probability, operators, cost_func, prob, T_f = 0.1, warm_up = 100):
    
    incumbent = init_sol
    best_sol = init_sol
    cost_incumb = init_cost#cost_func(incumbent, prob)
    costs = np.concatenate((np.zeros(prob['n_vehicles']), [init_cost]))
    best_cost = cost_incumb 
    delta = []
    for w in range(warm_up):
        operator = np.random.choice(operators, replace=True, p=probability )
        new_sol, new_costs = operator(incumbent, costs.copy(), prob)
        new_cost = new_costs.sum()
        delta_E = new_cost - cost_incumb
        feasiblity, c = True, 'Feasible'
        if feasiblity and delta_E < 0:
            incumbent = new_sol
            cost_incumb = new_cost
            costs = new_costs.copy()
            if cost_incumb < best_cost:
                best_sol = incumbent
                best_cost = cost_incumb
                
        elif feasiblity:
            if np.random.uniform() < 0.8:
                incumbent = new_sol
                cost_incumb = new_cost
                costs = new_costs.copy()
            delta.append(delta_E)
    delta_avg = np.mean(delta)
    if delta_avg == 0.0:
        delta_avg = new_cost/warm_up
    T_0 = -delta_avg / np.log(0.8)
    alpha = np.power((T_f/T_0), (1/(10000-warm_up)))
    T = T_0
    # Ts = [T]
    for itr in range(10000-warm_up):
        operator = np.random.choice(operators, replace=True, p=probability )

        new_sol, new_costs = operator(incumbent, costs.copy(), prob)
        new_cost = new_costs.sum()
        delta_E = new_cost - cost_incumb
        feasiblity, c = True, 'Feasible'
        if feasiblity and delta_E < 0:
            incumbent = new_sol
            cost_incumb = new_cost
            costs = new_costs.copy()
            if cost_incumb < best_cost:
                best_sol = incumbent
                best_cost = cost_incumb
                
        elif feasiblity and np.random.uniform() < np.exp(-delta_E/T):
            incumbent = new_sol
            cost_incumb = new_cost
            costs = new_costs.copy()
        T *= alpha
        # Ts.append(T)
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

if __name__ == '__main__':
    problems = [
                # 'Call_7_Vehicle_3',
                # 'Call_18_Vehicle_5',
                # 'Call_35_Vehicle_7',
                # 'Call_80_Vehicle_20',
                # 'Call_130_Vehicle_40',
                'Call_300_Vehicle_90'
                ]
    operators = [
        one_ins_best,
        one_ins_first_best, 
        multi_ins_new
        ]
    probabilities = [
        [1/len(operators) for i in operators],
        [5/12,1/6,5/12]
        ]
    
    repeat = 10
    for j, p in enumerate(problems):
        info = pd.DataFrame(columns=['problem','probability', 'solution', 'average_objective', 'best_objective',
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
                best_sol[i], best_cost[i] = SA(initial_sol, init_cost, prb, operators, cost_function, prob, warm_up = 100)
                # print(best_sol[i],best_cost[i])
            info = info.append({'problem': str(j),
                                'probability': prb,
                                'solution': best_sol[np.argmin(best_cost)],
                                'average_objective': np.mean(best_cost),
                                'best_objective':np.min(best_cost),
                                'improvement': 100*((init_cost-np.min(best_cost))/init_cost),
                                'running_time': (time()-start)/10}, ignore_index=True)
                # print(info)
        info.astype({'average_objective': float,
                            'best_objective':int,
                            'improvement': float,
                            'running_time': float}).set_index('problem').to_csv('results'+str(5)+'.csv')
# =============================================================================
# np.random.seed(23)
# prob = load_problem( "..//..//Data//" +'Call_7_Vehicle_3'+ ".txt")
# # # # # findBestPosForDel(2,1,np.array([2, 9, 9]),prob)
# # sol = np.array([4, 4, 2, 2, 0, 7, 7, 0, 1, 5, 5, 3, 3, 1, 0, 6, 6 ])
# sol = np.array([0]*prob['n_vehicles'] + [i for i in range(1,prob['n_calls']+1) for j in range(2)])
# init_cost = cost_function(sol, prob)
# # a = one_ins_best(sol, prob)
# start = time()
# # a = SA(sol,init_cost, 
# #         [0,0,1],
# #        # [1/3,1/3,1/3],
# #        [
# #         one_ins_best,
# #                 one_ins_first_best, 
# #                 multi_ins_new], cost_function, prob)
# # # for i in range(288):
# # #     np.random.randint(1)
# b = []
# for i in range(10):
#     b.append(SA(sol,init_cost, [1/3,1/3,1/3],[
#             one_ins_best,
#                     one_ins_first_best, 
#                     multi_ins_new], cost_function, prob))
# print(time()-start)
# # two_ex(sol, prob)
# 
# =============================================================================
