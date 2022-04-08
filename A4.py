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
    
    Vcalls = [Vcalls[i]-1 for i in range(len(Vcalls))]#Vcalls - 1
    
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

def cap_TW_4V(Vnum, Vcalls, features, p_ind, d_ind, problem): #Capacity and timewindow check for a vehicle
    FirstTravelCost = problem['FirstTravelCost']
    TravelCost = problem['TravelCost']
    PortCost = problem['PortCost']
    VesselCapacity = problem['VesselCapacity']
    Cargo = problem['Cargo']
    UnloadingTime = problem['UnloadingTime']
    LoadingTime = problem['LoadingTime']
    TravelTime = problem['TravelTime']
    FirstTravelTime = problem['FirstTravelTime']

    Vcalls = [Vcalls[i]-1 for i in range(len(Vcalls))] #Vcalls - 1
    call = Vcalls[p_ind]
    
    NoDoubleCallOnVehicle = len(Vcalls)
    LoadSize1, Timewindows1, PortIndex1, LU_Time1 = features
    
    currentTime = 0
    
    LoadSize1[Vnum][p_ind] = Cargo[call, 2]
    LoadSize1[Vnum][d_ind] = -Cargo[call, 2]
    
    Timewindows1[Vnum][0][p_ind] = Cargo[call, 4]
    Timewindows1[Vnum][0][d_ind] = Cargo[call, 6]
    Timewindows1[Vnum][1][p_ind] = Cargo[call, 5]
    Timewindows1[Vnum][1][d_ind] = Cargo[call, 7]
    
    PortIndex1[Vnum][p_ind] = Cargo[call, 0].astype(int) - 1
    PortIndex1[Vnum][d_ind] = Cargo[call, 1].astype(int) - 1 
    
    LU_Time1[Vnum][p_ind] = LoadingTime[Vnum, call]
    LU_Time1[Vnum][d_ind] = UnloadingTime[Vnum, call]
    
    features = [LoadSize1, Timewindows1, PortIndex1, LU_Time1]
    
    Diag_cost = TravelCost[Vnum, PortIndex1[Vnum][:-1], PortIndex1[Vnum][1:]]
    
    FirstVisitCost = FirstTravelCost[Vnum, int(Cargo[Vcalls[0], 0] - 1)]
    RouteTravelCost = np.sum(np.hstack((FirstVisitCost, Diag_cost.flatten())))
    CostInPorts = np.sum(PortCost[Vnum, Vcalls]) / 2

    Diag = TravelTime[Vnum, PortIndex1[Vnum][:-1], PortIndex1[Vnum][1:]]
    FirstVisitTime = FirstTravelTime[Vnum, int(Cargo[Vcalls[0], 0] - 1)]
    
    RouteTravelTime = np.hstack((FirstVisitTime, Diag.flatten()))
    
    ArriveTime = np.zeros(NoDoubleCallOnVehicle)
    for j in range(NoDoubleCallOnVehicle):
        if VesselCapacity[Vnum] - np.cumsum(LoadSize1[Vnum])[j] < 0:
            return False, 'Capacity exceeded at call {}'.format(j), np.inf, features
        ArriveTime[j] = np.max((currentTime + RouteTravelTime[j], Timewindows1[Vnum][0][j]))
        if ArriveTime[j] > Timewindows1[Vnum][1][j]:
            return False, 'Time window exceeded at call {}'.format(j), np.inf, features
        currentTime = ArriveTime[j] + LU_Time1[Vnum][j]
    return True, 'Feasible', CostInPorts + RouteTravelCost, features

def update_features_v(features, v, new_features):
    LoadSize, Timewindows, PortIndex, LU_Time = features
    new_LoadSize, new_Timewindows, new_PortIndex, new_LU_Time = new_features
    LoadSize[v] = new_LoadSize[v]
    Timewindows[v] = new_Timewindows[v]
    PortIndex[v] = new_PortIndex[v]
    LU_Time[v] = new_LU_Time[v]
    return [LoadSize, Timewindows, PortIndex, LU_Time ]

def findBestPosForDel(call, Vnum, Vcalls, pos, features, prob):
    PickIndex = pos
    best_cost = np.inf
    i_best = 0
    for i in range(PickIndex+1, len(Vcalls)+1):
        new_sol = Vcalls[:i] + [call] + Vcalls[i:] #np.insert(Vcalls, i, call)
        updated_features = features_insert(deepcopy(features), Vnum, PickIndex, i)
        new_feasibility, new_c, new_cost, new_features = cap_TW_4V(Vnum, new_sol, updated_features, PickIndex, i, prob)
        if new_c=='Feasible':
            c = new_c
            feasibility = new_feasibility
            best_features = update_features_v(deepcopy(features), Vnum, new_features)
            if i == len(Vcalls):
                return i, new_cost, feasibility, c, best_features
            if new_cost < best_cost:
                # best_sol = new_sol
                best_cost = new_cost
                i_best = i
        else:
            if (int(new_c[-1]) == PickIndex or int(new_c[-1]) == i) or \
                (int(new_c[-1]) > PickIndex and int(new_c[-1]) < i):
                break
    try:
        return i_best, best_cost, feasibility, c, best_features
    except:
        return i_best, best_cost, new_feasibility, new_c, new_features

def one_ins_first_best(sol, costs, features, prob):
    Solution = sol + [0]#np.append(sol, [0])
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
    ZeroIndexBef = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = len(ZeroIndexBef[ZeroIndexBef<call_locs[0]])
    if cur_v == prob['n_vehicles']:
        cur_cost = prob['Cargo'][selected_call-1,3]
    else:
        sol_2 = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:]#np.delete(Solution, call_locs)
        cur_cost = cost4V(cur_v, Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]], prob) - \
            (cost4V(cur_v, sol_2[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]-2], prob) if 
             ZeroIndexBef[cur_v]-2 >ZeroIndexBef[cur_v]-len_v[cur_v] else 0 )
        features = features_delete(features, cur_v, *(call_locs-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))))
    # best_cost = np.inf    
    costs[cur_v] -= cur_cost
    costs[-1] += prob['Cargo'][selected_call-1,3]
    best_sol = Solution
    best_features = features
    Solution = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:]#np.delete(Solution, call_locs)
    ZeroIndex = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    avail_vehs = [i for i in range(prob['n_vehicles']) if prob['VesselCargo'][i,selected_call-1] == 1]+ [prob['n_vehicles']]#np.append(np.where(prob['VesselCargo'][:,selected_call-1] == 1)[0], prob['n_vehicles'])
    len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
    best_v = prob['n_vehicles']
    best_cost = np.inf
    for v in avail_vehs:
        if v == prob['n_vehicles']:
            if best_cost >= prob['Cargo'][selected_call-1,3]:
                best_sol = Solution[:-1] + [selected_call,selected_call] + Solution[-1:] #np.insert(Solution,-1, [selected_call,selected_call])
                break
            continue
        for pos in range(len_v[v]+1):
            cost_v = costs[v]
            new_sol = Solution[:ZeroIndex[v]+ pos - len_v[v]] +[selected_call]+ Solution[ZeroIndex[v]+ pos - len_v[v]:]
            #np.insert(Solution, ZeroIndex[v]+ pos - len_v[v]], selected_call)
# =============================================================================
#             
# =============================================================================
            del_idx, new_cost, feasibility, c, new_features = \
                findBestPosForDel(selected_call, v, new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1],
                                  pos, features.copy(), prob)
            if not c.startswith('F'):
                if int(c[-1]) == pos or int(c[-1]) == del_idx:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    #np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
                    if c.startswith('T'):
                        break
                    else:
                        continue
                else:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    #np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
                    continue
            if new_cost - cost_v <= best_cost:
                best_sol = new_sol[:ZeroIndex[v]-len_v[v]+ del_idx] +[selected_call]+ new_sol[ZeroIndex[v]-len_v[v]+ del_idx:]
                #np.insert(new_sol,ZeroIndex[v]-len_v[v]+ del_idx, selected_call)
                best_cost = new_cost - cost_v
                best_v = v
                best_features = new_features
                break
            else:
               Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
               #np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
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
    return best_sol [:-1], costs, best_features

def one_ins_best(sol, costs, features, prob):
    Solution = sol + [0] #np.append(sol, [0])
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
    ZeroIndexBef = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = len(ZeroIndexBef[ZeroIndexBef<call_locs[0]])
    if cur_v == prob['n_vehicles']:
        cur_cost = prob['Cargo'][selected_call-1,3]
    else:
        sol_2 = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:]#np.delete(Solution, call_locs)
        cur_cost = cost4V(cur_v, Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]], prob) - \
            (cost4V(cur_v, sol_2[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]-2], prob) if 
              ZeroIndexBef[cur_v]-2 >ZeroIndexBef[cur_v]-len_v[cur_v] else 0 )
        features = features_delete(features, cur_v, *(call_locs-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))))
    # best_cost = np.inf    
    costs[cur_v] -= cur_cost
    costs[-1] += prob['Cargo'][selected_call-1,3]
    best_sol = Solution
    best_features = features
    Solution = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:] #np.delete(Solution, call_locs)
    ZeroIndex = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    avail_vehs = [i for i in range(prob['n_vehicles']) if prob['VesselCargo'][i,selected_call-1] == 1]+ [prob['n_vehicles']]#np.append(np.where(prob['VesselCargo'][:,selected_call-1] == 1)[0], prob['n_vehicles'])
    len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
    best_v = prob['n_vehicles']
    best_cost = np.inf
    for v in avail_vehs:
        if v == prob['n_vehicles']:
            if best_cost >= prob['Cargo'][selected_call-1,3]:
                best_sol = Solution[:-1] + [selected_call,selected_call] + Solution[-1:] #np.insert(Solution,-1, [selected_call,selected_call])
            continue
        for pos in range(len_v[v]+1):
# =============================================================================
#             if len_v[v] == 0:
#                 cost_v = 0
#             else:
#                 cost_v = cost4V(v, Solution[ZeroIndex[v]-len_v[v]:ZeroIndex[v]], prob)
# =============================================================================
            cost_v = costs[v]
            new_sol = Solution[:ZeroIndex[v]+ pos - len_v[v]] +[selected_call]+ Solution[ZeroIndex[v]+ pos - len_v[v]:]
            #np.insert(Solution,ZeroIndex[v]+ pos - len_v[v] , selected_call)
            
            del_idx, new_cost, feasibility, c, new_features = \
                findBestPosForDel(selected_call, v, new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1],
                                  pos, features.copy(), prob)
            if not c.startswith('F'):
                if int(c[-1]) == pos or int(c[-1]) == del_idx:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    #np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
                    if c.startswith('T'):
                        break
                    else:
                        continue
                else:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    #np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
                    continue
                
            if new_cost - cost_v <= best_cost:
                best_sol = new_sol[:ZeroIndex[v]-len_v[v]+ del_idx] +[selected_call]+ new_sol[ZeroIndex[v]-len_v[v]+ del_idx:]
                #np.insert(new_sol,ZeroIndex[v]-len_v[v]+ del_idx, selected_call)
                best_cost = new_cost - cost_v
                best_v = v
                best_features = new_features
            else:
               Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
               #np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
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
    return best_sol [:-1], costs, best_features

def features_delete(features, cur_v, p_ind, d_ind):
    LoadSize, Timewindows, PortIndex, LU_Time = features
    LoadSize[cur_v] = LoadSize[cur_v][:p_ind] + LoadSize[cur_v][p_ind+1:d_ind] + LoadSize[cur_v][d_ind+1:]#np.delete(LoadSize[cur_v],[p_ind, d_ind])
    Timewindows[cur_v][0] = Timewindows[cur_v][0][:p_ind] + Timewindows[cur_v][0][p_ind+1:d_ind] + Timewindows[cur_v][0][d_ind+1:] #np.delete(Timewindows[cur_v],[p_ind, d_ind], axis = 1)
    Timewindows[cur_v][1] = Timewindows[cur_v][1][:p_ind] + Timewindows[cur_v][1][p_ind+1:d_ind] + Timewindows[cur_v][1][d_ind+1:]
    
    PortIndex[cur_v] = PortIndex[cur_v][:p_ind] + PortIndex[cur_v][p_ind+1:d_ind] + PortIndex[cur_v][d_ind+1:] #np.delete(PortIndex[cur_v],[p_ind, d_ind])
    LU_Time[cur_v] = LU_Time[cur_v][:p_ind] + LU_Time[cur_v][p_ind+1:d_ind] + LU_Time[cur_v][d_ind+1:] #np.delete(LU_Time[cur_v],[p_ind, d_ind])
    return [LoadSize, Timewindows, PortIndex, LU_Time]

def features_insert(features, cur_v, p_ind, d_ind):
    LoadSize, Timewindows, PortIndex, LU_Time = features
    LoadSize[cur_v] = LoadSize[cur_v][:p_ind] + [0] + LoadSize[cur_v][p_ind:d_ind] + [0] + LoadSize[cur_v][d_ind:]#np.insert(LoadSize[cur_v],[p_ind, d_ind-1], 0)
    Timewindows[cur_v][0] = Timewindows[cur_v][0][:p_ind] + [0] + Timewindows[cur_v][0][p_ind:d_ind] + [0] + Timewindows[cur_v][0][d_ind:] #np.insert(Timewindows[cur_v],[p_ind, d_ind-1], np.zeros(2), axis = 1)
    Timewindows[cur_v][1] = Timewindows[cur_v][1][:p_ind] + [0] + Timewindows[cur_v][1][p_ind:d_ind] + [0] + Timewindows[cur_v][1][d_ind:]
    PortIndex[cur_v] = PortIndex[cur_v][:p_ind] + [0] + PortIndex[cur_v][p_ind:d_ind] + [0] + PortIndex[cur_v][d_ind:] #np.insert(PortIndex[cur_v],[p_ind, d_ind-1], 0)
    LU_Time[cur_v] = LU_Time[cur_v][:p_ind] + [0] + LU_Time[cur_v][p_ind:d_ind] + [0] + LU_Time[cur_v][d_ind:] #np.insert(LU_Time[cur_v],[p_ind, d_ind-1], 0)
    return [LoadSize, Timewindows, PortIndex, LU_Time]

def multi_ins_new(sol, costs, features, prob, rm_size = 3):
    Solution = sol + [0] #np.append(sol, [0])
    
    # selection_prob = np.sum(prob['VesselCargo'], axis = 0)/np.sum(prob['VesselCargo'])
    selected_calls = np.random.choice(np.arange(1,prob['n_calls']+1),
                                     size = rm_size,
                                     replace = False,
                                      # p = selection_prob
                                     )
    sel_loc = np.array([np.where(Solution == selected_calls[i])[0] for i in range(rm_size)])
    ZeroIndexBef = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    for c_id, call in enumerate(selected_calls):
        # call_locs = np.where(Solution==call)[0]
        # ZeroIndexBef = np.array(np.where(Solution == 0)[0], dtype=int)
        # len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
        cur_v = len(ZeroIndexBef[ZeroIndexBef<sel_loc[c_id][0]])
        #cur_v = len(ZeroIndexBef[ZeroIndexBef<call_locs[0]])
        if cur_v == prob['n_vehicles']:
            best_cost = prob['Cargo'][call-1,3]
            Solution = Solution[:sel_loc[c_id][0]] + Solution[sel_loc[c_id][0]+1:sel_loc[c_id][1]] + Solution[sel_loc[c_id][1]+1:]#np.delete(Solution, sel_loc[c_id])
            ZeroIndexBef[cur_v:] -= 2
            len_v[cur_v] -= 2
            sel_loc[sel_loc>sel_loc[c_id][1]]-=1
            sel_loc[sel_loc>sel_loc[c_id][0]]-=1
        else:
            sol_2 = Solution[:sel_loc[c_id][0]] + Solution[sel_loc[c_id][0]+1:sel_loc[c_id][1]] + Solution[sel_loc[c_id][1]+1:]#np.delete(Solution, sel_loc[c_id])
            best_cost = cost4V(cur_v, Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]], prob) - \
                (cost4V(cur_v, sol_2[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]-2], prob) if 
                  ZeroIndexBef[cur_v]-2 >ZeroIndexBef[cur_v]-len_v[cur_v] else 0 )
            Solution = sol_2
            ZeroIndexBef[cur_v:] -= 2
            len_v[cur_v] -= 2
            sel_loc[sel_loc>sel_loc[c_id][1]]-=1
            sel_loc[sel_loc>sel_loc[c_id][0]]-=1
            features = features_delete(features, cur_v, *(sel_loc[c_id]-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))))
        costs[cur_v] -= best_cost
        costs[-1] += prob['Cargo'][call-1,3]
    # print(sel_loc)
    # Solution = np.delete(Solution,sel_loc)
    # ZeroIndexBef = np.array(np.where(Solution == 0)[0], dtype=int)
    # len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    for k in range(rm_size):
        best_costs = [np.inf for i in range(len(selected_calls))]#np.ones(len(selected_calls))*np.inf
        best_sols = [[] for i in range(len(selected_calls)) ]#np.empty((len(selected_calls),len(Solution)+2), dtype=int)
        best_vs = [[] for i in range(prob['n_vehicles'])]#np.ones((len(selected_calls)), dtype=int)*prob['n_vehicles']
        best_features = [features.copy() for i in range(len(selected_calls))]
        ZeroIndex = np.array(np.where(Solution == np.array(0))[0], dtype=int)
        len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
        for s, selected_call in enumerate(selected_calls):
            avail_vehs = [i for i in range(prob['n_vehicles']) if prob['VesselCargo'][i,selected_call-1] == 1]+ [prob['n_vehicles']]#np.append(np.where(prob['VesselCargo'][:,selected_call-1] == 1)[0], prob['n_vehicles'])
            # best_costs[s] = prob['Cargo'][selected_call-1,3]
            for v in avail_vehs:
                if v == prob['n_vehicles']:
                    if best_costs[s] >= prob['Cargo'][selected_call-1,3]:
                        best_sols[s] = Solution[:-1] + [selected_call,selected_call] + Solution[-1:]#np.insert(Solution,-1, [selected_call,selected_call])
                        best_vs[s] = v
                    continue
                for pos in range(len_v[v]+1):
                    cost_v = costs[v]
                    new_sol = Solution[:ZeroIndex[v]+ pos - len_v[v]] +[selected_call]+ Solution[ZeroIndex[v]+ pos - len_v[v]:]
                    #np.insert(Solution,ZeroIndex[v]+ pos - len_v[v] , selected_call)
                    del_idx, new_cost, feasibility, c, new_features = \
                        findBestPosForDel(selected_call, v, 
                                          new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1],
                                          pos, features.copy(), prob)
                    if not c.startswith('F'):
                        if int(c[-1]) == pos or int(c[-1]) == del_idx:
                            Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                            #np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
                            if c.startswith('T'):
                                break
                            else:
                                continue
                        else:
                            Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                            #np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
                            continue
                    if new_cost - cost_v < best_costs[s]:
                        best_sols[s] = new_sol[:ZeroIndex[v]-len_v[v]+ del_idx] +[selected_call]+ new_sol[ZeroIndex[v]-len_v[v]+ del_idx:]
                        #np.insert(new_sol,ZeroIndex[v]-len_v[v]+ del_idx, selected_call)
                        best_costs[s] = new_cost - cost_v
                        best_vs[s] = v
                        best_features[s] = new_features
                    else:
                       Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                       #np.delete(new_sol, ZeroIndex[v]+ pos - len_v[v])
        best_idx = np.argmin(best_costs)
        Solution = best_sols[best_idx]
        features = best_features[best_idx].copy()
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
    return Solution [:-1], costs, features
    

def SA(init_sol, init_cost, probability, operators, prob, T_f = 0.1, warm_up = 100):
    
    incumbent = init_sol
    best_sol = init_sol
    cost_incumb = init_cost#cost_func(incumbent, prob)
    costs = [0 for i in range(prob['n_vehicles'])]+ [init_cost]
    LoadSize = [[] for i in range(prob['n_vehicles'])]
    Timewindows = [[[],[]] for i in range(prob['n_vehicles'])]
    PortIndex = [[] for i in range(prob['n_vehicles'])]
    LU_Time = [[] for i in range(prob['n_vehicles'])]
# =============================================================================
#     LoadSize = {i:np.array([]) for i in range(prob['n_vehicles'])}
#     Timewindows = {i:np.array([[],[]]) for i in range(prob['n_vehicles'])}
#     PortIndex = {i:np.array([],dtype=int) for i in range(prob['n_vehicles'])}
#     LU_Time = {i:np.array([]) for i in range(prob['n_vehicles'])}
# =============================================================================
    features = [LoadSize, Timewindows, PortIndex, LU_Time]
    best_cost = cost_incumb 
    delta = []
    for w in range(warm_up):
# =============================================================================
#         if w == 6:
#             print(w)
# =============================================================================
        operator = np.random.choice(operators, replace=True, p=probability )
        new_sol, new_costs, new_features = operator(incumbent, deepcopy(costs), deepcopy(features), prob)
        new_cost = sum(new_costs)
        delta_E = new_cost - cost_incumb
        feasiblity, c = True, 'Feasible'
        if feasiblity and delta_E < 0:
            incumbent = new_sol
            cost_incumb = new_cost
            costs = deepcopy(new_costs)
            features = deepcopy(new_features)
            if cost_incumb < best_cost:
                best_sol = incumbent
                best_cost = cost_incumb
                
        elif feasiblity:
            if np.random.uniform() < 0.8:
                incumbent = new_sol
                cost_incumb = new_cost
                costs = deepcopy(new_costs)
                features = deepcopy(new_features)
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
        
        new_sol, new_costs, new_features = operator(incumbent, deepcopy(costs), deepcopy(features), prob)
        new_cost = sum(new_costs)
        delta_E = new_cost - cost_incumb
        feasiblity, c = True, 'Feasible'
        if feasiblity and delta_E < 0:
            incumbent = new_sol
            cost_incumb = new_cost
            costs = deepcopy(new_costs)
            features = deepcopy(new_features)
            if cost_incumb < best_cost:
                best_sol = incumbent
                best_cost = cost_incumb
                
        elif feasiblity and np.random.uniform() < np.exp(-delta_E/T):
            incumbent = new_sol
            cost_incumb = new_cost
            costs = deepcopy(new_costs)
            features = deepcopy(new_features)
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

# =============================================================================
# if __name__ == '__main__':
#     problems = [
#                 # 'Call_7_Vehicle_3',
#                 # 'Call_18_Vehicle_5',
#                 # 'Call_35_Vehicle_7',
#                 # 'Call_80_Vehicle_20',
#                 'Call_130_Vehicle_40',
#                 # 'Call_300_Vehicle_90'
#                 ]
#     operators = [
#         one_ins_best,
#         one_ins_first_best, 
#         multi_ins_new
#         ]
#     probabilities = [
#         [1/len(operators) for i in operators],
#         # [5/12,1/6,5/12]
#         ]
#     
#     repeat = 1
#     for j, p in enumerate(problems):
#         info = pd.DataFrame(columns=['problem','probability', 'solution', 'average_objective', 'best_objective',
#                                       'improvement', 'running_time'])
#         for prb in probabilities:
#             start = time()
# # =============================================================================
# #             update this later before submission
# # =============================================================================
#             prob = load_problem( "..//..//Data//" +p+ ".txt")
#             initial_sol = np.array([0]*prob['n_vehicles'] + [i for i in range(1,prob['n_calls']+1) for j in range(2)])
# # =============================================================================
# #             calls_dic = {i:{'dummy_cost':prob['Cargo'][i-1,3],
# #                             'cur_v':prob['n_vehicles'],
# #                             'cur_cost':prob['Cargo'][i-1,3]}  for i in range(1,prob['n_calls']+1)}
# # =============================================================================
#             init_cost = cost_function(initial_sol, prob)
#             best_sol, best_cost = np.empty((repeat ,len(initial_sol)),dtype=int), np.empty(repeat )
#             for i in range(repeat ):
#                 np.random.seed(23+i)
#                 # print('seed', 23+i)
#                 best_sol[i], best_cost[i] = SA(initial_sol, init_cost, prb, operators, prob, warm_up = 100)
#                 # print(best_sol[i],best_cost[i])
#             info = info.append({'problem': str(j),
#                                 'probability': prb,
#                                 'solution': best_sol[np.argmin(best_cost)],
#                                 'average_objective': np.mean(best_cost),
#                                 'best_objective':np.min(best_cost),
#                                 'improvement': 100*((init_cost-np.min(best_cost))/init_cost),
#                                 'running_time': (time()-start)/repeat}, ignore_index=True)
#                 # print(info)
#         info.astype({'average_objective': float,
#                             'best_objective':int,
#                             'improvement': float,
#                             'running_time': float}).set_index('problem').to_csv('results'+str(5)+'.csv')
# =============================================================================
np.random.seed(23)
prob = load_problem( "..//..//Data//" +'Call_7_Vehicle_3'+ ".txt")
# # # # findBestPosForDel(2,1,np.array([2, 9, 9]),prob)
# sol = np.array([4, 4, 2, 2, 0, 7, 7, 0, 1, 5, 5, 3, 3, 1, 0, 6, 6 ])
sol = [0]*prob['n_vehicles'] + [i for i in range(1,prob['n_calls']+1) for j in range(2)]
init_cost = prob['Cargo'][:,3].sum()#cost_function(sol, prob)
# a = one_ins_best(sol, prob)
start = time()
a = SA(sol,init_cost, 
        # [1/2,1/2,0],
        [1/3,1/3,1/3],
        [
        one_ins_best,
                one_ins_first_best, 
                multi_ins_new], prob)
# # for i in range(288):
# #     np.random.randint(1)
# b = []
# for i in range(10):
#     b.append(SA(sol,init_cost, [1/3,1/3,1/3],[
#             one_ins_best,
#                     one_ins_first_best, 
#                     multi_ins_new], cost_function, prob))
print(time()-start)
# two_ex(sol, prob)


# =============================================================================
# 
# =============================================================================
