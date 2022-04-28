# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:40:15 2022

@author: erfan
"""
from auxiliary_functions import *
import numpy as np
from itertools import combinations

def one_ins_first_better(sol, costs, features, rng, prob):
    Solution = sol + [0]
    selected_call = rng.choice(np.arange(1,prob['n_calls']+1))
    call_locs = np.where(Solution==selected_call)[0]
    ZeroIndexBef = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = len(ZeroIndexBef[ZeroIndexBef<call_locs[0]])
    best_sol = Solution.copy()
    best_features = copy_features(features)
    if cur_v == prob['n_vehicles']:
        cur_cost = prob['Cargo'][selected_call-1,3]
    else:
        sol_2 = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:]
        cur_cost = cur_cost4V(cur_v, Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]],
                           sol_2[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]-2],
                           *(call_locs-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))),features[2],prob)
     
        features = features_delete(features, cur_v, *(call_locs-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))))   
    costs[cur_v] -= cur_cost
    costs[-1] += prob['Cargo'][selected_call-1,3]
    Solution = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:]
    ZeroIndex = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    avail_vehs = [i for i in range(prob['n_vehicles']) if prob['VesselCargo'][i,selected_call-1] == 1]+ [prob['n_vehicles']]
    rng.shuffle(avail_vehs)
    len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
    best_v = cur_v 
    best_cost = cur_cost.copy()
    for v in avail_vehs:
        if v == prob['n_vehicles']:
            if best_cost >= prob['Cargo'][selected_call-1,3]:
                best_sol = Solution[:-1] + [selected_call,selected_call] + Solution[-1:]
                break
            continue
        for pos in range(len_v[v]+1):
            cost_v = costs[v]
            new_sol = Solution[:ZeroIndex[v]+ pos - len_v[v]] +[selected_call]+ Solution[ZeroIndex[v]+ pos - len_v[v]:]
            
            del_idx, new_cost, feasibility, c, new_features = \
                findBestPosForDel(selected_call, v, new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1],
                                  pos, copy_features(features), prob)
            if not c.startswith('F'):
                if int(c[-1]) == pos or int(c[-1]) == del_idx:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    if c.startswith('T'):
                        break
                    else:
                        continue
                else:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    continue
            
            elif new_cost - cost_v < best_cost:
                best_sol = new_sol[:ZeroIndex[v]-len_v[v]+ del_idx] +[selected_call]+ new_sol[ZeroIndex[v]-len_v[v]+ del_idx:]
                
                best_cost = new_cost - cost_v
                best_v = v
                best_features = new_features
                break
            else:
               Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
        if best_v != prob['n_vehicles']:
            break
    if best_cost == cur_cost:
        costs[best_v] += best_cost
        costs[-1] -= prob['Cargo'][selected_call-1,3]
        return sol, costs, best_features
    if best_v != prob['n_vehicles']:
        costs[best_v] += best_cost
        costs[-1] -= prob['Cargo'][selected_call-1,3]
    return best_sol [:-1], costs, best_features

def one_ins_best(sol, costs, features, rng, prob):
    Solution = sol + [0]
    selected_call = rng.choice(np.arange(1,prob['n_calls']+1))
    call_locs = np.where(Solution==selected_call)[0]
    ZeroIndexBef = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = len(ZeroIndexBef[ZeroIndexBef<call_locs[0]])
    best_sol = Solution.copy()
    best_features = copy_features(features)
    if cur_v != prob['n_vehicles']:
        sol_2 = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:]
        cur_cost =  cur_cost4V(cur_v, Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]],
                              sol_2[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]-2],
                              *(call_locs-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))),features[2],prob) 
        
        features = features_delete(features, cur_v, *(call_locs-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))))
        costs[cur_v] -= cur_cost
        costs[-1] += prob['Cargo'][selected_call-1,3]

    Solution = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:] 
    ZeroIndex = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    avail_vehs = [i for i in range(prob['n_vehicles']) if prob['VesselCargo'][i,selected_call-1] == 1]+ [prob['n_vehicles']]
    len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
    best_v = prob['n_vehicles']
    best_cost = np.inf
    for v in avail_vehs:
        if v == prob['n_vehicles']:
            if best_cost >= prob['Cargo'][selected_call-1,3]:
                best_sol = Solution[:-1] + [selected_call,selected_call] + Solution[-1:] 
            continue
        for pos in range(len_v[v]+1):
            cost_v = costs[v]
            new_sol = Solution[:ZeroIndex[v]+ pos - len_v[v]] +[selected_call]+ Solution[ZeroIndex[v]+ pos - len_v[v]:]
            
            del_idx, new_cost, feasibility, c, new_features = \
                findBestPosForDel(selected_call, v, new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1],
                                  pos, copy_features(features), prob)
            if not c.startswith('F'):
                if int(c[-1]) == pos or int(c[-1]) == del_idx:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    # continue
                    if c.startswith('T'):
                        break
                    else:
                        continue
                else:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    continue
                
            elif new_cost - cost_v <= best_cost:
                best_sol = new_sol[:ZeroIndex[v]-len_v[v]+ del_idx] +[selected_call]+ new_sol[ZeroIndex[v]-len_v[v]+ del_idx:]
                best_cost = new_cost - cost_v
                best_v = v
                best_features = new_features
            else:
               Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]

    
    if best_v != prob['n_vehicles']:
        costs[best_v] += best_cost
        costs[-1] -= prob['Cargo'][selected_call-1,3]
    return best_sol [:-1], costs, best_features


def multi_ins_rand(sol, costs, features, rng, prob):
    Solution = sol + [0]
    rm_size = rng.choice(np.arange(2,5))
    selected_calls = rng.choice(np.arange(1,prob['n_calls']+1),
                                     size = rm_size,
                                     replace = False)

    sel_loc = np.array([np.where(Solution == selected_calls[i])[0] for i in range(rm_size)])
    ZeroIndexBef = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = []
    for c_id, call in enumerate(selected_calls):
        cur_v.append(len(ZeroIndexBef[ZeroIndexBef<sel_loc[c_id][0]]))
        if cur_v[c_id] == prob['n_vehicles']:
            Solution = Solution[:sel_loc[c_id][0]] + Solution[sel_loc[c_id][0]+1:sel_loc[c_id][1]] + Solution[sel_loc[c_id][1]+1:]
            ZeroIndexBef[cur_v[c_id]:] -= 2
            len_v[cur_v[c_id]] -= 2
            sel_loc[sel_loc>sel_loc[c_id][1]]-=1
            sel_loc[sel_loc>sel_loc[c_id][0]]-=1
            sel_loc[c_id][1] += 1
        else:
            sol_2 = Solution[:sel_loc[c_id][0]] + Solution[sel_loc[c_id][0]+1:sel_loc[c_id][1]] + Solution[sel_loc[c_id][1]+1:]
            best_cost = cur_cost4V(cur_v[c_id], Solution[ZeroIndexBef[cur_v[c_id]]-len_v[cur_v[c_id]]:ZeroIndexBef[cur_v[c_id]]],
                                  sol_2[ZeroIndexBef[cur_v[c_id]]-len_v[cur_v[c_id]]:ZeroIndexBef[cur_v[c_id]]-2],
                                  *(sel_loc[c_id]-(sum(len_v[:cur_v[c_id]])+ len(len_v[:cur_v[c_id]]))),features[2],prob)
            
            Solution = sol_2
            ZeroIndexBef[cur_v[c_id]:] -= 2
            len_v[cur_v[c_id]] -= 2
            sel_loc[sel_loc>sel_loc[c_id][1]]-=1
            sel_loc[sel_loc>sel_loc[c_id][0]]-=1
            sel_loc[c_id][1] += 1
            features = features_delete(features, cur_v[c_id], *(sel_loc[c_id]-(sum(len_v[:cur_v[c_id]])+ len(len_v[:cur_v[c_id]]))))
            costs[cur_v[c_id]] -= best_cost
            costs[-1] += prob['Cargo'][call-1,3]
    rng.shuffle(selected_calls)
    for selected_call in selected_calls:
        avail_vehs = [i for i in range(prob['n_vehicles']) if prob['VesselCargo'][i,selected_call-1] == 1]+ [prob['n_vehicles']]
        best_features = copy_features(features)
        best_v = prob['n_vehicles']
        best_cost = np.inf
        for v in avail_vehs:
            if v == prob['n_vehicles']:
                if best_cost >= prob['Cargo'][selected_call-1,3]:
                    best_sol = Solution[:-1] + [selected_call,selected_call] + Solution[-1:]
                    best_v = v
                    best_cost = prob['Cargo'][selected_call-1,3]
                    best_features = copy_features(features)
                continue
            for pos in range(len_v[v]+1):
                cost_v = costs[v]
                new_sol = Solution[:ZeroIndexBef[v]+ pos - len_v[v]] +[selected_call]+ Solution[ZeroIndexBef[v]+ pos - len_v[v]:]
                
                del_idx, new_cost, feasibility, c, new_features = \
                    findBestPosForDel(selected_call, v, new_sol[ZeroIndexBef[v]-len_v[v]:ZeroIndexBef[v]+1],
                                      pos, copy_features(features), prob)
                if not c.startswith('F'):
                    if int(c[-1]) == pos or int(c[-1]) == del_idx:
                        Solution = new_sol[:ZeroIndexBef[v]+ pos - len_v[v]] + new_sol[ZeroIndexBef[v]+ pos - len_v[v]+1:]
                        if c.startswith('T'):
                            break
                        else:
                            continue
                    else:
                        Solution = new_sol[:ZeroIndexBef[v]+ pos - len_v[v]] + new_sol[ZeroIndexBef[v]+ pos - len_v[v]+1:]
                        continue
                    
                elif new_cost - cost_v <= best_cost:
                    best_sol = new_sol[:ZeroIndexBef[v]-len_v[v]+ del_idx] +[selected_call]+ new_sol[ZeroIndexBef[v]-len_v[v]+ del_idx:]
                    best_cost = new_cost - cost_v
                    best_v = v
                    best_features = new_features
                else:
                   Solution = new_sol[:ZeroIndexBef[v]+ pos - len_v[v]] + new_sol[ZeroIndexBef[v]+ pos - len_v[v]+1:]

        Solution = best_sol
        features = copy_features(best_features)
        ZeroIndexBef[best_v:] += 2
        len_v[best_v] += 2
        if best_v != prob['n_vehicles']:
            costs[best_v] += best_cost
            costs[-1] -= prob['Cargo'][selected_call-1,3]
    
    return Solution [:-1], costs, features
