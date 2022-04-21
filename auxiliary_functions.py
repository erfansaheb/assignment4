# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:31:13 2022

@author: erfan
"""
import numpy as np

def cost4V(Vnum, Vcalls, problem):
    Cargo = problem['Cargo']
    FirstTravelCost = problem['FirstTravelCost']
    TravelCost = problem['TravelCost']
    PortCost = problem['PortCost']
    
    Vcalls = [Vcalls[i]-1 for i in range(len(Vcalls))]
    
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

def cur_cost4V(Vnum, Vcalls, Vcalls_aft, p_ind, d_ind, PortInd, problem): 
    Cargo = problem['Cargo']
    FirstTravelCost = problem['FirstTravelCost']
    TravelCost = problem['TravelCost']
    PortCost = problem['PortCost']
    Vcalls = [Vcalls[i]-1 for i in range(len(Vcalls))]
    Vcalls_aft = [Vcalls_aft[i]-1 for i in range(len(Vcalls_aft))]
    
    if len(Vcalls) == 2:
        FirstVisitCost = FirstTravelCost[Vnum, int(Cargo[Vcalls[0], 0] - 1)]
        RouteTravelCost = TravelCost[Vnum,int(Cargo[Vcalls[0], 1] - 1), int(Cargo[Vcalls[1], 0] - 1)]
        CostInPorts = PortCost[Vnum, Vcalls[0]]
        return sum ([FirstVisitCost, RouteTravelCost, CostInPorts])
    elif p_ind == 0 and d_ind == len(Vcalls)-1:
        FirstVisitCost_diff = FirstTravelCost[Vnum, int(Cargo[Vcalls[0], 0] - 1)] - FirstTravelCost[Vnum, int(Cargo[Vcalls_aft[0], 0] - 1)]
        Vcalls_changes = Vcalls[p_ind:p_ind+2] + Vcalls[d_ind-1:d_ind+2]
        CostInPorts = PortCost[Vnum, Vcalls_changes[0]]
        RouteTravelCost_diff = TravelCost[Vnum,int(Cargo[Vcalls_changes[0], 0] - 1), int(Cargo[Vcalls_changes[1], 0] - 1)] +\
            TravelCost[Vnum,int(Cargo[Vcalls_changes[2], 1] - 1), int(Cargo[Vcalls_changes[3], 1] - 1)]
    else:
        FirstVisitCost_diff = FirstTravelCost[Vnum, int(Cargo[Vcalls[0], 0] - 1)] - FirstTravelCost[Vnum, int(Cargo[Vcalls_aft[0], 0] - 1)]
        CostInPorts = PortCost[Vnum, Vcalls[p_ind]]
        PortIndex = PortInd[Vnum]
        RouteTravelCost_diff = TravelCost[Vnum, PortIndex[p_ind], PortIndex[p_ind+1]] + \
                (TravelCost[Vnum, PortIndex[d_ind-1], PortIndex[d_ind]] if d_ind -1 != p_ind else 0) + \
                    (TravelCost[Vnum, PortIndex[p_ind-1], PortIndex[p_ind]]if p_ind != 0 else 0) +\
                        (TravelCost[Vnum, PortIndex[d_ind], PortIndex[d_ind+1]]if d_ind != len(Vcalls)-1 else 0)-\
                     (TravelCost[Vnum, PortIndex[p_ind-1], PortIndex[p_ind+1]]if p_ind != 0 and p_ind+1 != d_ind else TravelCost[Vnum, PortIndex[p_ind-1], PortIndex[p_ind+2]] if p_ind != 0 and d_ind != len(Vcalls)-1 else 0)+\
                    (- TravelCost[Vnum, PortIndex[d_ind-1], PortIndex[d_ind+1]] if d_ind != len(Vcalls)-1 and d_ind-1 != p_ind else 0)
        # if p_ind == 0:
#             FirstVisitCost_diff = FirstTravelCost[Vnum, int(Cargo[Vcalls[0], 0] - 1)] - FirstTravelCost[Vnum, int(Cargo[Vcalls_aft[0], 0] - 1)]
#             Vcalls_changes = Vcalls[p_ind:p_ind+2] + Vcalls[d_ind-1:d_ind+2]
#             CostInPorts = PortCost[Vnum, Vcalls_changes[0]]
#             if len(Vcalls_changes) == 4:
#                 RouteTravelCost_diff = TravelCost[Vnum,int(Cargo[Vcalls_changes[0], 1] - 1), int(Cargo[Vcalls_changes[1], 0] - 1)] +\
#                     TravelCost[Vnum,int(Cargo[Vcalls_changes[2], 1] - 1), int(Cargo[Vcalls_changes[3], 0] - 1)]
#             else:
#                 RouteTravelCost_diff = TravelCost[Vnum,int(Cargo[Vcalls_changes[0], 1] - 1), int(Cargo[Vcalls_changes[1], 0] - 1)] +\
#                     TravelCost[Vnum,int(Cargo[Vcalls_changes[2], 1] - 1), int(Cargo[Vcalls_changes[3], 0] - 1)] + \
#                         TravelCost[Vnum,int(Cargo[Vcalls_changes[3], 1] - 1), int(Cargo[Vcalls_changes[4], 0] - 1)] -\
#                             TravelCost[Vnum,int(Cargo[Vcalls_changes[2], 1] - 1), int(Cargo[Vcalls_changes[4], 0] - 1)]
#         elif d_ind == len(Vcalls)-1:
#             FirstVisitCost_diff = 0
#             Vcalls_changes = Vcalls[p_ind-1:p_ind+2] + Vcalls[d_ind-1:d_ind+2]
#             CostInPorts = PortCost[Vnum, Vcalls_changes[-1]]
#             RouteTravelCost_diff = TravelCost[Vnum,int(Cargo[Vcalls_changes[0], 1] - 1), int(Cargo[Vcalls_changes[1], 0] - 1)] +\
#                 TravelCost[Vnum,int(Cargo[Vcalls_changes[1], 1] - 1), int(Cargo[Vcalls_changes[2], 0] - 1)] + \
#                     TravelCost[Vnum,int(Cargo[Vcalls_changes[3], 1] - 1), int(Cargo[Vcalls_changes[4], 0] - 1)] -\
#                         TravelCost[Vnum,int(Cargo[Vcalls_changes[0], 1] - 1), int(Cargo[Vcalls_changes[2], 0] - 1)]
#         else:
#             FirstVisitCost_diff = 0
#             Vcalls_changes = Vcalls[p_ind-1:p_ind+2] + Vcalls[d_ind-1:d_ind+2]
#             CostInPorts = PortCost[Vnum, Vcalls_changes[1]]
#             RouteTravelCost_diff = TravelCost[Vnum,int(Cargo[Vcalls_changes[0], 1] - 1), int(Cargo[Vcalls_changes[1], 0] - 1)] +\
#                 TravelCost[Vnum,int(Cargo[Vcalls_changes[1], 1] - 1), int(Cargo[Vcalls_changes[2], 0] - 1)] + \
#                     TravelCost[Vnum,int(Cargo[Vcalls_changes[3], 1] - 1), int(Cargo[Vcalls_changes[4], 0] - 1)] + \
#                         TravelCost[Vnum,int(Cargo[Vcalls_changes[4], 1] - 1), int(Cargo[Vcalls_changes[5], 0] - 1)]-\
#                             TravelCost[Vnum,int(Cargo[Vcalls_changes[0], 1] - 1), int(Cargo[Vcalls_changes[2], 0] - 1)]-\
#                                 TravelCost[Vnum,int(Cargo[Vcalls_changes[3], 1] - 1), int(Cargo[Vcalls_changes[5], 0] - 1)]
    return sum([FirstVisitCost_diff, RouteTravelCost_diff, CostInPorts])


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

    Vcalls = [Vcalls[i]-1 for i in range(len(Vcalls))]
    call = Vcalls[p_ind]
    
    NoDoubleCallOnVehicle = len(Vcalls)
    LoadSize, Timewindows, PortIndex, LU_Time = features
    
    currentTime = 0
    
    LoadSize[Vnum][p_ind] = Cargo[call, 2]
    LoadSize[Vnum][d_ind] = -Cargo[call, 2]
    
    Timewindows[Vnum][0][p_ind] = Cargo[call, 4]
    Timewindows[Vnum][0][d_ind] = Cargo[call, 6]
    Timewindows[Vnum][1][p_ind] = Cargo[call, 5]
    Timewindows[Vnum][1][d_ind] = Cargo[call, 7]
    
    PortIndex[Vnum][p_ind] = Cargo[call, 0].astype(int) - 1
    PortIndex[Vnum][d_ind] = Cargo[call, 1].astype(int) - 1 
    
    LU_Time[Vnum][p_ind] = LoadingTime[Vnum, call]
    LU_Time[Vnum][d_ind] = UnloadingTime[Vnum, call]
    
    features = [LoadSize, Timewindows, PortIndex, LU_Time]
    
    Diag_cost = TravelCost[Vnum, PortIndex[Vnum][:-1], PortIndex[Vnum][1:]]
    
    FirstVisitCost = FirstTravelCost[Vnum, int(Cargo[Vcalls[0], 0] - 1)]
    RouteTravelCost = sum(np.hstack((FirstVisitCost, Diag_cost)))
    CostInPorts = sum(PortCost[Vnum, Vcalls]) / 2

    Diag = TravelTime[Vnum, PortIndex[Vnum][:-1], PortIndex[Vnum][1:]]
    FirstVisitTime = FirstTravelTime[Vnum, int(Cargo[Vcalls[0], 0] - 1)]
    
    RouteTravelTime = np.hstack((FirstVisitTime, Diag.flatten()))
    
    ArriveTime = [0 for i in range(NoDoubleCallOnVehicle)]
    LS_cumsum = np.cumsum(LoadSize[Vnum])
    for j in range(NoDoubleCallOnVehicle):
        if VesselCapacity[Vnum] - LS_cumsum[j] < 0:
            return False, 'Capacity exceeded at call {}'.format(j), np.inf, features
        ArriveTime[j] = max(currentTime + RouteTravelTime[j], Timewindows[Vnum][0][j])
        if ArriveTime[j] > Timewindows[Vnum][1][j]:
            return False, 'Time window exceeded at call {}'.format(j), np.inf, features
        currentTime = ArriveTime[j] + LU_Time[Vnum][j]
    return True, 'Feasible', CostInPorts + RouteTravelCost, features

def update_features_v(features, v, new_features):
    LoadSize, Timewindows, PortIndex, LU_Time = features
    new_LoadSize, new_Timewindows, new_PortIndex, new_LU_Time = new_features
    LoadSize[v] = new_LoadSize[v]
    Timewindows[v] = new_Timewindows[v]
    PortIndex[v] = new_PortIndex[v]
    LU_Time[v] = new_LU_Time[v]
    return [LoadSize, Timewindows, PortIndex, LU_Time ]

def copy_features(features):
    copied_features = []
    for feature in features:
        feature_copy = [sub[:] for sub in feature]
        copied_features.append(feature_copy)
    return copied_features

def copy_costs(costs):
    return [sub for sub in costs]

def findBestPosForDel(call, Vnum, Vcalls, pos, features, prob):
    PickIndex = pos
    best_cost = np.inf
    i_best = pos
    for i in range(PickIndex+1, len(Vcalls)+1):
        new_sol = Vcalls[:i] + [call] + Vcalls[i:]
        updated_features = features_insert(copy_features(features), Vnum, PickIndex, i)
        
        new_feasibility, new_c, new_cost, new_features = cap_TW_4V(Vnum, new_sol, updated_features, PickIndex, i, prob)
        if new_c=='Feasible':
            
            # if i == len(Vcalls):
            #     return i, new_cost, new_feasibility, new_c, new_features
            if new_cost < best_cost:
                c = new_c
                feasibility = new_feasibility
                best_features = update_features_v(copy_features(features), Vnum, new_features)
                best_cost = new_cost
                i_best = i
        else:
            if (int(new_c[-1]) == PickIndex or int(new_c[-1]) == i) or \
                (int(new_c[-1]) > PickIndex and int(new_c[-1]) < i):
                break
    try:
        return i_best, best_cost, feasibility, c, best_features
    except:
        return i, new_cost, new_feasibility, new_c, new_features

def features_delete(features, cur_v, p_ind, d_ind):
    LoadSize, Timewindows, PortIndex, LU_Time = features
    LoadSize[cur_v] = LoadSize[cur_v][:p_ind] + LoadSize[cur_v][p_ind+1:d_ind] + LoadSize[cur_v][d_ind+1:]
    Timewindows[cur_v][0] = Timewindows[cur_v][0][:p_ind] + Timewindows[cur_v][0][p_ind+1:d_ind] + Timewindows[cur_v][0][d_ind+1:]
    Timewindows[cur_v][1] = Timewindows[cur_v][1][:p_ind] + Timewindows[cur_v][1][p_ind+1:d_ind] + Timewindows[cur_v][1][d_ind+1:]
    
    PortIndex[cur_v] = PortIndex[cur_v][:p_ind] + PortIndex[cur_v][p_ind+1:d_ind] + PortIndex[cur_v][d_ind+1:]
    LU_Time[cur_v] = LU_Time[cur_v][:p_ind] + LU_Time[cur_v][p_ind+1:d_ind] + LU_Time[cur_v][d_ind+1:]
    return [LoadSize, Timewindows, PortIndex, LU_Time]

def features_insert(features, cur_v, p_ind, d_ind):
    LoadSize, Timewindows, PortIndex, LU_Time = features
    d_ind -= 1
    LoadSize[cur_v] = LoadSize[cur_v][:p_ind] + [0] + LoadSize[cur_v][p_ind:d_ind] + [0] + LoadSize[cur_v][d_ind:]
    Timewindows[cur_v][0] = Timewindows[cur_v][0][:p_ind] + [0] + Timewindows[cur_v][0][p_ind:d_ind] + [0] + Timewindows[cur_v][0][d_ind:]
    Timewindows[cur_v][1] = Timewindows[cur_v][1][:p_ind] + [0] + Timewindows[cur_v][1][p_ind:d_ind] + [0] + Timewindows[cur_v][1][d_ind:]
    PortIndex[cur_v] = PortIndex[cur_v][:p_ind] + [0] + PortIndex[cur_v][p_ind:d_ind] + [0] + PortIndex[cur_v][d_ind:]
    LU_Time[cur_v] = LU_Time[cur_v][:p_ind] + [0] + LU_Time[cur_v][p_ind:d_ind] + [0] + LU_Time[cur_v][d_ind:]
    return [LoadSize, Timewindows, PortIndex, LU_Time]