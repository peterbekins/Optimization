from ortools.sat.python import cp_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# timer to measure computational time
time_start = time.perf_counter()


# Variable definitions
residents = 4
weeks = 4
shifts = weeks * 21

# initialize shift requests with random variable for now
shift_requests = np.random.randint(2, size = (residents, shifts))

# Set min/max for resident per shift
min_per_shift = 1
max_per_shift = 2

# Set max for resident total 
max_per_week = weeks * 10

# Create the model
shift_model = cp_model.CpModel()

# 1. Decision Variables
# a. x_rs is binary, 1 = resident r is scheduled for shift s
x = {}
for r in range(residents):
    for s in range(shifts):
        x[(r,s)] = shift_model.NewBoolVar('x_r%s%i' % (r, s))

# b. y_rs is binary, indicator variable used to turn on constraint for sequential shifts
y = {}
for r in range(residents):
    for s in range(shifts):
        y[(r,s)] = shift_model.NewBoolVar('y_r%s%i' % (r, s))

# 2. Contraints
    
# a. Min and max residents per shift
    
for s in range(shifts):
    shift_model.Add(sum(x[(r,s)] for r in range(residents)) >= min_per_shift)
    shift_model.Add(sum(x[(r,s)] for r in range(residents)) <= max_per_shift)


# b. Max shifts per week capped at 10 per resident but averaged over 4 weeks
for r in range(residents):
    shift_model.Add(sum(x[(r,s)] for s in range(shifts)) <= max_per_week)


# c. Resident must have 16 hours off after three consecutive shifts
for r in range(residents):
    for s in range(shifts-4):
        pattern_1 = x[(r,s)] + x[(r,s+1)] + x[(r,s+2)]
        
        # These constraints turn the indicator y[(s)]==1 for any pattern of three straight shifts
        
        shift_model.Add(3 > pattern_1 + y[(r,s)]*1000) 
        shift_model.Add(3 <= pattern_1 + (1-y[(r,s)])*1000) 
        
        # If y[(s)] == 1, then these constraints force X[(s+3)] == 0 and X[(s+4)] == 0
        shift_model.Add(y[(r,s)] + x[(r,s+3)] <= 1) 
        shift_model.Add(y[(r,s)] + x[(r,s+4)] <= 1)


# 3. Objective function
# Maximize the number of requested shifts assigned to each resident
# Need to look into balancing across residents as well
shift_model.Maximize(
        sum(shift_requests[r][s] * x[(r,s)] 
            for r in range(residents)
            for s in range(shifts)))

# 4. Solver
solver = cp_model.CpSolver()
solver.Solve(shift_model)

# 5. Chart

r_cat = []
r_shift_num = []
r_shift_type = []
for r in range(residents):
    for s in range(shifts):
        r_cat.append('R_%i'% (r))
        r_shift_num.append(s)
        if solver.Value(x[(r, s)]) == 1:
            if shift_requests[r][s] == 1:
                r_shift_type.append(1)
            else:
                r_shift_type.append(-1)
        else:
            r_shift_type.append(0)

df = pd.DataFrame(list(zip(r_cat, r_shift_num,r_shift_type)), columns = ['Resident', 'Shift_Num','Shift_Type'])
 
sns.catplot(x='Shift_Num', y='Resident', 
            hue='Shift_Type', 
            hue_order=[1,0,-1],
            palette=['green','gray','red'],
            jitter=False,
            data=df)

# Output for computational time
time_elapsed = (time.perf_counter() - time_start)
print(weeks, " weeks takes ", time_elapsed)