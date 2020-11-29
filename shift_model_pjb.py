from ortools.sat.python import cp_model
import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(441)


# timer to measure computational time
time_start = time.perf_counter()


# Variable definitions
residents = 4
weeks = 4
shifts = weeks * 21
shift_requests = []

# initialize shift requests with random variable for now
# this algorithm has each resident rate each shift 1-3
# I choose without replacement to ensure that the requests are similar
# each week, each resident will have 7 shifts rated 3, etc. 
week_ratings = [3,3,3,3,3,3,3,2,2,2,2,2,2,2,1,1,1,1,1,1,1]

for r in range(residents):
    week_requests = []
    for w in range(weeks):
        week_requests.append(np.random.choice(week_ratings, 21, replace=False))
    shift_requests.append(np.hstack(week_requests))

# vacation requests, 1 = available to work 0 = on vacation
v = []
v.append(np.tile(1, shifts))
v.append(np.tile(1, shifts))
v.append(np.tile(1, shifts))
v.append([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
          1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
          1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

# Set min/max for resident per shift
min_per_shift = 1
max_per_shift = 3

# Set max for resident total 
max_per_week = weeks * 10
min_per_week = weeks * 6

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

# c. z_rwd is binary, indicator variable turns on if resident r is scheduled on week w day d
    z = {}
    for r in range(residents):
        for w in range(0, shifts, 21):
            for d in range(w, w + 21, 3):
                z[(r,w,d)] = shift_model.NewBoolVar('z_r%iw%id%i' % (r,w,d))

# 2. Contraints
    
# a. Min and max residents per shift
    
for s in range(shifts):
    shift_model.Add(sum(x[(r,s)] for r in range(residents)) >= min_per_shift)
    shift_model.Add(sum(x[(r,s)] for r in range(residents)) <= max_per_shift)


# b. Max shifts per week capped at 10 per resident but averaged over 4 weeks
for r in range(residents):
    shift_model.Add(sum(x[(r,s)] for s in range(shifts)) <= max_per_week)
    shift_model.Add(sum(x[(r,s)] for s in range(shifts)) >= min_per_week)

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

    # This also checks the edge so 4 shifts can't be scheduled at the end of the period
    edge = x[(r, shifts-1)] + x[(r, shifts-2)]+ x[(r, shifts-3)] + x[(r, shifts-4)] 
    shift_model.Add(edge < 4)

# d. Resident must have one full day off per week 
for r in range(residents):
    for w in range(0, shifts, 21):
        weekly_total = 0
        for d in range(w, w+21, 3):
            daily_total = x[(r, d)] + x[(r, d+1)] + x[(r,d+2)]
                
            #Turns z on if r works day d
            shift_model.Add(daily_total >= 1 - (1 - z[(r,w,d)]) * 100)   
            shift_model.Add(daily_total < 1 + z[(r,w,d)] * 100)
            weekly_total += z[(r,w,d)]
        #can work max 6 days a week
        shift_model.Add(weekly_total <= 6)

# e. Block vacation
for r in range(residents):
    for s in range(shifts):
        shift_model.Add(x[(r,s)] <= v[r][s])
        
# 3. Objective function
# Maximize the number of requested shifts assigned to each resident
# Need to look into balancing across residents as well
shift_model.Minimize(
        sum(shift_requests[r][s] * x[(r,s)] 
            for r in range(residents)
            for s in range(shifts)))

# 4. Solver
solver = cp_model.CpSolver()
printer = cp_model.ObjectiveSolutionPrinter()
status = solver.SolveWithSolutionCallback(shift_model, printer)

if status == cp_model.FEASIBLE:
    print("Feasible!")
    print("Solution = ", solver.ObjectiveValue())
else:
    print("no solution")

# 5. Grid plot of to show preference matrix
plt.figure(figsize=(8,3))
plt.imshow(shift_requests, cmap="Oranges",aspect = 3)
plt.axvline(20.5, color='black')
plt.axvline(41.5, color='black')
plt.axvline(62.5, color='black')

# add borders
for res in range(residents):
    for shift in range(shifts):
        r = plt.Rectangle((shift-0.5,res-0.5), 1,1, facecolor="none", edgecolor="white", linewidth=1)
        plt.gca().add_patch(r)

for i in range(residents):
    for j in range(shifts):
        text = plt.text(j, i, shift_requests[i][j],
                       ha="center", va="center", color="w", fontsize=6)

week_set = [10.5,31.5,52.5,73.5]
plt.tick_params(axis='both', bottom=False)
plt.xticks(week_set,['Week 1', 'Week 2','Week 3','Week 4'],fontsize=10)
plt.yticks([0,1,2,3],['R1','R2','R3','R4'],fontsize=8)
plt.tick_params(axis = "both", which = "both", bottom = False, left = False)

# 6. Grid Plot of schedule

shift_matrix = []
value_matrix = []
for r in range(residents):
    shift_result = []
    value = 0
    tot_shifts = 0
    for s in range(shifts):
        if solver.Value(x[(r, s)]) == 1:
            value = value + shift_requests[r][s]
            tot_shifts = tot_shifts + 1
            if shift_requests[r][s] < 3:
                shift_result.append((128,128,128)) # dark gray for shift on
            else:
                shift_result.append((204,51,0)) # reddish for on but didn't request
        else:
            shift_result.append((224,224,224)) # light gray for shift off
    shift_matrix.append(shift_result)
    value_matrix.append((r,tot_shifts, value))

plt.figure(figsize=(8,3))

plt.imshow(shift_matrix, aspect = 3)
plt.axvline(20.5, color='black')
plt.axvline(41.5, color='black')
plt.axvline(62.5, color='black')


# add borders
for res in range(residents):
    for shift in range(shifts):
        r = plt.Rectangle((shift-0.5,res-0.5), 1,1, facecolor="none", edgecolor="white", linewidth=1)
        plt.gca().add_patch(r)


week_set = [10.5,31.5,52.5,73.5]
plt.tick_params(axis='both', bottom=False)
plt.xticks(week_set,['Week 1', 'Week 2','Week 3','Week 4'],fontsize=10)
plt.yticks([0,1,2,3],['R1','R2','R3','R4'],fontsize=8)
plt.tick_params(axis = "both", which = "both", bottom = False, left = False)

# 7. Some Summary Diagnostics
sum_value = 0
for row in value_matrix:
    print("Resident ", row[0], "works ", row[1], " shifts at a value of ", row[2])
    sum_value = sum_value + row[2]
print("total value was ", sum_value)

# Output for computational time
time_elapsed = (time.perf_counter() - time_start)
print(weeks, " weeks takes ", time_elapsed)
