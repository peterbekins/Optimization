from ortools.sat.python import cp_model
import numpy as np
import matplotlib.pyplot as plt
import time


################################################################################
# Loose ends and sundries

# timer to measure computational time
time_start = time.perf_counter()

# Create the model
rotation_model = cp_model.CpModel()

# Other variables

residents = 28
blocks = 10
clinics = 7

# indices to subset for year, S_0 = 1st year, etc
years = [[0,1,2,3,4,5,6],[7,8,9,10,11,12,13],[14,15,16,17,18,19,20],[21,22,23,24,25,26,27]]


# Initialize vacation preferences
vac_pref = np.random.randint(low = 1, high = 4, size = (residents, blocks)) 
        


################################################################################
# 1. Decision Variables
# a. x_rbc is binary, 1 = resident r is scheduled for clinic c during block b
x = {}
for r in range(residents):
    for b in range(blocks):
        for c in range(clinics):
            x[(r,b,c)] = rotation_model.NewBoolVar('x_%i%i%i' % (r, b, c))

# b. y_rbc is binary, 1 = resident r is scheduled for vacation during clinic c in block b
y = {}
for r in range(residents):
    for b in range(blocks):
        for c in range(clinics):
            y[(r,b,c)] = rotation_model.NewBoolVar('y_%i%i%i' % (r, b, c))



################################################################################
# 2. Constraints
# a. Every resident must be working in exactly one clinic during every block
for r in range(residents):
    for b in range(blocks):
        rotation_model.Add(sum(x[(r,b,c)] for c in range(clinics)) == 1)

# b. Every resident must work in each clinic at least once but at most twice       
for r in range(residents):
    for c in range(clinics):
        rotation_model.Add(sum(x[(r,b,c)] for b in range(blocks)) >= 1) 
        rotation_model.Add(sum(x[(r,b,c)] for b in range(blocks)) <= 2)

# c. Each clinic must have one resident from each year
for b in range(blocks):
    for c in range(clinics):
        for year in years:
            rotation_model.Add(sum(x[(y,b,c)] for y in year) == 1) 

# d. Each resident must take four vacations
for r in range(residents):
   rotation_model.Add(sum(y[(r,b,c)] 
            for b in range(blocks)
            for c in range(clinics)) == 4)

# e. Hard clinics (5-7) should not be back-to-back
for r in range(residents):
    for b in range(blocks-1):
        rotation_model.Add((sum(y[(r,b,c)] for c in range(4,7))) + 
                           (sum(y[(r,b+1,c)] for c in range(4,7))) <= 1)

# f. Vacations can only occur during clinics 1-4
for r in range(residents):
    rotation_model.Add(sum(y[(r,b,c)] 
                            for b in range(blocks)
                            for c in range(4,7)) == 0)

# g. Vacations can only occur in a clinic for which the resident is scheduled    
for r in range(residents):
    for b in range(blocks):
        for c in range(clinics):
            rotation_model.Add(y[(r,b,c)] <= x[(r,b,c)])

# h. Only two residents can take vacation within the same clinic in the same block
for b in range(blocks):
    for c in range(clinics):
        rotation_model.Add(sum(y[(r,b,c)] for r in range(residents)) <= 2)


################################################################################
# 3. Objective Function
# minimize vac pref * actual for each resident in each block
rotation_model.Minimize(
        sum(vac_pref[r][b] * y[(r,b,c)] 
            for r in range(residents)
            for b in range(blocks)
            for c in range(clinics)
            )) 



################################################################################
# 4. Solver
solver = cp_model.CpSolver()
status = solver.Solve(rotation_model)


################################################################################
# 5. Grid to show preference matrix


for r in range(residents):
    for b in range(blocks):
        print(vac_pref[r][b], end = " ")
    print("")


################################################################################
# 6. Grid to show rotations and clinics
#     for b in range(blocks):
#         for c in range(clinics):
#             print(solver.Value(y[(r, b, c)]), end = " ")
#         print(" | ", end = " ")
#     print("")

print("Solution = ", solver.ObjectiveValue())

time_elapsed = (time.perf_counter() - time_start)
print(status)
print("This run took ", time_elapsed)