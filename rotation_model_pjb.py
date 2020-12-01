from ortools.sat.python import cp_model
import numpy as np
import matplotlib.pyplot as plt
import time


################################################################################
# Loose ends and sundries

np.random.seed(359)

# timer to measure computational time
time_start = time.perf_counter()

# Create the model
rotation_model = cp_model.CpModel()

# Other variables

residents = 28
blocks = 13
clinics = 7

# indices to subset for year, S_0 = 1st year, etc
years = [[0,1,2,3,4,5,6],[7,8,9,10,11,12,13],[14,15,16,17,18,19,20],[21,22,23,24,25,26,27]]


# Initialize vacation preferences
prefs = [1,1,1,1,1,1,10,10,10,10,10,10,10] 
#prefs = [1,2,3,4,4,4,4,4,4,4,4,4,4] 
#prefs = [1,2,3,4,5,6,7,8,9,10,11,12,13] 
vac_pref = []
for r in range(residents):
    vac_pref.append(np.random.choice(prefs, blocks, replace=False))
        
#vac_pref = np.random.randint(low = 1, high = 4, size = (residents, blocks)) 
        


################################################################################
# 1. Decision Variables
# a. x_rbc is binary, 1 = resident r is scheduled for clinic c during block b
x = {}
for r in range(residents):
    for b in range(blocks):
        for c in range(clinics):
            x[(r,b,c)] = rotation_model.NewBoolVar('x_r%ib%ic%i' % (r, b, c))

# b. y_rbc is binary, 1 = resident r is scheduled for vacation during clinic c in block b
y = {}
for r in range(residents):
    for b in range(blocks):
        for c in range(clinics):
            y[(r,b,c)] = rotation_model.NewBoolVar('y_r%ib%ic%i' % (r, b, c))

# c. z_mnbc is binary, 1 = resident m and resident n worked together in clinic c during block b
z = {}
for m in range(residents):
    for n in range(residents):
        for b in range(blocks):
            for c in range(clinics):
                z[(m,n,b,c)] = rotation_model.NewBoolVar('z_m%in%ib%ic%i' % (m, n, b, c))

# K = rotation_model.NewIntVar(0,15,'K')

################################################################################
# 2. Constraints
# a. Every resident must be working in exactly one clinic during every block
for r in range(residents):
    for b in range(blocks):
        rotation_model.Add(sum(x[(r,b,c)] for c in range(clinics)) == 1)

# b. Every resident must work in each clinic at least once but at most thrice       
for r in range(residents):
    for c in range(clinics):
        rotation_model.Add(sum(x[(r,b,c)] for b in range(blocks)) >= 1) 
        rotation_model.Add(sum(x[(r,b,c)] for b in range(blocks)) <= 3)

# c. Each clinic must be staffed by one resident from each year
for b in range(blocks):
    for c in range(clinics):
        for year in years:
            rotation_model.Add(sum(x[(y,b,c)] for y in year) == 1) 

# d. Each resident must take X vacations
for r in range(residents):
   rotation_model.Add(sum(y[(r,b,c)] 
            for b in range(blocks)
            for c in range(clinics)) == 3)

# e. Hard clinics (5-7) should not be done back-to-back
for r in range(residents):
    for b in range(blocks-1):
        rotation_model.Add((sum(x[(r,b,c)] for c in range(4,7))) + 
                            (sum(x[(r,b+1,c)] for c in range(4,7))) <= 1)

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

# h. Only X residents can take vacation within the same clinic in the same block
for b in range(blocks):
    for c in range(clinics):
        rotation_model.Add(sum(y[(r,b,c)] for r in range(residents)) <= 4)

# i. New Social Constraint
# first turn on z if resident i works with resident j for a certain block and clinic 
for i in range(residents):
    for j in range(residents):
        for b in range(blocks):
            for c in range(clinics):
                rotation_model.Add(z[(i,j,b,c)] + 1 >= x[(i,b,c)] + x[(j,b,c)])
                rotation_model.Add(2 * z[(i,j,b,c)] <= x[(i,b,c)] + x[(j,b,c)])

# Next ensure that each resident in a year u works at least once with each resident v from the other years
for u in years[0]:
    for v in (years[1] + years[2] + years[3]):
        rotation_model.Add(sum(z[(u,v,b,c)] 
                                for b in range(blocks) 
                                for c in range(clinics)
                                ) > 0)
        
for u in years[1]:
    for v in (years[2] + years[3]):
        rotation_model.Add(sum(z[(u,v,b,c)] 
                                for b in range(blocks) 
                                for c in range(clinics)
                                ) > 0)

for u in years[2]:
    for v in (years[3]):
        rotation_model.Add(sum(z[(u,v,b,c)] 
                                for b in range(blocks) 
                                for c in range(clinics)
                                ) > 0)

# # next set K to be the max
# for i in range(residents):
#     for j in range(i+1, residents):
#         rotation_model.Add(K >= sum(z[(i,j,b,c)] 
#                            for b in range(blocks) 
#                            for c in range(clinics)))
           

      
# j. No resident should be scheduled in the same clinic in consecutive blocks
for r in range(residents):
    for b in range(blocks-1):
        for c in range(clinics):
            rotation_model.Add(x[(r,b,c)] + x[(r,b+1,c)] <= 1)

################################################################################
# 3. Objective Function
# minimize vac pref * actual for each resident in each block
rotation_model.Minimize(
        sum(vac_pref[r][b] * y[(r,b,c)] 
            for r in range(residents)
            for b in range(blocks)
            for c in range(clinics)
            )
        )


################################################################################
# 4. Solver
solver = cp_model.CpSolver()
printer = cp_model.ObjectiveSolutionPrinter()
status = solver.SolveWithSolutionCallback(rotation_model, printer)

if status == cp_model.OPTIMAL:
    print("Optimal solution = ", solver.ObjectiveValue())
    # for r in range(residents):
    #     for b in range(blocks):
    #         for c in range(clinics):
    #             print(solver.Value(x[(r,b,c)]), end = "")
    #         print(" | ", end = " ")
    #     print("")
    
    # for r in range(residents):
    #     for b in range(blocks):
    #         print(sum(solver.Value(y[(r,b,c)]) for c in range(clinics)), end = "")
    #     print("")
elif status == cp_model.FEASIBLE:
    print("Best feasible solution = ", solver.ObjectiveValue())
else:
    print("No feasible solution")
    
################################################################################
# 5. Grid to show preference matrix


# for r in range(residents):
#     for b in range(blocks):
#         print(vac_pref[r][b], end = " ")
#     print("")


################################################################################
#7. Dan's Viz
rot_matrix = []

for r in range(residents):
    rot_results = []
    for b in range(blocks):
        vacay = 0
        for c in range(clinics):
            if solver.Value(y[(r,b,c)]) > 0:
                vacay += 1
        if vacay == 1 and vac_pref[r][b] == 1:
            rot_results.append((255,255,0))
        elif vacay ==1 and vac_pref[r][b] > 1:
            rot_results.append((255,153,51))
        else:
            rot_results.append((224,224,224))
    rot_matrix.append(rot_results)

fig, ax = plt.subplots(figsize = (24,8), dpi = 1000)
#plt.figure(figsize = (24,8), dpi = 1000)
ax.imshow(rot_matrix, aspect=0.5)


#borders
for r in range(residents):
    for b in range(blocks):
        for c in range(clinics):
            rec = plt.Rectangle((b-0.5,r-0.5),1,1,facecolor="none",edgecolor="white", 
                                linewidth=0.5)
            plt.gca().add_patch(rec)
#clinics
for r in range(residents):
    for b in range(blocks):
        for c in range(clinics):
            if solver.Value(x[(r,b,c)]) >= 1:
                text = plt.text(b,r,c+1, ha = "center", va = "center", color = "black", fontsize = 12)

# Lines to split classes of residents
ax.axhline(6.5, color='black')
ax.axhline(13.5, color='black')
ax.axhline(20.5, color='black')

# Labels
block_pos = [0,1,2,3,4,5,6,7,8,9,10,11,12]
block_label = [1,2,3,4,5,6,7,8,9,10,11,12,13]

res_label =[]
for r in range(residents):
    res_label.append('R%i' % (r+1))
ax.set_xticks(block_pos)
ax.set_xticklabels(block_label)
ax.set_xlabel("Blocks")
ax.xaxis.set_label_position('top')
ax.set_yticks(range(residents))
ax.set_yticklabels(res_label)
ax.set_ylabel("Residents")
ax.tick_params(axis = "both", which = "both", bottom = False, left = False, labelbottom=False, labeltop=True)


# 8. Dan's loop to test resident combos
for i in range(residents):
    print()
    for j in range(residents):
       works_tot = 0
       for b in range(blocks):
           for c in range(clinics):
               works_tot += solver.Value(z[(i,j,b,c)])
       print(works_tot, end=" ")

print("")    
# print("max K was ", solver.Value(K))   
################################################################################
# 6. Grid to show rotations and clinics



time_elapsed = (time.perf_counter() - time_start)
print("This run took ", time_elapsed)
