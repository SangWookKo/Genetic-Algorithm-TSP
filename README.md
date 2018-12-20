# Genetic-Algorithm-TSP

### Genetic Algorithm based on TSP
### Selection Operator ; Roulette Wheel Selection
### Crossover Operator ; Partially Mapped Crossover
### Mutation Operator ; Insertion Mutation

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from pylab import legend

#%matplotlib inline

################################################################################################################################################################

# Probability of operators#

Crossover_Probability = 0.7
Mutation_Probability = 0.002

## Defining Number of city & Population size
popSize = 100 #100 #200 # 350
'''
city edition
'''
city = 48

################################################################################################################################################################

## 1. load file & check distance matrix 
test1 = pd.read_csv('48city.txt', sep = ",", header = None)
'''
Delete last column & Later i will upload distance matrix with txt file.
'''
test1 = test1.drop(48,1)

'''
change city index
'''
test1.index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]#,49,50]#,51]
test1.columns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]#,49,50]#,51]

################################################################################################################################################################

## Calculate Start time#

Start_Time = time.time()

########################################################################################################################
## Initialize initial population#
########################################################################################################################

Init_pop = [x for x in range(popSize)]

for i in range(popSize):
    Init_pop[i] = random.sample(range(1,city+1), city)
    #Init_pop[i].append(Init_pop[i][0])   <<<< This code means that return to origin city. 
    
########################################################################################################################
## print Initial Solution, New generated offspring becomes New Population
########################################################################################################################

Population = list(range(popSize))

for i in range(popSize):
    Population[i] = list(range(city))

for i in range(popSize) :
    for j in range(city) :
        Population[i][j] = Init_pop[i][j]
        
Pop_New = []
Solution_Sets = []
Best_Solution = []
Changed_Chromosome = []
Second_Changed_Chromosome = []
Average_Solution = []
Dist_Sum = []
Eval_New_Space = []
Pop_New_Space = []
Dummy_Space_New = []
Dummy_Space_Second = []
t = 0

print("Initial Solution ", Init_pop)

while t < 5000:

    
########################################################################################################################
## Evaluate Initial solution #
########################################################################################################################

    print(Population)
    Eval_pop = [x for x in range(popSize)]

    for i in range(popSize):
        Eval_pop[i] = list(range(city))
    
    for i in range(popSize):
        for j in range(city):
            Eval_pop[i][j] = Population[i][j]

    for i in range(popSize):
        Eval_pop[i].append(Eval_pop[i][0])

    city_dist = []

    Dummy_label = [x for x in range(0, (city*popSize)+city, city )]
    city_sum_dist = [x for x in range(popSize)]

    for i in range(popSize):
        for j in range(city):
        
            city_dist.append(test1[Eval_pop[i][j]][Eval_pop[i][j+1]])

    for i in range(popSize):
        city_sum_dist[i] = sum(city_dist[Dummy_label[i]:Dummy_label[i+1]])

    print("Evaluate city's distance", city_sum_dist)
    
    total_fitness = sum(city_sum_dist)

    Average_fitness = sum(city_sum_dist) / len(city_sum_dist)
    
    Average_Solution.append(Average_fitness)
    
    Solution_Sets.append(total_fitness)
    
    #Sorted_Each_Solution =  sorted(city_sum_dist)
    
    Best_Solution.append(city_sum_dist)#append(Sorted_Each_Solution)
    
########################################################################################################################
## Roulette Wheel Selection #
########################################################################################################################
    
    Sample_TEST = [ (1000/f)**6.5 for f in city_sum_dist] # You can fix number 6.5
    
    fit_Sample = sum(Sample_TEST)
    
    Roulette_Sample = [ f/fit_Sample for f in Sample_TEST ] 
    
    probs = [sum(Roulette_Sample[:i+1]) for i in range(len(Roulette_Sample))] # Cummulative Summation #
    
    new_population = []
    new_population_sample = [x for x in range(popSize)]

    for n in range(popSize):
        r= random.random()
        for i in range(popSize):
            if r <= probs[i] :
                new_population.append(Population[i])
            
                break
            
    for i in range(popSize):
        new_population[i] = new_population[i][:city]      

########################################################################################################################
## Crossover based on PMX method #
########################################################################################################################

    r = []
    ind_r = []
    ind_r2 = []
    cross_op = []
    parent_ind = []
    no_crossover = []

    for i in range(popSize):
        r  = random.random()
        if r < Crossover_Probability :
            ind_r.append(new_population[i])
            parent_ind.append(i)
    if len(ind_r) <= 1 :
        no_crossover = new_population
        print('You don't need to crossover')
        new_population = no_crossover
    else:
        if len(ind_r) % 2 == 0:
            print('Even number crossover')
            cross_op = ind_r
        else:
            print('Odd number crossover')
            cross_op = ind_r
            del cross_op[len(cross_op)-1]
            del parent_ind[len(parent_ind)-1]       

########################################################################################################################
## prevent origin chromosome(below code) to change #
########################################################################################################################

    origin = list(range(len(cross_op)))
    for i in range(len(cross_op)):
        origin[i] = list(range(city))
    
    for i in range(len(cross_op)):
        for j in range(city):
            origin[i][j] = cross_op[i][j]
        
########################################################################################################################
## Setting Crossover Two-Point; Points are Randomly selected#
########################################################################################################################

    Cross_Range = []
    for i in range(1) :
        Cross_Range = random.sample(range(2,city-1),2)
        if Cross_Range[0] >= Cross_Range[1] :
            Cross_Range.reverse()
        if Cross_Range[0] == Cross_Range[1] - 1 :
            Cross_Range[0] = Cross_Range[0] - 1
            if Cross_Range[0] == 2:
                Cross_Range[1] = Cross_Range[1] + 1
            if Cross_Range[0] == 1:
                Cross_Range[0] = Cross_Range[0] + 1
                Cross_Range[1] = Cross_Range[1] + 1  

    print("First Cut-Point", Cross_Range[0])
    print("Second Cut-Point", Cross_Range[1])

    PMX_Sample = []
    PMX_Mapped = []

    if len(cross_op) > 0 :
    
        r = random.random() # Randomly select Cutting points
    
        PMX_Sample = [x for x in range(len(cross_op))]
    
        for i in range(len(cross_op)) : 
            PMX_Sample[i] = cross_op[i]
    
            PMX_Mapped = PMX_Sample
        
    print(PMX_Mapped)

    PMX_op = [x for x in range(len(PMX_Sample))]

    for i in range(len(PMX_Sample)) :
        PMX_op[i] = PMX_Mapped[i][Cross_Range[0]:Cross_Range[1]]

    #print(PMX_op)
    PMX_Mapped

########################################################################################################################
## Change the order of Operator #
########################################################################################################################

    Offspring_op = []

    for i in range(len(PMX_Mapped)) :
        if i % 2 == 1 :
            Offspring_op.append(PMX_op[i-1])
        if i % 2 == 0 :
            Offspring_op.append(PMX_op[i+1])
    

    PMX_Mapped_second = PMX_Mapped

    for i in range(len(PMX_op)) :
        PMX_Mapped_second[i][Cross_Range[0]:Cross_Range[1]] = Offspring_op[i]

    print(" Second Offspring_op ", Offspring_op)
    print(" Second PMX_op ", PMX_op)
    
########################################################################################################################    
# Very Important notion - PMX operator' core code#
########################################################################################################################

    TEST_Sample1 = list(range(len(PMX_Mapped_second)))
    for i in range(len(PMX_Mapped_second)):
        TEST_Sample1[i] = list(range(city))

    TEST_Sample2 = []
    TEST_Sample3 = []
    TEST_Sample4 = []
    TEST_Sample5 = []

    for i in range(len(PMX_Mapped_second)):
        v = []
        for j in range(city):
            TEST_Sample1[i][j] = PMX_Mapped_second[i][j]
            if j in range(Cross_Range[0],Cross_Range[1]):
                continue
           
            if TEST_Sample1[i][j] in Offspring_op[i]:
               
                TEST_Sample3.append(TEST_Sample1[i][j])
                v.append(TEST_Sample1[i][j])
            
        TEST_Sample2.append(v)
                

    for i in range(len(TEST_Sample2)):
        if i % 2 ==  0 :
            TEST_Sample4.append(TEST_Sample2[i+1])
        if i % 2 == 1 :
            TEST_Sample4.append(TEST_Sample2[i-1])

    for i in range(len(TEST_Sample4)) :
        TEST_Sample5 = TEST_Sample5 + TEST_Sample4[i]

    print(" Second PMX_Mapped_second; ",PMX_Mapped_second)
    print(" TEST_Sample1 ",TEST_Sample1) 
    print(" TEST_Sample2 ",TEST_Sample2)
    print(" TEST_Sample3 ",TEST_Sample3)
    print(" TEST_Sample4 ",TEST_Sample4)
    print(" TEST_Sample5 ",TEST_Sample5)
    Second_Changed_Chromosome.append(PMX_Mapped_second)
    v = 0 
    for i in range(len(PMX_Mapped_second)):
        for j in range(city):
            TEST_Sample1[i][j] = PMX_Mapped_second[i][j]
        
            if j in range(Cross_Range[0], Cross_Range[1]) :
                continue
            if TEST_Sample1[i][j] in TEST_Sample2[i]:
                TEST_Sample1[i][j] = TEST_Sample5[v]
                v = v + 1 
                
                if len(parent_ind) >= 1 :
                    for q in range(len(parent_ind)) :
                        new_population[parent_ind[q]] = TEST_Sample1[q]
                        
    print("Changed Test_Sample1", TEST_Sample1)
    Changed_Chromosome.append(TEST_Sample1)

###########################################################################################################################        
######################################## Generate New Population ##########################################################           
######################################## Insertion Mutation ###############################################################
###########################################################################################################################

    mutation_op = [x for x in range(len(new_population))]

    for i in range(len(new_population)):
        mutation_op[i] = list(range(city))

# Below Condition checks whether Mutation or not#

    for i in range(len(new_population)):
        for j in range(city):
            mutation_op[i][j] = random.random()
        
        # Mutation이 new_population 안에 있는지 확인                        

            if mutation_op[i][j] < Mutation_Probability:
            
           
                # Random함수를 생성하고, 이 Random함수는 해당하는 Chromosome의 위치를 찾는 것이다.            

                
                new_population[i].insert( random.randrange(city),  new_population[i].pop(j) )
                

    print(t," th Solution" ,new_population )
    Pop_New.append(new_population)    
    #Dist_Sum.append(Best_Solution)
    print(t)
    
    Population = Pop_New[t]
    PMX_Mapped_second = []
    Offspring_op = []
    PMX_op = []
    PMX_Mapped = []
    
    
    t = t + 1
################################################################################################################################################################
for i in range(len(Best_Solution)) :
    for j in range(popSize) :
         Eval_New_Space.append(Best_Solution[i][j])

Sorting_Method = sorted(Eval_New_Space)
  
for i in range(len(Pop_New)) :
    for j in range(popSize):
        Pop_New_Space.append(Pop_New[i][j])

#print(Pop_New_Space[Eval_New_Space.index(Sorting_Method[0])])
################################################################################################################################################################
# Graph  #
################################################################################################################################################################

x_label = [ x for x in range(t)]
plt.plot(x_label, Solution_Sets)#, marker="o")
plt.title('Comparison of Total Solutions')
plt.xlabel('Generation')
plt.ylabel('Solution')
plt.legend(['Total Solution'], loc ='best')
plt.show()

################################################################################################################################################################
#plt.plot(x_label, Average_Solution)#, x_label, Best_Solution)
#plt.title(' Comparison of Average & Best Solutions')
#plt.xlabel('Generation')
#plt.ylabel('Solution')
#plt.legend(['Average Solution', 'Best Solution'], loc = 'best')
################################################################################################################################################################
# Best Solution #
################################################################################################################################################################

print(" 1. Best Solution = %s ------------------------------------- " %(Sorting_Method[0] ))
################################################################################################################################################################

print(" 2. Best Chromosome  ", (Pop_New_Space[Eval_New_Space.index(Sorting_Method[0])-popSize]))
################################################################################################################################################################
x_label = [ x for x in range(len(Pop_New_Space))]
plt.plot(x_label, Eval_New_Space)#, marker="o")
plt.title('Comparison of Best Solutions')
plt.xlabel('Sets Size')
plt.ylabel('Solution')
plt.legend(['Total Solution'], loc ='best')


################################################################################################################################################################
## End calculation time#
################################################################################################################################################################
End_Time = time.time()

Elapsed = End_Time - Start_Time
print(" 3. Total calculation time = %s seconds --------------- " %(Elapsed))
