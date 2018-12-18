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
'''
프로그램을 실행하기 전에 수정해야 하는 부분; Matrix에 들어가는 도시 수
'''
# 오퍼레이터에 대한 확률값 #

Crossover_Probability = 0.7
Mutation_Probability = 0.002

# 도시의 개수 및 인구 수 #

popSize = 100 #100 #200 # 350
'''
city 수정
'''
city = 48

################################################################################################################################################################

# 1. 거리 행렬 불러오기 #
test1 = pd.read_csv('48city.txt', sep = ",", header = None)
'''
마지막 열 제거
'''
test1 = test1.drop(48,1)

'''
city 인덱스 수정
'''
test1.index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]#,49,50]#,51]
test1.columns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]#,49,50]#,51]

################################################################################################################################################################

# 시작 시간 #

Start_Time = time.time()

########################################################################################################################
# 초기해 생성 #
########################################################################################################################

Init_pop = [x for x in range(popSize)]

for i in range(popSize):
    Init_pop[i] = random.sample(range(1,city+1), city)
    #Init_pop[i].append(Init_pop[i][0])   <<<< 마지막에 자기 자신으로 돌아오는 것, Evaluation 하기 위해 필요
    
########################################################################################################################
# 초기 해 출력 & 새롭게 생성된 자식들이 Population이 된다. #    
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

print("초기해 ", Init_pop)

while t < 5000:

    
########################################################################################################################
# 초기해를 평가; Evaluate Initial solution #
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

    print("거리 평가", city_sum_dist)
    
    total_fitness = sum(city_sum_dist)

    Average_fitness = sum(city_sum_dist) / len(city_sum_dist)
    
    Average_Solution.append(Average_fitness)
    
    Solution_Sets.append(total_fitness)
    
    #Sorted_Each_Solution =  sorted(city_sum_dist)
    
    Best_Solution.append(city_sum_dist)#append(Sorted_Each_Solution)
    
########################################################################################################################
# 룰랫 휠 선택; Roulette Wheel Selection #
########################################################################################################################
    
    Sample_TEST = [ (1000/f)**6.5 for f in city_sum_dist]
    
    fit_Sample = sum(Sample_TEST)
    
    Roulette_Sample = [ f/fit_Sample for f in Sample_TEST ] 
    
    probs = [sum(Roulette_Sample[:i+1]) for i in range(len(Roulette_Sample))] # 누적 합; Cummulative Summation #
    
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
# 크로스오버, Crossover based on PMX method #
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
        print('크로스오버 할 필요 없다.')
        new_population = no_crossover
    else:
        if len(ind_r) % 2 == 0:
            print('크로스 오버 짝수')
            cross_op = ind_r
        else:
            print('크로스 오버 홀수')
            cross_op = ind_r
            del cross_op[len(cross_op)-1]
            del parent_ind[len(parent_ind)-1]       

########################################################################################################################
# 밑의 식이 변경되면서 origin의 값이 변경되는 것을 방지 #
########################################################################################################################

    origin = list(range(len(cross_op)))
    for i in range(len(cross_op)):
        origin[i] = list(range(city))
    
    for i in range(len(cross_op)):
        for j in range(city):
            origin[i][j] = cross_op[i][j]
        
########################################################################################################################
# Crossover Two-Point 지정; Point를 Random하게 선정한다. #
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

    print("첫 번째 Cut-Point", Cross_Range[0])
    print("두 번째 Cut-Point", Cross_Range[1])

    PMX_Sample = []
    PMX_Mapped = []

    if len(cross_op) > 0 :
    
        r = random.random() # 랜덤하게 Cutting point 지정
    
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
# Operator의 순서를 바꿔주는 것이다. #
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

    print(" 두 번째 Offspring_op ", Offspring_op)
    print(" 두 번째 PMX_op ", PMX_op)
    
########################################################################################################################    
# Very Important notion - PMX 오퍼레이터의 핵심 코드 #
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

    print(" 두 번째 PMX_Mapped_second 는 ",PMX_Mapped_second)
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
                        
    print("바뀐 Test_Sample1", TEST_Sample1)
    Changed_Chromosome.append(TEST_Sample1)

###########################################################################################################################        
######################################## Generate New Population ##########################################################           
######################################## Insertion Mutation ###############################################################
###########################################################################################################################

    mutation_op = [x for x in range(len(new_population))]

    for i in range(len(new_population)):
        mutation_op[i] = list(range(city))

# Mutation 할 것인지 판단하는 조건 #

    for i in range(len(new_population)):
        for j in range(city):
            mutation_op[i][j] = random.random()
        
        # Mutation이 new_population 안에 있는지 확인                        

            if mutation_op[i][j] < Mutation_Probability:
            
           
                # Random함수를 생성하고, 이 Random함수는 해당하는 Chromosome의 위치를 찾는 것이다.            

                
                new_population[i].insert( random.randrange(city),  new_population[i].pop(j) )
                

    print(t," 번째 해" ,new_population )
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
# 종료 시간 #
################################################################################################################################################################
End_Time = time.time()

Elapsed = End_Time - Start_Time
print(" 3. 총 소요 시간 = %s seconds --------------- " %(Elapsed))
