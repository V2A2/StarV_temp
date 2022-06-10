#1: PosLin – Yuntao Li – 10/1/2021 - 5/31/2022:  
PosLin class contains method for reachability analysis for Layer with ReLU activation function (ReLU operator in NN)

#2: Methods:
1. evaluate(x):
2. stepReach(*args):
3. stepReachMultipleInputs(*args):
4. reach_star_exact(*args):
5. stepReachStarApprox(I, index):
6. reach_star_approx(I):
7. stepReachZonoApprox(I, index, lb, ub):
8. reach_zono_approx(*args):
9. reach(*args):

#3: Descriptions: 
1. evaluate(x): (evaluate method and reachability analysis with stars)
2. stepReach(*args): (stepReach method, compute reachable set for a single step)
   1. @I, input set, a star (array of a single star)
   2. @index, index of current x[index] of current step,
   3. should be the number from matlab - 1
   4. @xmin, min of x[index]
   5. @xmax, max of x[index]
   6. @S, star output set
3. stepReachMultipleInputs(*args): (stepReach with multiple inputs)
   1. @I: an list of stars
   2. @index: index where stepReach is performed
   3. @option: = 'parallel' use parallel computing OR = not declare -> don't use parallel computing
4. reach_star_exact(*args): (exact reachability analysis using star)
   1. @I: star input sets
   2. @option: = 'parallel' use parallel computing OR = not declare -> don't use parallel computing
5. stepReachStarApprox(I, index): (step reach approximation using star)
   1. @I: Star set input
   2. @index: index of the neuron performing stepReach
   3. @S: star output
6. reach_star_approx(I): (over-approximate reachability analysis using Star)
   1. @I: star input set
   2. @S: star output set
7. stepReachZonoApprox(I, index, lb, ub): (step over-approximate reachability analysis using zonotope)
   1. @I: zonotope input set
   2. @lb: lower bound of input at specific neuron i
   3. @ub: lower bound of input at specfic neuron i
   4. @index: index of the neuron we want to perform stepReach
   5. @Z: zonotope output set
8. reach_zono_approx(*args): (over-approximate reachability analysis use zonotope)
   1. @I: zonotope input
   2. @Z: zonotope output
9. reach(*args): (main function for reachability analysis)
   1. @I: an array of star input sets
   2. @method: 'exact-star' or 'approx-star' or 'approx-zono'
   3. @option: = 'parallel' use parallel option OR = '' do use parallel option
   
#4: Corresponding text example:
   
#5: References:

#6: Dependency:  
Star, Zono