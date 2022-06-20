#1: Layer – Yuntao Li – 11/15/2021 - 5/31/2022:  
LAYERS is a class that contains only reachability analysis method using stars

#2: Methods:
1. __init__();
2. evaluate(obj, x):
3. sample(obj, V):
4. reach(*args):
5. flatten(obj, reachMethod):

#3: Descriptions: 
1. __init__(obj,
            W = np.matrix([]), # weight_mat
            b = np.matrix([]), # bias vector
            f = '', # activation function
            N = 0, # number of neurons
            gamma = 0, # used only for leakReLU layer
            option = '', # parallel option, 'parallel' or '' '
            dis_opt = '', # display option, 'display' or '' '
            lp_solver = 'gurobi', # lp solver option, 'gurobi'
            relaxFactor = 0 # use only for approx-star method
            ):
2. evaluate(obj, x): # evaluation of this layer with a specific vector
3. sample(obj, V): # Evaluate the value of the layer output with a set of vertices
4. reach(*args):
    1. @I: an array of inputs (list)
    2. @method: 'exact-star' or 'approx-star' or 'approx-zono' or 'abs-dom', i.e., abstract domain (support later) or 'face-latice', i.e., face latice (support later)
    3. @option:  'parallel' use parallel computing OR '[]' or not declared -> don't use parallel computing
5. flatten(obj, reachMethod): # flattening a layer into a sequence of operation
    1. @reachMethod: reachability method
    2. @Ops: an array of operations for the reachability of the layer

#4: Corresponding text example:
   
#5: References:

#6: Dependency:  
PosLin, Operation